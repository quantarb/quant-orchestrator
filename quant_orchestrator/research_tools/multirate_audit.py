"""Polars corpus gates and event-only chronological model evaluation."""

from datetime import datetime
import hashlib
import json
from pathlib import Path

import polars as pl

from quant_orchestrator.research_tools.multirate_supervision import (
    StreamingSupervision,
    instrument_asset_groups,
)
from quant_orchestrator.platforms.ml_frameworks.torch.models.transformers.multirate.temporal_tasks import (
    ORACLE_SUPERVISED_TASK_NAMES,
    HITS_SUPERVISED_TASK_NAMES,
)

TASKS = (*ORACLE_SUPERVISED_TASK_NAMES, *HITS_SUPERVISED_TASK_NAMES)


def verify_corpus_files(root: Path, manifest: dict) -> str:
    hashes = manifest.get("input_sha256")
    if not hashes:
        raise ValueError(
            "Corpus has no input fingerprints; rebuild using the Polars corpus builder"
        )
    for name, expected in hashes.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Corpus fingerprint path escapes the corpus directory")
        with path.open("rb") as handle:
            actual = hashlib.file_digest(handle, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"Corpus input changed after assembly: {name}")
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def audit_corpus(root: Path, cutoff: str) -> dict:
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    fingerprint = verify_corpus_files(root, manifest)
    taxonomy = pl.read_csv(root / "taxonomy.csv")
    groups = instrument_asset_groups(taxonomy)
    events = pl.scan_parquet(root / "sparse_events.parquet")
    store = StreamingSupervision(events, cutoff=datetime.fromisoformat(cutoff))
    coverage = store.coverage(symbols_by_asset=groups, required_tasks=TASKS)
    binary = []
    for asset, symbols in groups.items():
        rows = store.scan.filter(pl.col("symbol").is_in(symbols))
        for task in ORACLE_SUPERVISED_TASK_NAMES:
            counts = (
                rows.select(
                    pl.col(task).count().alias("count"), pl.col(task).sum().alias("positive")
                )
                .collect(engine="streaming")
                .row(0, named=True)
            )
            if not 0 < counts["positive"] < counts["count"]:
                raise ValueError(f"{asset}/{task} lacks both positive and negative event labels")
            binary.append(dict(asset_class=asset, task=task, **counts))
    rates = {}
    for rate in ("annual", "quarterly", "daily", "sparse_events"):
        rows = pl.scan_parquet(root / f"{rate}.parquet").filter(
            pl.col("date") < datetime.fromisoformat(cutoff)
        )
        rates[rate] = (
            rows.group_by("symbol")
            .agg(
                pl.len().alias("rows"),
                pl.col("date").n_unique().alias("dates"),
                pl.col("date").min().alias("first"),
                pl.col("date").max().alias("last"),
            )
            .collect(engine="streaming")
            .to_dicts()
        )
    result = dict(
        input_fingerprint=fingerprint,
        cutoff=cutoff,
        coverage=coverage,
        oracle_binary_coverage=binary,
        rates=rates,
    )
    (root / f"audit_{cutoff}.json").write_text(json.dumps(result, indent=2, default=str))
    return result


def evaluate_predictions(root: Path, predictions: Path, *, start: str, end: str, training_cutoff: str | None = None) -> dict:
    training_cutoff = training_cutoff or start
    if datetime.fromisoformat(training_cutoff) > datetime.fromisoformat(start):
        raise ValueError("Training cutoff must not follow evaluation start")
    taxonomy = pl.read_csv(root / "taxonomy.csv").select("symbol", "issuer", "asset_class")
    scores = pl.scan_csv(predictions, try_parse_dates=True).with_columns(
        pl.col("date").cast(pl.Datetime("ns"))
    )
    scores = scores.filter(
        pl.col("date").is_between(datetime.fromisoformat(start), datetime.fromisoformat(end))
    )
    invalid = (
        scores.select(
            pl.any_horizontal(
                ~pl.col(task).is_finite() | pl.col(task).is_null() for task in TASKS
            ).sum()
        )
        .collect(engine="streaming")
        .item()
    )
    if invalid:
        raise ValueError(f"{invalid} scored rows contain invalid trading predictions")
    duplicates = (
        scores.group_by("symbol", "date")
        .len()
        .filter(pl.col("len") != 1)
        .select(pl.len())
        .collect(engine="streaming")
        .item()
    )
    if duplicates:
        raise ValueError("Duplicate instrument/date predictions")
    expected = (
        pl.scan_parquet(root / "daily.parquet")
        .select("symbol", "date")
        .filter(
            pl.col("date").is_between(datetime.fromisoformat(start), datetime.fromisoformat(end))
        )
        .unique()
    )
    missing = (
        expected.join(scores.select("symbol", "date"), on=["symbol", "date"], how="anti")
        .select(pl.len())
        .collect(engine="streaming")
        .item()
    )
    if missing:
        raise ValueError(f"{missing} eligible instrument/date predictions are missing")
    store = StreamingSupervision(pl.scan_parquet(root / "sparse_events.parquet"))
    # Availability, not just the event date, determines baseline fitting.
    train = StreamingSupervision(
        pl.scan_parquet(root / "sparse_events.parquet"), cutoff=datetime.fromisoformat(training_cutoff)
    ).scan
    joined = scores.join(store.scan, on=["symbol", "date"], how="inner", suffix="_target").join(
        taxonomy.lazy(), on="symbol"
    )
    metrics = []
    for asset in sorted(taxonomy["asset_class"].unique()):
        symbols = taxonomy.filter(pl.col("asset_class") == asset)["symbol"].to_list()
        for task in TASKS:
            baseline = (
                train.filter(pl.col("symbol").is_in(symbols))
                .select(pl.col(task).mean())
                .collect(engine="streaming")
                .item()
            )
            observed = joined.filter(
                (pl.col("asset_class") == asset) & pl.col(task + "_target").is_not_null()
            )
            aggregates = [
                pl.len().alias("observations"),
                (pl.col(task) - pl.col(task + "_target")).pow(2).mean().alias("mse"),
                (pl.lit(baseline) - pl.col(task + "_target"))
                .pow(2)
                .mean()
                .alias("train_mean_baseline_mse"),
            ]
            if task in ORACLE_SUPERVISED_TASK_NAMES:
                aggregates += [
                    ((pl.col(task) >= 0.5) == (pl.col(task + "_target") >= 0.5))
                    .mean()
                    .alias("accuracy"),
                    (pl.col(task) >= 0.5)
                    .filter(pl.col(task + "_target") >= 0.5)
                    .mean()
                    .alias("true_positive_rate"),
                    (pl.col(task) < 0.5)
                    .filter(pl.col(task + "_target") < 0.5)
                    .mean()
                    .alias("true_negative_rate"),
                ]
            row = observed.select(*aggregates).collect(engine="streaming").row(0, named=True)
            metrics.append(dict(asset_class=asset, task=task, **row))
    return dict(
        start=start,
        end=end,
        training_cutoff=training_cutoff,
        scored_rows=scores.select(pl.len()).collect(engine="streaming").item(),
        missing_rows=missing,
        event_metrics=metrics,
    )
