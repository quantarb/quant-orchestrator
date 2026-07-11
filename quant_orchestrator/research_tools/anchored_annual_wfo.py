from __future__ import annotations

import json
import resource
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd

from quant_orchestrator.research_tools.family_score_pipeline import (
    ClassifierFactory,
    CleanupCallback,
    FamilyClassifierConfig,
    FamilyScoreRunResult,
    FamilyScoreStore,
    FeatureFamilyBatch,
    ProgressLogger,
    ScoreMaterializationConfig,
    train_and_materialize_family_scores,
)
from quant_orchestrator.research_tools.ml_trading import classification_probability_diagnostics


@dataclass(frozen=True)
class AnchoredAnnualWFOConfig:
    """Calendar-year folds with an expanding, strictly prior-year training window."""

    output_dir: Path
    run_id: str
    target_id: str
    input_lineage_paths: tuple[Path, ...]
    test_years: tuple[int, ...]
    min_feature_coverage: float = 0.50
    min_train_rows: int = 250
    min_classes: int = 2
    random_seed: int = 20260702
    target_col: str = "collapsed_label"
    model_params: dict[str, Any] = field(default_factory=dict)
    rows_per_chunk: int = 250_000
    persist_models: bool = True
    run_diagnostics: bool = True
    restart: bool = True
    top_k: int = 20


@dataclass(frozen=True)
class AnchoredAnnualWFOResult:
    output_dir: Path
    folds: pd.DataFrame
    annual_metrics: pd.DataFrame
    checkpoint_path: Path
    elapsed_seconds: float
    peak_unified_memory_rss_mb: float


BatchFactory = Callable[[], Iterable[FeatureFamilyBatch]]
MemoryProbe = Callable[[], float]


def run_anchored_annual_wfo(
    batch_factory: BatchFactory,
    label_rows: pd.DataFrame,
    *,
    config: AnchoredAnnualWFOConfig,
    classifier_factory: ClassifierFactory | None = None,
    cleanup_callback: CleanupCallback | None = None,
    progress_logger: ProgressLogger | None = None,
    memory_probe: MemoryProbe | None = None,
) -> AnchoredAnnualWFOResult:
    """Run independently streamed family classifiers for anchored annual OOS folds.

    ``batch_factory`` is called afresh for every fold so callers can load or stream
    one family at a time instead of retaining all family panels in memory.
    """

    started = perf_counter()
    probe = memory_probe or process_peak_rss_mb
    peak_unified_memory_rss_mb = float(probe())
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "annual_wfo_checkpoint.json"
    checkpoint = _read_checkpoint(checkpoint_path) if config.restart else {"folds": {}}
    completed = dict(checkpoint.get("folds", {}))
    fold_rows: list[dict[str, Any]] = []
    metric_frames: list[pd.DataFrame] = []

    for test_year in _validated_years(config.test_years):
        fold_id = str(test_year)
        fold_dir = output_dir / f"test_year={test_year}"
        manifest_path = fold_dir / "family_score_manifest.json"
        if config.restart and completed.get(fold_id, {}).get("status") == "complete" and manifest_path.is_file():
            row = dict(completed[fold_id])
            row["resumed"] = True
            fold_rows.append(row)
            if callable(progress_logger):
                progress_logger(f"[annual-wfo] resumed completed test_year={test_year}")
            metric_frames.append(
                evaluate_annual_oos_scores(
                    FamilyScoreStore(fold_dir / "scores").read_scores(),
                    label_rows,
                    test_year=test_year,
                    target_col=config.target_col,
                    top_k=config.top_k,
                )
            )
            continue

        train_end = pd.Timestamp(year=test_year - 1, month=12, day=31)
        score_start = pd.Timestamp(year=test_year, month=1, day=1)
        score_end = pd.Timestamp(year=test_year, month=12, day=31)
        if callable(progress_logger):
            progress_logger(f"[annual-wfo] starting test_year={test_year} train_end={train_end.date()}")
        fold_started = perf_counter()
        result = train_and_materialize_family_scores(
            _bounded_batches(batch_factory(), end=score_end),
            label_rows,
            classifier=FamilyClassifierConfig(
                train_end=str(train_end.date()),
                score_start=str(score_start.date()),
                min_feature_coverage=config.min_feature_coverage,
                min_train_rows=config.min_train_rows,
                min_classes=config.min_classes,
                random_seed=config.random_seed,
                target_col=config.target_col,
                model_params=config.model_params or FamilyClassifierConfig(
                    train_end=str(train_end.date()), score_start=str(score_start.date())
                ).model_params,
                run_diagnostics=config.run_diagnostics,
            ),
            materialization=ScoreMaterializationConfig(
                output_dir=fold_dir,
                run_id=f"{config.run_id}.test_year_{test_year}",
                target_id=config.target_id,
                input_lineage_paths=config.input_lineage_paths,
                rows_per_chunk=config.rows_per_chunk,
                persist_models=config.persist_models,
                overwrite=fold_dir.exists(),
            ),
            classifier_factory=classifier_factory,
            cleanup_callback=cleanup_callback,
            progress_logger=progress_logger,
        )
        peak_unified_memory_rss_mb = max(peak_unified_memory_rss_mb, float(probe()))
        row = _fold_row(
            test_year,
            train_end,
            result,
            perf_counter() - fold_started,
            peak_unified_memory_rss_mb,
        )
        completed[fold_id] = row
        fold_rows.append({**row, "resumed": False})
        metric_frames.append(
            evaluate_annual_oos_scores(
                FamilyScoreStore(fold_dir / "scores").read_scores(),
                label_rows,
                test_year=test_year,
                target_col=config.target_col,
                top_k=config.top_k,
            )
        )
        _write_checkpoint(checkpoint_path, config, completed)

    folds = pd.DataFrame(fold_rows).sort_values("test_year").reset_index(drop=True) if fold_rows else pd.DataFrame()
    annual_metrics = (
        pd.concat(metric_frames, ignore_index=True).sort_values(["test_year", "model_id"]).reset_index(drop=True)
        if metric_frames and any(not frame.empty for frame in metric_frames)
        else pd.DataFrame()
    )
    folds.to_parquet(output_dir / "annual_wfo_folds.parquet", index=False)
    annual_metrics.to_parquet(output_dir / "annual_wfo_metrics.parquet", index=False)
    return AnchoredAnnualWFOResult(
        output_dir=output_dir,
        folds=folds,
        annual_metrics=annual_metrics,
        checkpoint_path=checkpoint_path,
        elapsed_seconds=perf_counter() - started,
        peak_unified_memory_rss_mb=peak_unified_memory_rss_mb,
    )


def evaluate_annual_oos_scores(
    scores: pd.DataFrame,
    label_rows: pd.DataFrame,
    *,
    test_year: int,
    target_col: str = "collapsed_label",
    top_k: int = 20,
) -> pd.DataFrame:
    """Evaluate annual OOS side classification and daily top-k ranking by model."""

    if int(top_k) < 1:
        raise ValueError("top_k must be >= 1")
    required_scores = {"model_id", "symbol", "date", "long_score", "short_score"}
    missing_scores = required_scores.difference(scores.columns)
    if missing_scores:
        raise KeyError(f"scores missing columns: {sorted(missing_scores)}")
    required_labels = {"symbol", "date", target_col}
    missing_labels = required_labels.difference(label_rows.columns)
    if missing_labels:
        raise KeyError(f"label rows missing columns: {sorted(missing_labels)}")

    score_frame = scores.copy()
    score_frame["date"] = pd.to_datetime(score_frame["date"], errors="coerce").dt.normalize()
    score_frame = score_frame.loc[score_frame["date"].dt.year.eq(int(test_year))].copy()
    labels = label_rows[["symbol", "date", target_col]].copy()
    labels["symbol"] = labels["symbol"].astype(str).str.strip().str.upper()
    labels["date"] = pd.to_datetime(labels["date"], errors="coerce").dt.normalize()
    labels = labels.loc[labels[target_col].astype(str).isin(("oracle_long", "oracle_short"))]
    merged = score_frame.merge(labels, on=["symbol", "date"], how="inner")
    rows: list[dict[str, Any]] = []
    for model_id, model_frame in merged.groupby("model_id", sort=True):
        probability = pd.DataFrame(
            {
                "prob__oracle_long": pd.to_numeric(model_frame["long_score"], errors="coerce"),
                "prob__oracle_short": pd.to_numeric(model_frame["short_score"], errors="coerce"),
            },
            index=model_frame.index,
        )
        diagnostics = classification_probability_diagnostics(
            model_frame,
            probability,
            target_col=target_col,
            labels=("oracle_long", "oracle_short"),
        )
        long_selected = _daily_top_k(model_frame, "long_score", top_k)
        short_selected = _daily_top_k(model_frame, "short_score", top_k)
        long_correct = long_selected[target_col].astype(str).eq("oracle_long")
        short_correct = short_selected[target_col].astype(str).eq("oracle_short")
        rows.append(
            {
                "test_year": int(test_year),
                "model_id": str(model_id),
                "top_k": int(top_k),
                **{f"classification_{key}": value for key, value in diagnostics.items()},
                "ranking_days": int(model_frame["date"].nunique()),
                "top_k_long_rows": int(len(long_selected)),
                "top_k_short_rows": int(len(short_selected)),
                "top_k_long_precision": float(long_correct.mean()) if len(long_correct) else float("nan"),
                "top_k_short_precision": float(short_correct.mean()) if len(short_correct) else float("nan"),
                "top_k_balanced_precision": float((long_correct.mean() + short_correct.mean()) / 2.0)
                if len(long_correct) and len(short_correct)
                else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _daily_top_k(frame: pd.DataFrame, score_col: str, top_k: int) -> pd.DataFrame:
    ranked = frame.dropna(subset=[score_col]).sort_values(
        ["date", score_col, "symbol"], ascending=[True, False, True], kind="stable"
    )
    return ranked.groupby("date", sort=False).head(int(top_k))


def _bounded_batches(batches: Iterable[FeatureFamilyBatch], *, end: pd.Timestamp) -> Iterable[FeatureFamilyBatch]:
    for batch in batches:
        dates = pd.to_datetime(batch.frame["date"], errors="coerce")
        yield replace(batch, frame=batch.frame.loc[dates.le(end)].copy())


def _validated_years(years: tuple[int, ...]) -> tuple[int, ...]:
    normalized = tuple(sorted(set(int(year) for year in years)))
    if not normalized:
        raise ValueError("test_years must not be empty")
    if any(year < 1901 for year in normalized):
        raise ValueError("test_years must be >= 1901")
    return normalized


def _fold_row(
    test_year: int,
    train_end: pd.Timestamp,
    result: FamilyScoreRunResult,
    seconds: float,
    peak_unified_memory_rss_mb: float,
) -> dict[str, Any]:
    return {
        "test_year": int(test_year),
        "train_end": str(train_end.date()),
        "status": "complete",
        "models_requested": int(len(result.model_results)),
        "models_trained": int(result.model_results.get("status", pd.Series(dtype=str)).eq("ok").sum()),
        "score_rows": int(result.score_rows),
        "elapsed_seconds": float(seconds),
        "peak_unified_memory_rss_mb": float(peak_unified_memory_rss_mb),
        "manifest_path": str(result.manifest_path),
    }


def process_peak_rss_mb() -> float:
    """Return this process's peak resident memory in MiB.

    Linux ``VmHWM`` includes host-resident CUDA unified-memory pages, making it
    the useful reproducible process-level bound for the CUDA-first WFO smoke run.
    ``ru_maxrss`` is retained as a portable fallback (bytes on macOS, KiB on
    Linux and other supported Unix platforms).
    """

    status_path = Path("/proc/self/status")
    if status_path.is_file():
        for line in status_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmHWM:"):
                return float(line.split()[1]) / 1024.0
    peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if __import__("sys").platform == "darwin":
        peak /= 1024.0
    return peak / 1024.0


def _read_checkpoint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"folds": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_checkpoint(path: Path, config: AnchoredAnnualWFOConfig, folds: dict[str, Any]) -> None:
    payload = {"schema_version": 1, "config": _jsonable(asdict(config)), "folds": folds}
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value
