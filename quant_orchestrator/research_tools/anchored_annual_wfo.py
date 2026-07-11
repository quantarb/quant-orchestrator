from __future__ import annotations

import json
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
    FeatureFamilyBatch,
    ProgressLogger,
    ScoreMaterializationConfig,
    train_and_materialize_family_scores,
)


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


@dataclass(frozen=True)
class AnchoredAnnualWFOResult:
    output_dir: Path
    folds: pd.DataFrame
    checkpoint_path: Path
    elapsed_seconds: float


BatchFactory = Callable[[], Iterable[FeatureFamilyBatch]]


def run_anchored_annual_wfo(
    batch_factory: BatchFactory,
    label_rows: pd.DataFrame,
    *,
    config: AnchoredAnnualWFOConfig,
    classifier_factory: ClassifierFactory | None = None,
    cleanup_callback: CleanupCallback | None = None,
    progress_logger: ProgressLogger | None = None,
) -> AnchoredAnnualWFOResult:
    """Run independently streamed family classifiers for anchored annual OOS folds.

    ``batch_factory`` is called afresh for every fold so callers can load or stream
    one family at a time instead of retaining all family panels in memory.
    """

    started = perf_counter()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "annual_wfo_checkpoint.json"
    checkpoint = _read_checkpoint(checkpoint_path) if config.restart else {"folds": {}}
    completed = dict(checkpoint.get("folds", {}))
    fold_rows: list[dict[str, Any]] = []

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
        row = _fold_row(test_year, train_end, result, perf_counter() - fold_started)
        completed[fold_id] = row
        fold_rows.append({**row, "resumed": False})
        _write_checkpoint(checkpoint_path, config, completed)

    folds = pd.DataFrame(fold_rows).sort_values("test_year").reset_index(drop=True) if fold_rows else pd.DataFrame()
    folds.to_parquet(output_dir / "annual_wfo_folds.parquet", index=False)
    return AnchoredAnnualWFOResult(
        output_dir=output_dir,
        folds=folds,
        checkpoint_path=checkpoint_path,
        elapsed_seconds=perf_counter() - started,
    )


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
) -> dict[str, Any]:
    return {
        "test_year": int(test_year),
        "train_end": str(train_end.date()),
        "status": "complete",
        "models_requested": int(len(result.model_results)),
        "models_trained": int(result.model_results.get("status", pd.Series(dtype=str)).eq("ok").sum()),
        "score_rows": int(result.score_rows),
        "elapsed_seconds": float(seconds),
        "manifest_path": str(result.manifest_path),
    }


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
