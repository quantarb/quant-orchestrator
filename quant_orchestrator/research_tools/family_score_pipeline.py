from __future__ import annotations

import gc
import json
import pickle
import re
import shutil
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

import pandas as pd

from quant_orchestrator.platforms.ml_frameworks.rapids import RapidsRandomForestClassifier
from quant_orchestrator.research_tools.ml_trading import (
    build_strategy_score_frame,
    classification_probability_diagnostics,
)


SCORE_COLUMNS = (
    "run_id",
    "model_id",
    "model_version",
    "target_id",
    "feature_family",
    "strategy_source",
    "source",
    "family",
    "symbol",
    "date",
    "long_score",
    "short_score",
    "net_score",
    "score_rank",
    "training_end",
    "is_out_of_sample",
)


class FamilyClassifier(Protocol):
    encoder: Any

    def predict_proba_frame(self, frame: pd.DataFrame, features: Iterable[str]) -> pd.DataFrame: ...


ClassifierFactory = Callable[[pd.DataFrame, list[str], "FamilyClassifierConfig"], FamilyClassifier]
CleanupCallback = Callable[[], None]
ProgressLogger = Callable[[str], None]


@dataclass(frozen=True)
class FamilyClassifierConfig:
    """Settings required to train one raw feature-family classifier."""

    train_end: str
    score_start: str
    min_feature_coverage: float = 0.50
    min_train_rows: int = 250
    min_classes: int = 2
    random_seed: int = 20260702
    target_col: str = "collapsed_label"
    model_backend: str = "rapids_random_forest"
    model_params: Mapping[str, Any] = field(
        default_factory=lambda: {
            "n_estimators": 300,
            "max_depth": 16,
            "max_features": "sqrt",
            "n_bins": 128,
            "n_streams": 8,
        }
    )
    run_diagnostics: bool = True


@dataclass(frozen=True)
class ScoreMaterializationConfig:
    """Controls bounded scoring and reusable artifact persistence."""

    output_dir: Path
    run_id: str
    target_id: str
    rows_per_chunk: int = 250_000
    model_version: str = "1"
    persist_models: bool = True
    overwrite: bool = False


@dataclass(frozen=True)
class FeatureFamilyBatch:
    """One independently trainable and scoreable feature family."""

    source: str
    family: str
    frame: pd.DataFrame
    feature_columns: tuple[str, ...]

    @property
    def model_id(self) -> str:
        return f"{self.source}.{self.family}"


@dataclass(frozen=True)
class FamilyScoreRunResult:
    run_id: str
    output_dir: Path
    model_results: pd.DataFrame
    manifest_path: Path
    score_rows: int


class FamilyScoreStore:
    """Partitioned Parquet store for model scores reused by strategies."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def write_chunk(self, frame: pd.DataFrame, *, model_id: str, chunk_number: int) -> list[Path]:
        if frame.empty:
            return []
        written: list[Path] = []
        scores = normalize_family_scores(frame)
        years = pd.to_datetime(scores["date"], errors="coerce").dt.year.astype("Int64")
        for year, part in scores.groupby(years, dropna=False):
            year_text = "unknown" if pd.isna(year) else str(int(year))
            directory = self.root / f"model={_safe_path_token(model_id)}" / f"year={year_text}"
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"part-{int(chunk_number):06d}.parquet"
            part.to_parquet(path, index=False)
            written.append(path)
        return written

    def read_scores(
        self,
        *,
        model_ids: Iterable[str] | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        wanted = {str(value) for value in model_ids or ()}
        paths = sorted(self.root.glob("model=*/year=*/part-*.parquet"))
        frames = []
        for path in paths:
            frame = pd.read_parquet(path)
            if wanted and ("model_id" not in frame.columns or not frame["model_id"].astype(str).isin(wanted).any()):
                continue
            frames.append(frame)
        if not frames:
            return pd.DataFrame(columns=list(SCORE_COLUMNS))
        out = normalize_family_scores(pd.concat(frames, ignore_index=True, sort=False))
        if wanted:
            out = out.loc[out["model_id"].astype(str).isin(wanted)].copy()
        dates = pd.to_datetime(out["date"], errors="coerce")
        if start is not None:
            out = out.loc[dates.ge(pd.Timestamp(start))].copy()
            dates = pd.to_datetime(out["date"], errors="coerce")
        if end is not None:
            out = out.loc[dates.le(pd.Timestamp(end))].copy()
        return out.reset_index(drop=True)


def iter_feature_family_batches(
    feature_panel: pd.DataFrame,
    feature_metadata: pd.DataFrame,
) -> Iterator[FeatureFamilyBatch]:
    """Yield lightweight family views from an existing combined panel."""

    required = {"source", "family", "feature"}
    missing = required.difference(feature_metadata.columns)
    if missing:
        raise KeyError(f"feature metadata missing columns: {sorted(missing)}")
    key_columns = [col for col in ("symbol", "date") if col in feature_panel.columns]
    if len(key_columns) != 2:
        raise KeyError("feature panel requires symbol and date columns")
    families = feature_metadata[["source", "family"]].drop_duplicates().sort_values(["source", "family"])
    for source, family in families.itertuples(index=False, name=None):
        selected = feature_metadata.loc[
            feature_metadata["source"].eq(source) & feature_metadata["family"].eq(family), "feature"
        ]
        features = tuple(dict.fromkeys(str(col) for col in selected if str(col) in feature_panel.columns))
        if not features:
            continue
        yield FeatureFamilyBatch(
            source=str(source),
            family=str(family),
            frame=feature_panel[[*key_columns, *features]],
            feature_columns=features,
        )


def train_and_materialize_family_scores(
    batches: Iterable[FeatureFamilyBatch],
    label_rows: pd.DataFrame,
    *,
    classifier: FamilyClassifierConfig,
    materialization: ScoreMaterializationConfig,
    classifier_factory: ClassifierFactory | None = None,
    cleanup_callback: CleanupCallback | None = None,
    progress_logger: ProgressLogger | None = None,
) -> FamilyScoreRunResult:
    """Train, score, persist, and release one family at a time."""

    output_dir = Path(materialization.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not materialization.overwrite:
        raise FileExistsError(f"score output directory is not empty: {output_dir}")
    if output_dir.exists() and materialization.overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    score_store = FamilyScoreStore(output_dir / "scores")
    model_dir = output_dir / "models"
    if materialization.persist_models:
        model_dir.mkdir(parents=True, exist_ok=True)

    labels = _normalize_labels(label_rows, target_col=classifier.target_col)
    factory = classifier_factory or _fit_rapids_classifier
    cleanup = cleanup_callback or release_training_memory
    model_rows: list[dict[str, Any]] = []
    score_paths: list[str] = []
    total_score_rows = 0
    started = perf_counter()

    for batch in batches:
        family_started = perf_counter()
        if callable(progress_logger):
            progress_logger(f"[family-scores] training {batch.model_id}")
        model: FamilyClassifier | None = None
        training_frame = pd.DataFrame()
        try:
            score_frame, usable_features = _prepare_score_frame(batch, classifier.min_feature_coverage)
            training_frame = _event_training_frame(score_frame, labels, target_col=classifier.target_col)
            train_end = pd.Timestamp(classifier.train_end).normalize()
            training_frame = training_frame.loc[training_frame["date"].le(train_end)].copy()
            if len(training_frame) < classifier.min_train_rows or training_frame[classifier.target_col].nunique() < classifier.min_classes:
                model_rows.append(
                    _model_result_row(
                        batch,
                        status="skipped_sparse_train",
                        features=len(usable_features),
                        train_rows=len(training_frame),
                        score_rows=0,
                        seconds=perf_counter() - family_started,
                    )
                )
                continue

            medians = (
                training_frame[usable_features]
                .median(axis=0)
                .replace([float("inf"), float("-inf")], float("nan"))
                .fillna(0.0)
            )
            training_frame[usable_features] = training_frame[usable_features].fillna(medians).astype("float32")
            score_frame[usable_features] = score_frame[usable_features].fillna(medians).astype("float32")

            model = factory(training_frame, usable_features, classifier)
            diagnostics = _classifier_diagnostics(model, training_frame, usable_features, classifier)
            model_path = None
            if materialization.persist_models:
                model_path = model_dir / f"{_safe_path_token(batch.model_id)}.pkl"
                with model_path.open("wb") as handle:
                    pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

            family_score_rows = 0
            filtered_scores = score_frame.loc[score_frame["date"].ge(pd.Timestamp(classifier.score_start))].copy()
            chunk_size = max(1, int(materialization.rows_per_chunk))
            for chunk_number, start in enumerate(range(0, len(filtered_scores), chunk_size)):
                chunk = filtered_scores.iloc[start : start + chunk_size].copy()
                probability = model.predict_proba_frame(chunk, usable_features)
                scores = build_strategy_score_frame(
                    source=batch.source,
                    family=batch.family,
                    prediction_frame=chunk,
                    probability_frame=probability,
                    ae_familiarity_frame=None,
                    apply_ae_to_exits=False,
                )
                scores = _score_contract_columns(
                    scores,
                    batch=batch,
                    classifier=classifier,
                    materialization=materialization,
                )
                paths = score_store.write_chunk(scores, model_id=batch.model_id, chunk_number=chunk_number)
                score_paths.extend(str(path.relative_to(output_dir)) for path in paths)
                family_score_rows += len(scores)
                del probability, scores, chunk

            total_score_rows += family_score_rows
            model_rows.append(
                {
                    **_model_result_row(
                        batch,
                        status="ok",
                        features=len(usable_features),
                        train_rows=len(training_frame),
                        score_rows=family_score_rows,
                        seconds=perf_counter() - family_started,
                    ),
                    "model_path": None if model_path is None else str(model_path.relative_to(output_dir)),
                    **diagnostics,
                }
            )
            if callable(progress_logger):
                progress_logger(
                    f"[family-scores] completed {batch.model_id} "
                    f"train_rows={len(training_frame)} score_rows={family_score_rows}"
                )
        finally:
            del model, training_frame
            cleanup()

    model_results = pd.DataFrame(model_rows)
    model_results_path = output_dir / "model_results.parquet"
    model_results.to_parquet(model_results_path, index=False)
    manifest = {
        "schema_version": 1,
        "run_id": materialization.run_id,
        "target_id": materialization.target_id,
        "score_columns": list(SCORE_COLUMNS),
        "score_rows": int(total_score_rows),
        "models_requested": int(len(model_rows)),
        "models_trained": int(sum(row.get("status") == "ok" for row in model_rows)),
        "classifier": _jsonable(asdict(classifier)),
        "materialization": _jsonable(asdict(materialization)),
        "score_parts": score_paths,
        "model_results": model_results_path.name,
        "elapsed_seconds": round(perf_counter() - started, 3),
    }
    manifest_path = output_dir / "family_score_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return FamilyScoreRunResult(
        run_id=materialization.run_id,
        output_dir=output_dir,
        model_results=model_results,
        manifest_path=manifest_path,
        score_rows=int(total_score_rows),
    )


def normalize_family_scores(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in SCORE_COLUMNS:
        if column not in out.columns:
            out[column] = pd.NA
    out["symbol"] = out["symbol"].astype(str).str.strip().str.upper()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    for column in ("long_score", "short_score", "net_score", "score_rank"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out.loc[:, list(SCORE_COLUMNS)].sort_values(["model_id", "date", "symbol"], kind="stable").reset_index(drop=True)


def build_score_ensemble(
    scores: pd.DataFrame,
    *,
    ensemble_id: str = "ensemble_mean",
    model_ids: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Build a mean ensemble from any selected reusable score streams."""

    if scores is None or scores.empty:
        return pd.DataFrame(columns=list(SCORE_COLUMNS))
    selected = normalize_family_scores(scores)
    wanted = {str(value) for value in model_ids or ()}
    if wanted:
        selected = selected.loc[selected["model_id"].astype(str).isin(wanted)].copy()
    if selected.empty:
        return pd.DataFrame(columns=list(SCORE_COLUMNS))
    grouped = (
        selected.groupby(["run_id", "target_id", "symbol", "date"], as_index=False)
        .agg(
            long_score=("long_score", "mean"),
            short_score=("short_score", "mean"),
            training_end=("training_end", "max"),
            is_out_of_sample=("is_out_of_sample", "all"),
        )
    )
    grouped["model_id"] = ensemble_id
    grouped["model_version"] = "1"
    grouped["feature_family"] = "ensemble"
    grouped["strategy_source"] = ensemble_id
    grouped["source"] = "ensemble"
    grouped["family"] = "mean"
    grouped["net_score"] = grouped["long_score"] - grouped["short_score"]
    grouped["score_rank"] = grouped.groupby("date")["long_score"].rank(method="average", pct=True)
    return normalize_family_scores(grouped)


def release_training_memory() -> None:
    """Release Python objects and any already-imported CuPy cached allocations."""

    gc.collect()
    try:
        import sys

        cp = sys.modules.get("cupy")
        if cp is not None:
            cp.cuda.runtime.deviceSynchronize()
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
    except (AttributeError, RuntimeError):
        pass


def _prepare_score_frame(batch: FeatureFamilyBatch, min_coverage: float) -> tuple[pd.DataFrame, list[str]]:
    features = [
        col
        for col in batch.feature_columns
        if col in batch.frame.columns and pd.to_numeric(batch.frame[col], errors="coerce").notna().any()
    ]
    if not features:
        return pd.DataFrame(columns=["symbol", "date"]), []
    numeric = batch.frame[features].apply(pd.to_numeric, errors="coerce").replace(
        [float("inf"), float("-inf")], float("nan")
    )
    coverage = numeric.notna().mean(axis=1)
    frame = batch.frame.loc[coverage.ge(float(min_coverage)), ["symbol", "date"]].copy()
    frame[features] = numeric.loc[frame.index]
    frame["symbol"] = frame["symbol"].astype(str).str.strip().str.upper()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    return frame.dropna(subset=["symbol", "date"]).reset_index(drop=True), features


def _event_training_frame(score_frame: pd.DataFrame, labels: pd.DataFrame, *, target_col: str) -> pd.DataFrame:
    if score_frame.empty or labels.empty:
        return pd.DataFrame(columns=[*score_frame.columns, target_col])
    return score_frame.merge(labels[["symbol", "date", target_col]], on=["symbol", "date"], how="inner")


def _normalize_labels(labels: pd.DataFrame, *, target_col: str) -> pd.DataFrame:
    required = {"symbol", "date", target_col}
    missing = required.difference(labels.columns)
    if missing:
        raise KeyError(f"label rows missing columns: {sorted(missing)}")
    out = labels[["symbol", "date", target_col]].copy()
    out["symbol"] = out["symbol"].astype(str).str.strip().str.upper()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    return out.dropna().drop_duplicates(["symbol", "date", target_col]).reset_index(drop=True)


def _fit_rapids_classifier(
    frame: pd.DataFrame,
    features: list[str],
    config: FamilyClassifierConfig,
) -> FamilyClassifier:
    if config.model_backend != "rapids_random_forest":
        raise ValueError(f"unsupported classifier backend: {config.model_backend}")
    return RapidsRandomForestClassifier.fit(
        frame,
        features=features,
        target_col=config.target_col,
        random_state=config.random_seed,
        params=dict(config.model_params),
    )


def _classifier_diagnostics(
    model: FamilyClassifier,
    frame: pd.DataFrame,
    features: list[str],
    config: FamilyClassifierConfig,
) -> dict[str, Any]:
    if not config.run_diagnostics:
        return {}
    probability = model.predict_proba_frame(frame, features)
    metrics = classification_probability_diagnostics(
        frame,
        probability,
        target_col=config.target_col,
        labels=model.encoder.classes_,
    )
    return {f"train_{key}": value for key, value in metrics.items()}


def _score_contract_columns(
    scores: pd.DataFrame,
    *,
    batch: FeatureFamilyBatch,
    classifier: FamilyClassifierConfig,
    materialization: ScoreMaterializationConfig,
) -> pd.DataFrame:
    out = scores.copy()
    out["run_id"] = materialization.run_id
    out["model_id"] = batch.model_id
    out["model_version"] = materialization.model_version
    out["target_id"] = materialization.target_id
    out["feature_family"] = batch.family
    out["training_end"] = pd.Timestamp(classifier.train_end).normalize()
    out["is_out_of_sample"] = pd.to_datetime(out["date"], errors="coerce").gt(pd.Timestamp(classifier.train_end))
    out["score_rank"] = out.groupby("date")["long_score"].rank(method="average", pct=True)
    return normalize_family_scores(out)


def _model_result_row(
    batch: FeatureFamilyBatch,
    *,
    status: str,
    features: int,
    train_rows: int,
    score_rows: int,
    seconds: float,
) -> dict[str, Any]:
    return {
        "model_id": batch.model_id,
        "source": batch.source,
        "family": batch.family,
        "status": status,
        "features": int(features),
        "train_rows": int(train_rows),
        "score_rows": int(score_rows),
        "elapsed_seconds": float(seconds),
    }


def _safe_path_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._") or "model"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value
