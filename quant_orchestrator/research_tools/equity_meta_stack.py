from __future__ import annotations

import json
import pickle
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from quant_orchestrator.platforms.ml_frameworks.rapids import RapidsRandomForestClassifier
from quant_orchestrator.research_tools.family_score_pipeline import SCORE_COLUMNS, normalize_family_scores


class MetaClassifier(Protocol):
    encoder: Any

    def predict_proba_frame(self, frame: pd.DataFrame, features: list[str]) -> pd.DataFrame: ...


MetaClassifierFactory = Callable[[pd.DataFrame, list[str], "EquityMetaStackConfig"], MetaClassifier]


@dataclass(frozen=True)
class EquityMetaStackConfig:
    output_dir: Path
    target_col: str = "collapsed_label"
    long_label: str = "oracle_long"
    short_label: str = "oracle_short"
    model_id: str = "equity_meta_stack"
    model_version: str = "1"
    min_train_rows: int = 250
    min_family_models: int = 2
    random_seed: int = 20260712
    model_params: Mapping[str, Any] = field(
        default_factory=lambda: {
            "n_estimators": 300,
            "max_depth": 12,
            "max_features": "sqrt",
            "n_bins": 128,
            "n_streams": 8,
        }
    )


@dataclass
class EquityMetaStackModel:
    classifier: MetaClassifier
    feature_columns: tuple[str, ...]
    medians: dict[str, float]
    config: EquityMetaStackConfig

    def score(self, frame: pd.DataFrame) -> pd.DataFrame:
        features = list(self.feature_columns)
        work = frame.copy()
        for feature in features:
            if feature not in work.columns:
                work[feature] = pd.NA
        work[features] = (
            work[features]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(pd.Series(self.medians))
            .astype("float32")
        )
        return self.classifier.predict_proba_frame(work, features)


@dataclass(frozen=True)
class EquityMetaStackResult:
    model_path: Path
    summary_path: Path
    scores_path: Path
    scores: pd.DataFrame
    training_rows: int
    family_models: int
    features: tuple[str, ...]


def train_equity_meta_stack(
    family_scores: pd.DataFrame,
    label_rows: pd.DataFrame,
    config: EquityMetaStackConfig,
    *,
    classifier_factory: MetaClassifierFactory | None = None,
) -> EquityMetaStackResult:
    """Train a stacker on in-sample family predictions for the same oracle rows."""

    wide, model_ids, lineage_fingerprint = build_equity_meta_feature_frame(family_scores)
    if len(model_ids) < int(config.min_family_models):
        raise ValueError(
            f"Equity meta-stack requires at least {config.min_family_models} family models; "
            f"received {len(model_ids)}"
        )
    labels = _normalize_meta_labels(label_rows, config.target_col)
    training = wide.merge(labels, on=["symbol", "date"], how="inner", validate="one_to_one")
    if len(training) < int(config.min_train_rows):
        raise ValueError(
            f"Equity meta-stack requires at least {config.min_train_rows} oracle rows; "
            f"received {len(training)}"
        )
    if training[config.target_col].nunique() < 2:
        raise ValueError("Equity meta-stack requires at least two target classes")

    features = tuple(
        column
        for column in wide.columns
        if column not in {"symbol", "date", "run_id", "target_id", "training_end"}
    )
    medians = (
        training[list(features)]
        .apply(pd.to_numeric, errors="coerce")
        .median(axis=0)
        .fillna(0.0)
        .astype(float)
    )
    training[list(features)] = (
        training[list(features)].apply(pd.to_numeric, errors="coerce").fillna(medians).astype("float32")
    )
    factory = classifier_factory or _fit_meta_classifier
    classifier = factory(training, list(features), config)
    artifact = EquityMetaStackModel(
        classifier=classifier,
        feature_columns=features,
        medians=medians.to_dict(),
        config=config,
    )
    probability = artifact.score(wide)
    scores = _meta_probability_scores(
        wide,
        probability,
        config=config,
        lineage_fingerprint=lineage_fingerprint,
    )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "equity_meta_stack.pkl"
    scores_path = output_dir / "equity_meta_stack_scores.parquet"
    summary_path = output_dir / "equity_meta_stack_summary.json"
    with model_path.open("wb") as handle:
        pickle.dump(artifact, handle, protocol=pickle.HIGHEST_PROTOCOL)
    scores.to_parquet(scores_path, index=False)
    summary = {
        "schema_version": 1,
        "training_prediction_scope": "in_sample_same_oracle_rows",
        "meta_feature_contract": "family_long_probability_only",
        "training_rows": int(len(training)),
        "family_models": len(model_ids),
        "model_ids": model_ids,
        "features": list(features),
        "lineage_fingerprint": lineage_fingerprint,
        "config": {**asdict(config), "output_dir": str(output_dir)},
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return EquityMetaStackResult(
        model_path=model_path,
        summary_path=summary_path,
        scores_path=scores_path,
        scores=scores,
        training_rows=len(training),
        family_models=len(model_ids),
        features=features,
    )


def build_equity_meta_feature_frame(
    family_scores: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], str]:
    scores = normalize_family_scores(family_scores)
    scores = scores.loc[~scores["model_id"].astype(str).str.startswith("ensemble")].copy()
    if scores.empty:
        raise ValueError("Family scores are empty")
    lineage = sorted(scores["lineage_fingerprint"].dropna().astype(str).unique())
    if len(lineage) != 1:
        raise ValueError(f"Equity meta-stack requires one score lineage; received {lineage}")
    model_ids = sorted(scores["model_id"].dropna().astype(str).unique())
    keys = ["symbol", "date"]
    wide = scores[keys].drop_duplicates().sort_values(keys).reset_index(drop=True)
    pivot = scores.pivot_table(
        index=keys,
        columns="model_id",
        values="long_score",
        aggfunc="first",
    )
    pivot.columns = [f"long_score__{column}" for column in pivot.columns]
    wide = wide.merge(pivot.reset_index(), on=keys, how="left", validate="one_to_one")
    return wide, model_ids, lineage[0]


def _normalize_meta_labels(label_rows: pd.DataFrame, target_col: str) -> pd.DataFrame:
    required = {"symbol", "date", target_col}
    missing = required.difference(label_rows.columns)
    if missing:
        raise KeyError(f"Meta-stack labels missing columns: {sorted(missing)}")
    labels = label_rows[["symbol", "date", target_col]].copy()
    labels["symbol"] = labels["symbol"].astype(str).str.strip().str.upper()
    labels["date"] = pd.to_datetime(labels["date"], errors="coerce").dt.normalize()
    labels = labels.dropna().drop_duplicates()
    conflicts = labels.groupby(["symbol", "date"])[target_col].nunique()
    if conflicts.gt(1).any():
        raise ValueError("Meta-stack labels contain conflicting classes for the same symbol/date")
    return labels.drop_duplicates(["symbol", "date"])


def _fit_meta_classifier(
    frame: pd.DataFrame,
    features: list[str],
    config: EquityMetaStackConfig,
) -> MetaClassifier:
    return RapidsRandomForestClassifier.fit(
        frame,
        features=features,
        target_col=config.target_col,
        random_state=config.random_seed,
        params=dict(config.model_params),
    )


def _meta_probability_scores(
    wide: pd.DataFrame,
    probability: pd.DataFrame,
    *,
    config: EquityMetaStackConfig,
    lineage_fingerprint: str,
) -> pd.DataFrame:
    long_col = f"prob__{config.long_label}"
    short_col = f"prob__{config.short_label}"
    out = wide[["symbol", "date"]].copy()
    long_values = probability[long_col] if long_col in probability else pd.Series(0.0, index=wide.index)
    short_values = probability[short_col] if short_col in probability else pd.Series(0.0, index=wide.index)
    out["long_score"] = pd.to_numeric(long_values, errors="coerce").fillna(0.0).to_numpy()
    out["short_score"] = pd.to_numeric(short_values, errors="coerce").fillna(0.0).to_numpy()
    out["net_score"] = out["long_score"] - out["short_score"]
    out["score_rank"] = out.groupby("date")["long_score"].rank(method="average", pct=True)
    out["run_id"] = config.model_id
    out["model_id"] = config.model_id
    out["model_version"] = config.model_version
    out["target_id"] = "oracle_side_ye_k1_3"
    out["feature_family"] = "meta_stack"
    out["strategy_source"] = config.model_id
    out["source"] = "meta"
    out["family"] = "stack"
    out["training_end"] = pd.to_datetime(out["date"], errors="coerce").max()
    out["is_out_of_sample"] = False
    out["lineage_fingerprint"] = lineage_fingerprint
    return normalize_family_scores(out[list(SCORE_COLUMNS)])
