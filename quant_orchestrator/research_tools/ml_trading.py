from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from quant_orchestrator.artifacts import ArtifactStore, RunRecord
from quant_orchestrator.platforms.ml_frameworks.rapids.random_forest import ensure_probability_columns


@dataclass(frozen=True)
class ExperimentArtifacts:
    run: RunRecord
    artifact_uris: dict[str, str]


def prepare_family_dataset(
    feature_panel: pd.DataFrame,
    feature_metadata: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    source: str,
    family: str,
    min_feature_coverage: float,
) -> tuple[pd.DataFrame, list[str]]:
    family_meta = feature_metadata.loc[
        feature_metadata["source"].astype(str).eq(source)
        & feature_metadata["family"].astype(str).eq(family)
    ]
    features = [f for f in family_meta["feature"].drop_duplicates().tolist() if f in feature_panel.columns]
    numeric_features = [f for f in features if pd.to_numeric(feature_panel[f], errors="coerce").notna().any()]
    if not numeric_features:
        return pd.DataFrame(), []
    merged = labels.merge(feature_panel[["symbol", "date", *numeric_features]], on=["symbol", "date"], how="inner")
    if merged.empty:
        return merged, numeric_features
    numeric = merged[numeric_features].apply(pd.to_numeric, errors="coerce")
    coverage = numeric.notna().mean(axis=1)
    merged = merged.loc[coverage.ge(min_feature_coverage)].copy()
    if merged.empty:
        return merged, numeric_features
    numeric = numeric.loc[merged.index]
    medians = numeric.median(axis=0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    merged[numeric_features] = numeric.replace([np.inf, -np.inf], np.nan).fillna(medians).astype("float32")
    return merged.reset_index(drop=True), numeric_features


def build_family_prediction_frame(
    feature_panel: pd.DataFrame,
    features: Iterable[str],
    *,
    min_feature_coverage: float,
) -> pd.DataFrame:
    feature_list = list(features)
    base_cols = ["symbol", "date"]
    numeric = feature_panel[feature_list].apply(pd.to_numeric, errors="coerce")
    coverage = numeric.notna().mean(axis=1)
    out = feature_panel.loc[coverage.ge(min_feature_coverage), base_cols].copy()
    if out.empty:
        return out
    numeric = numeric.loc[out.index]
    medians = numeric.median(axis=0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out[feature_list] = numeric.replace([np.inf, -np.inf], np.nan).fillna(medians).astype("float32")
    return out.reset_index(drop=True)


def build_strategy_score_frame(
    *,
    source: str,
    family: str,
    prediction_frame: pd.DataFrame,
    probability_frame: pd.DataFrame,
    ae_familiarity_frame: pd.DataFrame | None = None,
    apply_ae_to_exits: bool = True,
) -> pd.DataFrame:
    proba = ensure_probability_columns(probability_frame, ["event_bullish", "oracle_long", "oracle_short"])
    classifier_long = (proba["prob__event_bullish"] + proba["prob__oracle_long"]).clip(0, 1).to_numpy(dtype="float64")
    classifier_short = proba["prob__oracle_short"].clip(0, 1).to_numpy(dtype="float64")
    if ae_familiarity_frame is None:
        ae = pd.DataFrame(
            {
                "ae_familiarity": np.ones(len(prediction_frame), dtype="float64"),
                "ae_recon_error": np.zeros(len(prediction_frame), dtype="float64"),
                "ae_latent_distance": np.zeros(len(prediction_frame), dtype="float64"),
            },
            index=prediction_frame.index,
        )
    else:
        ae = ae_familiarity_frame.reindex(prediction_frame.index).copy()
        for col, default in {
            "ae_familiarity": 1.0,
            "ae_recon_error": 0.0,
            "ae_latent_distance": 0.0,
        }.items():
            if col not in ae.columns:
                ae[col] = default
        ae = ae[["ae_familiarity", "ae_recon_error", "ae_latent_distance"]].astype("float64")
    familiarity = ae["ae_familiarity"].clip(0, 1).to_numpy(dtype="float64")
    out = prediction_frame[["symbol", "date"]].copy()
    out["source"] = str(source)
    out["family"] = str(family)
    out["strategy_source"] = strategy_source_name(source, family)
    out["classifier_long_score"] = classifier_long
    out["classifier_short_score"] = classifier_short
    out["ae_familiarity"] = familiarity
    out["ae_recon_error"] = ae["ae_recon_error"].to_numpy(dtype="float64")
    out["ae_latent_distance"] = ae["ae_latent_distance"].to_numpy(dtype="float64")
    out["long_score"] = np.clip(classifier_long * familiarity, 0.0, 1.0)
    out["short_score"] = np.clip(classifier_short * familiarity, 0.0, 1.0)
    if apply_ae_to_exits:
        out["long_exit_score"] = np.clip(classifier_long * familiarity, 0.0, 1.0)
        out["short_exit_score"] = np.clip(classifier_short * familiarity, 0.0, 1.0)
    else:
        out["long_exit_score"] = classifier_long
        out["short_exit_score"] = classifier_short
    out["model_count"] = 1
    out["net_score"] = out["long_score"] - out["short_score"]
    return out


def strategy_source_name(source: str, family: str) -> str:
    return f"{source}.{family}"


def save_experiment_artifacts(
    *,
    experiment_name: str,
    params: dict,
    metrics: dict,
    model_results: pd.DataFrame,
    strategy_scores: pd.DataFrame,
    backtest_summary: pd.DataFrame,
    trade_log: pd.DataFrame,
    analysis_markdown: str,
    store: ArtifactStore | None = None,
) -> ExperimentArtifacts:
    artifact_store = store or ArtifactStore()
    run = artifact_store.create_run(
        run_type="ml_trading",
        name=experiment_name,
        params=params,
        tags={"experiment_name": experiment_name},
    )
    artifact_uris = {
        "model_results": artifact_store.save_dataframe(
            run_id=run.id,
            kind=f"ml_trading_{experiment_name}",
            name="model_results",
            frame=model_results,
        ).uri,
        "strategy_scores": artifact_store.save_dataframe(
            run_id=run.id,
            kind=f"ml_trading_{experiment_name}",
            name="strategy_scores",
            frame=strategy_scores,
        ).uri,
        "backtest_summary": artifact_store.save_dataframe(
            run_id=run.id,
            kind=f"ml_trading_{experiment_name}",
            name="backtest_summary",
            frame=backtest_summary,
        ).uri,
        "trade_log": artifact_store.save_dataframe(
            run_id=run.id,
            kind=f"ml_trading_{experiment_name}",
            name="trade_log",
            frame=trade_log,
        ).uri,
        "analysis": artifact_store.save_text(
            run_id=run.id,
            kind=f"ml_trading_{experiment_name}",
            name="analysis",
            text=analysis_markdown,
            extension="md",
        ).uri,
    }
    run = artifact_store.complete_run(run.id, metrics={**metrics, "artifact_uris": artifact_uris})
    return ExperimentArtifacts(run=run, artifact_uris=artifact_uris)


def load_latest_experiment_artifacts(
    experiment_name: str,
    *,
    store: ArtifactStore | None = None,
) -> dict[str, pd.DataFrame | str]:
    artifact_store = store or ArtifactStore()
    kind = f"ml_trading_{experiment_name}"
    artifacts = {
        "model_results": artifact_store.latest_artifact(kind=kind, name="model_results").uri,
        "strategy_scores": artifact_store.latest_artifact(kind=kind, name="strategy_scores").uri,
        "backtest_summary": artifact_store.latest_artifact(kind=kind, name="backtest_summary").uri,
        "trade_log": artifact_store.latest_artifact(kind=kind, name="trade_log").uri,
        "analysis": artifact_store.latest_artifact(kind=kind, name="analysis").uri,
    }
    return {
        "model_results": artifact_store.load_dataframe(artifacts["model_results"]),
        "strategy_scores": artifact_store.load_dataframe(artifacts["strategy_scores"]),
        "backtest_summary": artifact_store.load_dataframe(artifacts["backtest_summary"]),
        "trade_log": artifact_store.load_dataframe(artifacts["trade_log"]),
        "analysis": artifact_store.load_text(artifacts["analysis"]),
        "artifact_uris": artifacts,
    }
