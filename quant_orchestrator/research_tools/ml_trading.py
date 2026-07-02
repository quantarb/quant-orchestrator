from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Iterable

import numpy as np
import pandas as pd

from quant_orchestrator.platforms.ml_frameworks.rapids.random_forest import ensure_probability_columns
from quant_orchestrator.tracking import DEFAULT_TRACKING_URI


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


ML_TRADING_ARTIFACT_FILENAMES = {
    "model_results": "model_results.csv",
    "strategy_scores": "strategy_scores.csv",
    "backtest_summary": "backtest_summary.csv",
    "trade_log": "trade_log.csv",
    "analysis": "analysis.md",
}


def write_ml_trading_artifact_files(
    *,
    model_results: pd.DataFrame,
    strategy_scores: pd.DataFrame,
    backtest_summary: pd.DataFrame,
    trade_log: pd.DataFrame,
    analysis_markdown: str,
    directory: str | Path,
) -> dict[str, Path]:
    output_dir = Path(directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "model_results": output_dir / ML_TRADING_ARTIFACT_FILENAMES["model_results"],
        "strategy_scores": output_dir / ML_TRADING_ARTIFACT_FILENAMES["strategy_scores"],
        "backtest_summary": output_dir / ML_TRADING_ARTIFACT_FILENAMES["backtest_summary"],
        "trade_log": output_dir / ML_TRADING_ARTIFACT_FILENAMES["trade_log"],
        "analysis": output_dir / ML_TRADING_ARTIFACT_FILENAMES["analysis"],
    }
    model_results.to_csv(paths["model_results"], index=False)
    strategy_scores.to_csv(paths["strategy_scores"], index=False)
    backtest_summary.to_csv(paths["backtest_summary"], index=False)
    trade_log.to_csv(paths["trade_log"], index=False)
    paths["analysis"].write_text(analysis_markdown, encoding="utf-8")
    return paths


def load_latest_mlflow_experiment_artifacts(
    experiment_name: str,
    *,
    mlflow_experiment: str = "ml_trading",
    tracking_uri: str | None = None,
) -> dict[str, pd.DataFrame | str]:
    import mlflow

    mlflow.set_tracking_uri(tracking_uri or DEFAULT_TRACKING_URI)
    experiment = mlflow.get_experiment_by_name(mlflow_experiment)
    if experiment is None:
        raise FileNotFoundError(f"MLflow experiment not found: {mlflow_experiment}")
    runs = mlflow.search_runs(
        [experiment.experiment_id],
        filter_string=f"tags.`quant_orchestrator.experiment_name` = '{experiment_name}'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if runs.empty:
        raise FileNotFoundError(f"No MLflow run found for experiment_name={experiment_name!r}")
    run_id = str(runs.iloc[0]["run_id"])
    return load_mlflow_run_artifacts(run_id, tracking_uri=tracking_uri)


def load_mlflow_run_artifacts(
    run_id: str,
    *,
    tracking_uri: str | None = None,
) -> dict[str, pd.DataFrame | str]:
    import mlflow

    mlflow.set_tracking_uri(tracking_uri or DEFAULT_TRACKING_URI)
    with tempfile.TemporaryDirectory(prefix="quant-orchestrator-mlflow-") as tmp_dir:
        local_dir = Path(mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="ml_trading", dst_path=tmp_dir))
        return {
            "run_id": run_id,
            "model_results": pd.read_csv(local_dir / ML_TRADING_ARTIFACT_FILENAMES["model_results"]),
            "strategy_scores": pd.read_csv(local_dir / ML_TRADING_ARTIFACT_FILENAMES["strategy_scores"]),
            "backtest_summary": pd.read_csv(local_dir / ML_TRADING_ARTIFACT_FILENAMES["backtest_summary"]),
            "trade_log": pd.read_csv(local_dir / ML_TRADING_ARTIFACT_FILENAMES["trade_log"]),
            "analysis": (local_dir / ML_TRADING_ARTIFACT_FILENAMES["analysis"]).read_text(encoding="utf-8"),
        }
