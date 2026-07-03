from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

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
    out["long_agree_count"] = (out["long_exit_score"] >= out["short_exit_score"]).astype("int64")
    out["short_agree_count"] = (out["short_exit_score"] > out["long_exit_score"]).astype("int64")
    out["model_count"] = 1
    out["net_score"] = out["long_score"] - out["short_score"]
    return out


def strategy_source_name(source: str, family: str) -> str:
    return f"{source}.{family}"


def classification_probability_diagnostics(
    frame: pd.DataFrame,
    probability_frame: pd.DataFrame,
    *,
    target_col: str,
    labels: Iterable[str],
    n_bins: int = 10,
) -> dict[str, float | int]:
    """Return classifier metrics that are useful when probabilities drive trades."""

    if frame.empty:
        return {
            "rows": 0,
            "accuracy": np.nan,
            "balanced_accuracy": np.nan,
            "macro_f1": np.nan,
            "log_loss": np.nan,
            "brier_macro": np.nan,
            "expected_calibration_error": np.nan,
            "mean_confidence": np.nan,
        }
    label_list = [str(label) for label in labels]
    proba = ensure_probability_columns(probability_frame, label_list)
    class_cols = [f"prob__{label}" for label in label_list]
    probabilities = proba[class_cols].astype("float64").clip(0.0, 1.0)
    row_sum = probabilities.sum(axis=1).replace(0.0, np.nan)
    probabilities = probabilities.div(row_sum, axis=0).fillna(1.0 / max(1, len(class_cols)))

    y_true = frame[target_col].astype(str).to_numpy()
    predicted_idx = probabilities.to_numpy().argmax(axis=1)
    y_pred = np.asarray(label_list, dtype=object)[predicted_idx]
    confidence = probabilities.to_numpy().max(axis=1)
    correctness = (y_pred == y_true).astype("float64")

    brier_values = []
    for label in label_list:
        actual = (y_true == label).astype("float64")
        predicted = probabilities[f"prob__{label}"].to_numpy(dtype="float64")
        brier_values.append(float(np.mean((predicted - actual) ** 2)))

    return {
        "rows": int(len(frame)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "log_loss": _safe_log_loss(y_true, probabilities.to_numpy(dtype="float64"), label_list),
        "brier_macro": float(np.mean(brier_values)) if brier_values else np.nan,
        "expected_calibration_error": expected_calibration_error(confidence, correctness, n_bins=n_bins),
        "mean_confidence": float(np.mean(confidence)) if len(confidence) else np.nan,
    }


def expected_calibration_error(confidence: Iterable[float], correctness: Iterable[float], *, n_bins: int = 10) -> float:
    confidence_array = np.asarray(list(confidence), dtype="float64")
    correctness_array = np.asarray(list(correctness), dtype="float64")
    mask = np.isfinite(confidence_array) & np.isfinite(correctness_array)
    confidence_array = confidence_array[mask]
    correctness_array = correctness_array[mask]
    if len(confidence_array) == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, int(n_bins) + 1)
    ece = 0.0
    for left, right in zip(bins[:-1], bins[1:]):
        if right == 1.0:
            in_bin = (confidence_array >= left) & (confidence_array <= right)
        else:
            in_bin = (confidence_array >= left) & (confidence_array < right)
        if not in_bin.any():
            continue
        weight = float(in_bin.mean())
        ece += weight * abs(float(confidence_array[in_bin].mean()) - float(correctness_array[in_bin].mean()))
    return float(ece)


def model_vs_trading_summary(model_results: pd.DataFrame, backtest_summary: pd.DataFrame) -> pd.DataFrame:
    if model_results.empty or backtest_summary.empty:
        return pd.DataFrame()
    model_cols = [
        "source",
        "family",
        "strategy_source",
        "features",
        "train_rows",
        "oos_rows",
        "oos_accuracy",
        "oos_balanced_accuracy",
        "oos_macro_f1",
        "oos_log_loss",
        "oos_brier_macro",
        "oos_expected_calibration_error",
        "oos_mean_confidence",
    ]
    models = model_results.loc[model_results.get("status", "").eq("ok")].copy()
    if "strategy_source" not in models.columns:
        models["strategy_source"] = models.apply(lambda row: strategy_source_name(row["source"], row["family"]), axis=1)
    models = models[[col for col in model_cols if col in models.columns]].drop_duplicates("strategy_source")
    trading = (
        backtest_summary.loc[backtest_summary["strategy_source"].ne("ensemble_mean")]
        .sort_values(["strategy_source", "sharpe", "total_return"], ascending=[True, False, False])
        .groupby("strategy_source", as_index=False)
        .head(1)
    )
    trading_cols = [
        "strategy_source",
        "variant",
        "top_k",
        "total_return",
        "annualized_return",
        "annualized_vol",
        "sharpe",
        "max_drawdown",
        "win_rate",
        "signal_events",
        "avg_gross_exposure",
        "avg_net_exposure",
    ]
    out = models.merge(trading[[col for col in trading_cols if col in trading.columns]], on="strategy_source", how="left")
    return out.sort_values(["sharpe", "total_return"], ascending=False).reset_index(drop=True)


def metric_correlation_summary(frame: pd.DataFrame, *, x_cols: Iterable[str], y_cols: Iterable[str]) -> pd.DataFrame:
    rows = []
    if frame.empty:
        return pd.DataFrame(columns=["x", "y", "rows", "pearson", "spearman"])
    for x_col in x_cols:
        for y_col in y_cols:
            if x_col not in frame.columns or y_col not in frame.columns:
                continue
            pair = frame[[x_col, y_col]].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            rows.append(
                {
                    "x": x_col,
                    "y": y_col,
                    "rows": int(len(pair)),
                    "pearson": float(pair[x_col].corr(pair[y_col], method="pearson")) if len(pair) > 1 else np.nan,
                    "spearman": float(pair[x_col].corr(pair[y_col], method="spearman")) if len(pair) > 1 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _safe_log_loss(y_true: np.ndarray, probabilities: np.ndarray, labels: list[str]) -> float:
    if len(y_true) == 0 or probabilities.size == 0:
        return float("nan")
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    row_idx = []
    for value in y_true:
        idx = label_to_idx.get(str(value))
        if idx is None:
            return float("nan")
        row_idx.append(idx)
    clipped = np.clip(probabilities.astype("float64"), 1e-15, 1.0)
    selected = clipped[np.arange(len(row_idx)), np.asarray(row_idx, dtype=int)]
    return float(-np.log(selected).mean())


ML_TRADING_ARTIFACT_FILENAMES = {
    "model_results": "model_results.csv",
    "strategy_scores": "strategy_scores.csv",
    "backtest_summary": "backtest_summary.csv",
    "trade_log": "trade_log.csv",
    "model_vs_trading": "model_vs_trading.csv",
    "metric_correlations": "metric_correlations.csv",
    "yearly_backtest_summary": "yearly_backtest_summary.csv",
    "symbol_strategy_summary": "symbol_strategy_summary.csv",
    "symbol_robustness_summary": "symbol_robustness_summary.csv",
    "backtesting_py_symbol_validation": "backtesting_py_symbol_validation.csv",
    "phase_timings": "phase_timings.csv",
    "analysis": "analysis.md",
}


def write_ml_trading_artifact_files(
    *,
    model_results: pd.DataFrame,
    strategy_scores: pd.DataFrame,
    backtest_summary: pd.DataFrame,
    trade_log: pd.DataFrame,
    model_vs_trading: pd.DataFrame | None = None,
    metric_correlations: pd.DataFrame | None = None,
    yearly_backtest_summary: pd.DataFrame | None = None,
    symbol_strategy_summary: pd.DataFrame | None = None,
    symbol_robustness_summary: pd.DataFrame | None = None,
    backtesting_py_symbol_validation: pd.DataFrame | None = None,
    phase_timings: pd.DataFrame | None = None,
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
        "model_vs_trading": output_dir / ML_TRADING_ARTIFACT_FILENAMES["model_vs_trading"],
        "metric_correlations": output_dir / ML_TRADING_ARTIFACT_FILENAMES["metric_correlations"],
        "yearly_backtest_summary": output_dir / ML_TRADING_ARTIFACT_FILENAMES["yearly_backtest_summary"],
        "symbol_strategy_summary": output_dir / ML_TRADING_ARTIFACT_FILENAMES["symbol_strategy_summary"],
        "symbol_robustness_summary": output_dir / ML_TRADING_ARTIFACT_FILENAMES["symbol_robustness_summary"],
        "backtesting_py_symbol_validation": output_dir / ML_TRADING_ARTIFACT_FILENAMES["backtesting_py_symbol_validation"],
        "phase_timings": output_dir / ML_TRADING_ARTIFACT_FILENAMES["phase_timings"],
        "analysis": output_dir / ML_TRADING_ARTIFACT_FILENAMES["analysis"],
    }
    model_results.to_csv(paths["model_results"], index=False)
    strategy_scores.to_csv(paths["strategy_scores"], index=False)
    backtest_summary.to_csv(paths["backtest_summary"], index=False)
    trade_log.to_csv(paths["trade_log"], index=False)
    (model_vs_trading if model_vs_trading is not None else pd.DataFrame()).to_csv(paths["model_vs_trading"], index=False)
    (metric_correlations if metric_correlations is not None else pd.DataFrame()).to_csv(paths["metric_correlations"], index=False)
    (yearly_backtest_summary if yearly_backtest_summary is not None else pd.DataFrame()).to_csv(paths["yearly_backtest_summary"], index=False)
    (symbol_strategy_summary if symbol_strategy_summary is not None else pd.DataFrame()).to_csv(
        paths["symbol_strategy_summary"],
        index=False,
    )
    (symbol_robustness_summary if symbol_robustness_summary is not None else pd.DataFrame()).to_csv(
        paths["symbol_robustness_summary"],
        index=False,
    )
    (backtesting_py_symbol_validation if backtesting_py_symbol_validation is not None else pd.DataFrame()).to_csv(
        paths["backtesting_py_symbol_validation"],
        index=False,
    )
    (phase_timings if phase_timings is not None else pd.DataFrame()).to_csv(paths["phase_timings"], index=False)
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
            "model_vs_trading": _read_optional_csv(local_dir / ML_TRADING_ARTIFACT_FILENAMES["model_vs_trading"]),
            "metric_correlations": _read_optional_csv(local_dir / ML_TRADING_ARTIFACT_FILENAMES["metric_correlations"]),
            "yearly_backtest_summary": _read_optional_csv(local_dir / ML_TRADING_ARTIFACT_FILENAMES["yearly_backtest_summary"]),
            "symbol_strategy_summary": _read_optional_csv(local_dir / ML_TRADING_ARTIFACT_FILENAMES["symbol_strategy_summary"]),
            "symbol_robustness_summary": _read_optional_csv(local_dir / ML_TRADING_ARTIFACT_FILENAMES["symbol_robustness_summary"]),
            "backtesting_py_symbol_validation": _read_optional_csv(local_dir / ML_TRADING_ARTIFACT_FILENAMES["backtesting_py_symbol_validation"]),
            "phase_timings": _read_optional_csv(local_dir / ML_TRADING_ARTIFACT_FILENAMES["phase_timings"]),
            "analysis": (local_dir / ML_TRADING_ARTIFACT_FILENAMES["analysis"]).read_text(encoding="utf-8"),
        }


def _read_optional_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() and path.stat().st_size > 0 else pd.DataFrame()
