from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys
from time import perf_counter

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quant_orchestrator.platforms.backtesting_frameworks.shared_book import (
    build_shared_book_weights,
    run_shared_book_backtest,
    shared_book_performance_metrics,
)
from quant_orchestrator.platforms.ml_frameworks.rapids import RapidsRandomForestClassifier
from quant_orchestrator.research_tools.ml_trading import (
    build_family_prediction_frame,
    build_strategy_score_frame,
    classification_probability_diagnostics,
    prepare_family_dataset,
)
from quant_orchestrator.research_tools.ml_trading_experiment import (
    MLTradingExperimentConfig,
    _load_price_frames,
    _prepare_quant_warehouse_import,
    _warehouse_imports,
)


SOURCE = "cross_sectional_autoencoder_ablation"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_dir")
    parser.add_argument("--oos-start", default="2021-01-01")
    parser.add_argument("--train-end", default="2020-12-31")
    parser.add_argument("--provider", default="fmp")
    parser.add_argument("--price-start", default="1900-01-01")
    parser.add_argument("--cost-bps", type=float, default=5.5)
    parser.add_argument("--capital-base", type=float, default=1_000_000.0)
    parser.add_argument("--top-k", default="5,10,20,40")
    parser.add_argument("--random-seed", type=int, default=20260702)
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    metadata = json.loads((artifact_dir / "metadata.json").read_text())
    symbols = tuple(str(symbol).upper() for symbol in metadata["symbols"])
    train_end = pd.Timestamp(args.train_end)
    oos_start = pd.Timestamp(args.oos_start)
    top_k_values = tuple(int(value) for value in str(args.top_k).split(",") if str(value).strip())

    feature_panel = pd.read_parquet(artifact_dir / "feature_panel.parquet")
    feature_panel["symbol"] = feature_panel["symbol"].astype(str).str.upper()
    feature_panel["date"] = pd.to_datetime(feature_panel["date"], errors="coerce").dt.normalize()
    label_rows = pd.read_csv(artifact_dir / "label_rows.csv")
    label_rows["symbol"] = label_rows["symbol"].astype(str).str.upper()
    label_rows["date"] = pd.to_datetime(label_rows["date"], errors="coerce").dt.normalize()

    _prepare_quant_warehouse_import("/home/jlee153232/PycharmProjects/quant-warehouse")
    Warehouse, *_ = _warehouse_imports()
    warehouse = Warehouse()
    price_frames = _load_price_frames(
        warehouse,
        symbols,
        provider=args.provider,
        start=args.price_start,
        end=None,
    )
    wide_close = pd.DataFrame({symbol: frame["close"] for symbol, frame in price_frames.items()}).sort_index().ffill()
    next_returns = wide_close.pct_change().shift(-1)

    out_dir = artifact_dir / "ablations"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_rows = []
    return_rows = []
    importance_frames = []

    for feature_set, features in _feature_sets(feature_panel).items():
        if not features:
            continue
        started = perf_counter()
        family = feature_set
        feature_metadata = pd.DataFrame({"source": SOURCE, "family": family, "feature": features})
        dataset, classifier_features = prepare_family_dataset(
            feature_panel[["symbol", "date", *features]],
            feature_metadata,
            label_rows,
            source=SOURCE,
            family=family,
            min_feature_coverage=0.50,
        )
        train = dataset.loc[pd.to_datetime(dataset["date"]).le(train_end)].copy()
        oos = dataset.loc[pd.to_datetime(dataset["date"]).ge(oos_start)].copy()
        if len(train) < 250 or train["collapsed_label"].nunique() < 2:
            model_rows.append(
                {
                    "feature_set": feature_set,
                    "status": "skipped_sparse_train",
                    "features": len(classifier_features),
                    "rows": len(dataset),
                    "train_rows": len(train),
                    "oos_rows": len(oos),
                    "elapsed_seconds": perf_counter() - started,
                }
            )
            continue

        classifier = RapidsRandomForestClassifier.fit(
            train,
            features=classifier_features,
            target_col="collapsed_label",
            random_state=int(args.random_seed),
            params=MLTradingExperimentConfig().rf_params,
        )
        train_scores = classification_probability_diagnostics(
            train,
            classifier.predict_proba_frame(train, classifier_features),
            target_col="collapsed_label",
            labels=classifier.encoder.classes_,
        )
        oos_scores = (
            classification_probability_diagnostics(
                oos,
                classifier.predict_proba_frame(oos, classifier_features),
                target_col="collapsed_label",
                labels=classifier.encoder.classes_,
            )
            if not oos.empty and oos["collapsed_label"].nunique() > 1
            else _empty_scores(len(oos))
        )

        set_dir = out_dir / feature_set
        set_dir.mkdir(parents=True, exist_ok=True)
        with (set_dir / "classifier.pkl").open("wb") as handle:
            pickle.dump(classifier, handle, protocol=pickle.HIGHEST_PROTOCOL)
        importance = _feature_importances(classifier, classifier_features)
        importance.insert(0, "feature_set", feature_set)
        importance.to_csv(set_dir / "feature_importance.csv", index=False)
        importance_frames.append(importance)

        strategy_scores = _strategy_scores(
            feature_panel,
            classifier,
            classifier_features,
            source=SOURCE,
            family=family,
            oos_start=oos_start,
        )
        strategy_scores.to_parquet(set_dir / "strategy_scores.parquet", index=False)
        returns = _long_only_returns(
            strategy_scores,
            next_returns,
            top_k_values=top_k_values,
            cost_bps=float(args.cost_bps),
            capital_base=float(args.capital_base),
            source=SOURCE,
            family=family,
        )
        returns.insert(0, "feature_set", feature_set)
        returns.to_csv(set_dir / "backtest_summary_long_only.csv", index=False)
        return_rows.extend(returns.to_dict("records"))

        model_rows.append(
            {
                "feature_set": feature_set,
                "source": SOURCE,
                "family": family,
                "status": "ok",
                "features": len(classifier_features),
                "rows": len(dataset),
                "train_rows": len(train),
                "oos_rows": len(oos),
                "classes": dataset["collapsed_label"].nunique(),
                **{f"train_{key}": value for key, value in train_scores.items()},
                **{f"oos_{key}": value for key, value in oos_scores.items()},
                "elapsed_seconds": perf_counter() - started,
            }
        )

    model_results = pd.DataFrame(model_rows)
    return_summary = pd.DataFrame(return_rows)
    feature_importance = pd.concat(importance_frames, ignore_index=True) if importance_frames else pd.DataFrame()
    model_results.to_csv(out_dir / "model_results.csv", index=False)
    return_summary.to_csv(out_dir / "backtest_summary_long_only.csv", index=False)
    feature_importance.to_csv(out_dir / "feature_importance.csv", index=False)

    print("artifact_dir", out_dir)
    print("\nClassifier metrics:")
    print(
        model_results.sort_values("oos_macro_f1", ascending=False)[
            ["feature_set", "features", "train_rows", "oos_rows", "oos_macro_f1", "oos_balanced_accuracy"]
        ].to_string(index=False)
    )
    print("\nLong-only returns:")
    print(
        return_summary.sort_values(["sharpe", "total_return"], ascending=False)[
            ["feature_set", "top_k", "total_return", "annualized_return", "sharpe", "max_drawdown", "trades"]
        ].to_string(index=False)
    )


def _feature_sets(feature_panel: pd.DataFrame) -> dict[str, list[str]]:
    cols = [col for col in feature_panel.columns if col not in {"symbol", "date"}]
    return {
        "raw_returns": [col for col in cols if col.startswith("csra_return_")],
        "recon": [col for col in cols if col.startswith("csra_recon_")],
        "signed_residuals": [col for col in cols if col.startswith("csra_residual_")],
        "abs_residuals": [col for col in cols if col.startswith("csra_abs_residual_")],
        "residual_summary": [
            col
            for col in ("csra_signed_residual", "csra_signed_sq_score", "csra_residual_norm")
            if col in feature_panel.columns
        ],
        "all_csra": cols,
    }


def _strategy_scores(
    feature_panel: pd.DataFrame,
    classifier: RapidsRandomForestClassifier,
    features: list[str],
    *,
    source: str,
    family: str,
    oos_start: pd.Timestamp,
) -> pd.DataFrame:
    oos_features = feature_panel.loc[pd.to_datetime(feature_panel["date"]).ge(oos_start), ["symbol", "date", *features]].copy()
    prediction_frame = build_family_prediction_frame(
        oos_features,
        features,
        min_feature_coverage=0.50,
    )
    proba = classifier.predict_proba_frame(prediction_frame, features)
    return build_strategy_score_frame(
        source=source,
        family=family,
        prediction_frame=prediction_frame,
        probability_frame=proba,
        ae_familiarity_frame=None,
        apply_ae_to_exits=False,
    )


def _long_only_returns(
    strategy_scores: pd.DataFrame,
    next_returns: pd.DataFrame,
    *,
    top_k_values: tuple[int, ...],
    cost_bps: float,
    capital_base: float,
    source: str,
    family: str,
) -> pd.DataFrame:
    dates = pd.DatetimeIndex(sorted(set(strategy_scores["date"]).intersection(next_returns.index)))
    symbols = tuple(sorted(set(strategy_scores["symbol"]).intersection(next_returns.columns)))
    rows = []
    for top_k in top_k_values:
        weights, trades = build_shared_book_weights(
            strategy_scores,
            symbols,
            dates,
            top_k=top_k,
            variant="long_only",
            entry_threshold=0.50,
            exit_threshold=0.50,
            long_exit_score_col="long_exit_score",
            short_exit_score_col="short_exit_score",
        )
        returns, equity, _turnover = run_shared_book_backtest(
            weights,
            next_returns,
            cost_bps=cost_bps,
            capital_base=capital_base,
        )
        row = shared_book_performance_metrics(
            returns,
            equity,
            weights,
            trades,
            framework="vectorized_shared_book",
            variant="long_only",
            top_k=top_k,
            cost_bps=cost_bps,
        )
        row.update(
            {
                "source": source,
                "family": family,
                "symbols": len(symbols),
                "score_rows": len(strategy_scores),
                "score_dates": strategy_scores["date"].nunique(),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _feature_importances(classifier: RapidsRandomForestClassifier, features: list[str]) -> pd.DataFrame:
    values = _to_numpy(getattr(classifier.model, "feature_importances_", None))
    if values is None:
        values = _to_numpy(getattr(classifier.model, "feature_importances", None))
    if values is None:
        raise ValueError("Classifier model does not expose feature_importances_")
    values = values[: len(features)]
    total = float(np.nansum(values))
    normalized = values / total if total else values
    out = pd.DataFrame(
        {
            "feature": features[: len(values)],
            "importance": values.astype("float64"),
            "importance_norm": normalized.astype("float64"),
        }
    )
    out["rank"] = out["importance_norm"].rank(method="first", ascending=False).astype("int64")
    return out.sort_values("rank").reset_index(drop=True)


def _to_numpy(value) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "to_numpy"):
        try:
            return np.asarray(value.to_numpy()).ravel()
        except Exception:
            pass
    try:
        import cupy as cp

        if isinstance(value, cp.ndarray):
            return cp.asnumpy(value).ravel()
    except Exception:
        pass
    try:
        return np.asarray(value).ravel()
    except Exception:
        return None


def _empty_scores(rows: int) -> dict[str, float | int]:
    return {
        "rows": int(rows),
        "accuracy": np.nan,
        "balanced_accuracy": np.nan,
        "macro_f1": np.nan,
        "log_loss": np.nan,
        "brier_macro": np.nan,
        "expected_calibration_error": np.nan,
        "mean_confidence": np.nan,
    }


if __name__ == "__main__":
    main()
