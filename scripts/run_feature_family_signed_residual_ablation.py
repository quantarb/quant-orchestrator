from __future__ import annotations

import argparse
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
from quant_orchestrator.platforms.ml_frameworks.torch_autoencoder import (
    LatentAutoencoderConfig,
    LatentAutoencoderIndex,
)
from quant_orchestrator.platforms.ml_frameworks.torch_autoencoder.latent_index import _standardize_apply
from quant_orchestrator.research_tools.ml_trading import (
    build_family_prediction_frame,
    build_strategy_score_frame,
    classification_probability_diagnostics,
    prepare_family_dataset,
)
from quant_orchestrator.research_tools.ml_trading_experiment import (
    MLTradingExperimentConfig,
    _build_oracle_trade_label_rows_sparse,
    _load_price_frames,
    _prepare_quant_warehouse_import,
    _warehouse_imports,
)


DEFAULT_SOURCES = (
    "fmp.fmp_cash_mcap",
    "fmp.fmp_income_mcap",
    "fmp.fmp_balance_mcap",
    "fmp.fmp_daily_mcap_yield",
    "fmp.fmp_daily_ev_yield",
    "financetoolkit.ft_ratios_valuation",
    "financetoolkit.ft_ratios_profitability",
    "financetoolkit.ft_growth_balance",
    "financetoolkit.ft_growth_cash",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", default="artifacts/feature_family_signed_residual_ablation_100b")
    parser.add_argument("--min-market-cap", type=int, default=100_000_000_000)
    parser.add_argument("--provider", default="fmp")
    parser.add_argument("--start-date", default="1900-01-01")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--train-end", default="2020-12-31")
    parser.add_argument("--oos-start", default="2021-01-01")
    parser.add_argument("--oracle-frequency", default="YE")
    parser.add_argument("--oracle-k-min", type=int, default=1)
    parser.add_argument("--oracle-k-max", type=int, default=12)
    parser.add_argument("--strategy-sources", default=",".join(DEFAULT_SOURCES))
    parser.add_argument("--max-features-per-family", type=int, default=50)
    parser.add_argument("--min-feature-coverage", type=float, default=0.50)
    parser.add_argument("--top-k", default="5,10,20,40")
    parser.add_argument("--cost-bps", type=float, default=5.5)
    parser.add_argument("--capital-base", type=float, default=1_000_000.0)
    parser.add_argument("--random-seed", type=int, default=20260702)
    parser.add_argument("--max-ae-train-rows", type=int, default=100_000)
    parser.add_argument("--ae-epochs", type=int, default=12)
    args = parser.parse_args()

    started = perf_counter()
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    train_end = pd.Timestamp(args.train_end)
    oos_start = pd.Timestamp(args.oos_start)
    end_date = str(args.end_date).strip() or None
    wanted_sources = tuple(source.strip() for source in args.strategy_sources.split(",") if source.strip())
    top_k_values = tuple(int(value) for value in str(args.top_k).split(",") if str(value).strip())

    _prepare_quant_warehouse_import("/home/jlee153232/PycharmProjects/quant-warehouse")
    (
        Warehouse,
        _EventPairStore,
        BinaryTargetConfig,
        FamilyEvaluationConfig,
        _build_collapsed_bullish_event_target_panel,
        build_fundamental_feature_panel,
        _build_oracle_trade_target_panel,
        cap_features_by_quality,
        _load_fmp_event_pairs,
        screen_fmp_equity_universe,
    ) = _warehouse_imports()
    warehouse = Warehouse()
    feature_config = FamilyEvaluationConfig(
        provider=args.provider,
        market_cap_min=int(args.min_market_cap),
        start_date=args.start_date,
        end_date=end_date,
        max_features_per_family=int(args.max_features_per_family),
    )
    symbols, raw_universe, universe_eligibility, universe_source = screen_fmp_equity_universe(
        feature_config,
        warehouse=warehouse,
    )
    raw_feature_panel, raw_feature_metadata, feature_diagnostics, feature_timings = build_fundamental_feature_panel(
        symbols,
        feature_config,
        warehouse=warehouse,
    )
    selected_features, selected_feature_metadata, feature_quality = cap_features_by_quality(
        raw_feature_panel,
        raw_feature_metadata,
        max_features=int(args.max_features_per_family),
    )
    selected_feature_metadata = selected_feature_metadata.copy()
    selected_feature_metadata["strategy_source"] = (
        selected_feature_metadata["source"].astype(str) + "." + selected_feature_metadata["family"].astype(str)
    )
    if wanted_sources:
        selected_feature_metadata = selected_feature_metadata.loc[
            selected_feature_metadata["strategy_source"].isin(wanted_sources)
        ].copy()
        selected_features = [
            feature for feature in selected_features if feature in set(selected_feature_metadata["feature"].astype(str))
        ]
    feature_panel = raw_feature_panel[["symbol", "date", *selected_features]].copy()
    feature_panel["symbol"] = feature_panel["symbol"].astype(str).str.upper()
    feature_panel["date"] = pd.to_datetime(feature_panel["date"], errors="coerce").dt.normalize()

    target_config = BinaryTargetConfig(
        provider=args.provider,
        start_date=args.start_date,
        end_date=end_date,
        oracle_trade_k_by_frequency={
            str(args.oracle_frequency).upper(): tuple(range(int(args.oracle_k_min), int(args.oracle_k_max) + 1))
        },
    )
    label_rows, label_diagnostics, oracle_seconds, _oracle_unique_windows = _build_oracle_trade_label_rows_sparse(
        symbols,
        target_config,
        warehouse=warehouse,
    )
    price_frames = _load_price_frames(
        warehouse,
        symbols,
        provider=args.provider,
        start=args.start_date,
        end=end_date,
    )
    wide_close = pd.DataFrame({symbol: frame["close"] for symbol, frame in price_frames.items()}).sort_index().ffill()
    next_returns = wide_close.pct_change().shift(-1)

    model_rows = []
    return_rows = []
    importance_rows = []
    for source, family in (
        selected_feature_metadata[["source", "family"]]
        .drop_duplicates()
        .sort_values(["source", "family"])
        .itertuples(index=False, name=None)
    ):
        family_source = f"{source}.{family}"
        family_meta = selected_feature_metadata.loc[
            selected_feature_metadata["source"].astype(str).eq(str(source))
            & selected_feature_metadata["family"].astype(str).eq(str(family))
        ]
        family_features = [feature for feature in family_meta["feature"].astype(str).tolist() if feature in feature_panel]
        if not family_features:
            continue
        family_dir = artifact_dir / _safe_filename(family_source)
        family_dir.mkdir(parents=True, exist_ok=True)
        raw_panel = feature_panel[["symbol", "date", *family_features]].copy()
        family_model_rows, family_return_rows, family_importance_rows = _run_family(
            raw_panel,
            label_rows,
            next_returns,
            source=str(source),
            family=str(family),
            raw_features=family_features,
            train_end=train_end,
            oos_start=oos_start,
            min_feature_coverage=float(args.min_feature_coverage),
            random_seed=int(args.random_seed),
            top_k_values=top_k_values,
            cost_bps=float(args.cost_bps),
            capital_base=float(args.capital_base),
            artifact_dir=family_dir,
            max_ae_train_rows=int(args.max_ae_train_rows),
            ae_epochs=int(args.ae_epochs),
        )
        model_rows.extend(family_model_rows)
        return_rows.extend(family_return_rows)
        importance_rows.extend(family_importance_rows)

    model_results = pd.DataFrame(model_rows)
    return_summary = pd.DataFrame(return_rows)
    feature_importance = pd.DataFrame(importance_rows)
    model_results.to_csv(artifact_dir / "model_results.csv", index=False)
    return_summary.to_csv(artifact_dir / "backtest_summary_long_only.csv", index=False)
    feature_importance.to_csv(artifact_dir / "feature_importance.csv", index=False)
    raw_universe.to_csv(artifact_dir / "raw_universe.csv", index=False)
    universe_eligibility.to_csv(artifact_dir / "universe_eligibility.csv", index=False)
    feature_diagnostics.to_csv(artifact_dir / "feature_diagnostics.csv", index=False)
    feature_quality.to_csv(artifact_dir / "feature_quality.csv", index=False)
    label_diagnostics.to_csv(artifact_dir / "label_diagnostics.csv", index=False)
    (artifact_dir / "run_metadata.json").write_text(
        pd.Series(
            {
                "universe_source": universe_source,
                "symbols": len(symbols),
                "strategy_sources": wanted_sources,
                "oracle_frequency": args.oracle_frequency,
                "oracle_k_min": args.oracle_k_min,
                "oracle_k_max": args.oracle_k_max,
                "oracle_seconds": oracle_seconds,
                "feature_timings": feature_timings,
                "elapsed_seconds": perf_counter() - started,
            }
        ).to_json(indent=2),
        encoding="utf-8",
    )

    print("artifact_dir", artifact_dir)
    if not model_results.empty:
        print("\nClassifier metrics:")
        print(
            model_results.sort_values(["family", "representation"])[
                [
                    "strategy_source",
                    "representation",
                    "features",
                    "train_rows",
                    "oos_rows",
                    "oos_macro_f1",
                    "oos_balanced_accuracy",
                ]
            ].to_string(index=False)
        )
    if not return_summary.empty:
        best = return_summary.loc[return_summary.groupby(["strategy_source", "representation"])["sharpe"].idxmax()]
        print("\nBest long-only return per family/representation:")
        print(
            best.sort_values(["strategy_source", "representation"])[
                ["strategy_source", "representation", "top_k", "total_return", "annualized_return", "sharpe", "max_drawdown"]
            ].to_string(index=False)
        )


def _run_family(
    raw_panel: pd.DataFrame,
    label_rows: pd.DataFrame,
    next_returns: pd.DataFrame,
    *,
    source: str,
    family: str,
    raw_features: list[str],
    train_end: pd.Timestamp,
    oos_start: pd.Timestamp,
    min_feature_coverage: float,
    random_seed: int,
    top_k_values: tuple[int, ...],
    cost_bps: float,
    capital_base: float,
    artifact_dir: Path,
    max_ae_train_rows: int,
    ae_epochs: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    model_rows = []
    return_rows = []
    importance_rows = []
    raw_metadata = pd.DataFrame({"source": source, "family": family, "feature": raw_features})

    raw_result = _train_score_return(
        raw_panel,
        raw_metadata,
        label_rows,
        next_returns,
        source=source,
        family=family,
        representation="raw",
        train_end=train_end,
        oos_start=oos_start,
        min_feature_coverage=min_feature_coverage,
        random_seed=random_seed,
        top_k_values=top_k_values,
        cost_bps=cost_bps,
        capital_base=capital_base,
        artifact_dir=artifact_dir / "raw",
    )
    model_rows.extend(raw_result[0])
    return_rows.extend(raw_result[1])
    importance_rows.extend(raw_result[2])

    residual_panel, residual_features, ae_metadata = _signed_residual_panel(
        raw_panel,
        raw_features,
        train_end=train_end,
        max_train_rows=max_ae_train_rows,
        epochs=ae_epochs,
        random_seed=random_seed,
    )
    residual_metadata = pd.DataFrame({"source": source, "family": family, "feature": residual_features})
    residual_result = _train_score_return(
        residual_panel,
        residual_metadata,
        label_rows,
        next_returns,
        source=source,
        family=family,
        representation="signed_residual",
        train_end=train_end,
        oos_start=oos_start,
        min_feature_coverage=min_feature_coverage,
        random_seed=random_seed,
        top_k_values=top_k_values,
        cost_bps=cost_bps,
        capital_base=capital_base,
        artifact_dir=artifact_dir / "signed_residual",
        extra_model_metadata=ae_metadata,
    )
    model_rows.extend(residual_result[0])
    return_rows.extend(residual_result[1])
    importance_rows.extend(residual_result[2])

    combined_panel = raw_panel.merge(residual_panel, on=["symbol", "date"], how="inner")
    combined_features = [*raw_features, *residual_features]
    combined_metadata = pd.DataFrame({"source": source, "family": family, "feature": combined_features})
    combined_result = _train_score_return(
        combined_panel,
        combined_metadata,
        label_rows,
        next_returns,
        source=source,
        family=family,
        representation="raw_plus_signed_residual",
        train_end=train_end,
        oos_start=oos_start,
        min_feature_coverage=min_feature_coverage,
        random_seed=random_seed,
        top_k_values=top_k_values,
        cost_bps=cost_bps,
        capital_base=capital_base,
        artifact_dir=artifact_dir / "raw_plus_signed_residual",
        extra_model_metadata=ae_metadata,
    )
    model_rows.extend(combined_result[0])
    return_rows.extend(combined_result[1])
    importance_rows.extend(combined_result[2])
    return model_rows, return_rows, importance_rows


def _signed_residual_panel(
    raw_panel: pd.DataFrame,
    raw_features: list[str],
    *,
    train_end: pd.Timestamp,
    max_train_rows: int,
    epochs: int,
    random_seed: int,
) -> tuple[pd.DataFrame, list[str], dict[str, object]]:
    train_frame = raw_panel.loc[pd.to_datetime(raw_panel["date"]).le(train_end)].copy()
    if int(max_train_rows) > 0 and len(train_frame) > int(max_train_rows):
        train_frame = train_frame.sample(n=int(max_train_rows), random_state=int(random_seed)).sort_values(["date", "symbol"])
    ae = LatentAutoencoderIndex.fit(
        train_frame,
        features=raw_features,
        config=LatentAutoencoderConfig(
            epochs=int(epochs),
            tuning_epochs=0,
            batch_size=4096,
            tune_architecture=False,
        ),
    )
    if ae is None:
        raise ValueError("Could not train autoencoder for signed residual representation")
    residual = _standardized_reconstruction_residual(ae, raw_panel)
    residual_features = [f"resid__{feature}" for feature in raw_features]
    out = raw_panel[["symbol", "date"]].copy()
    for idx, feature in enumerate(residual_features):
        out[feature] = residual[:, idx].astype("float32")
    return out, residual_features, ae.metadata()


def _standardized_reconstruction_residual(ae: LatentAutoencoderIndex, frame: pd.DataFrame) -> np.ndarray:
    import torch

    x = _standardize_apply(frame, ae.features, ae.center, ae.scale, ae.lower, ae.upper)
    ae.model.eval()
    batches = []
    with torch.no_grad():
        for start in range(0, len(x), ae.config.batch_size):
            batch = torch.tensor(x[start : start + ae.config.batch_size], dtype=torch.float32, device=ae.device)
            recon = ae.model(batch).detach().cpu().numpy().astype("float32")
            batches.append(recon)
    reconstructed = np.vstack(batches) if batches else np.empty_like(x)
    return (x - reconstructed).astype("float32")


def _train_score_return(
    feature_panel: pd.DataFrame,
    feature_metadata: pd.DataFrame,
    label_rows: pd.DataFrame,
    next_returns: pd.DataFrame,
    *,
    source: str,
    family: str,
    representation: str,
    train_end: pd.Timestamp,
    oos_start: pd.Timestamp,
    min_feature_coverage: float,
    random_seed: int,
    top_k_values: tuple[int, ...],
    cost_bps: float,
    capital_base: float,
    artifact_dir: Path,
    extra_model_metadata: dict[str, object] | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    dataset, features = prepare_family_dataset(
        feature_panel,
        feature_metadata,
        label_rows,
        source=source,
        family=family,
        min_feature_coverage=min_feature_coverage,
    )
    train = dataset.loc[pd.to_datetime(dataset["date"]).le(train_end)].copy()
    oos = dataset.loc[pd.to_datetime(dataset["date"]).ge(oos_start)].copy()
    strategy_source = f"{source}.{family}.{representation}"
    if len(train) < 250 or train["collapsed_label"].nunique() < 2:
        return (
            [
                {
                    "strategy_source": strategy_source,
                    "source": source,
                    "family": family,
                    "representation": representation,
                    "status": "skipped_sparse_train",
                    "features": len(features),
                    "rows": len(dataset),
                    "train_rows": len(train),
                    "oos_rows": len(oos),
                }
            ],
            [],
            [],
        )
    classifier = RapidsRandomForestClassifier.fit(
        train,
        features=features,
        target_col="collapsed_label",
        random_state=random_seed,
        params=MLTradingExperimentConfig().rf_params,
    )
    train_scores = classification_probability_diagnostics(
        train,
        classifier.predict_proba_frame(train, features),
        target_col="collapsed_label",
        labels=classifier.encoder.classes_,
    )
    oos_scores = (
        classification_probability_diagnostics(
            oos,
            classifier.predict_proba_frame(oos, features),
            target_col="collapsed_label",
            labels=classifier.encoder.classes_,
        )
        if not oos.empty and oos["collapsed_label"].nunique() > 1
        else _empty_scores(len(oos))
    )
    with (artifact_dir / "classifier.pkl").open("wb") as handle:
        pickle.dump(classifier, handle, protocol=pickle.HIGHEST_PROTOCOL)
    importance = _feature_importances(classifier, features)
    importance.insert(0, "strategy_source", strategy_source)
    importance.insert(1, "source", source)
    importance.insert(2, "family", family)
    importance.insert(3, "representation", representation)
    importance.to_csv(artifact_dir / "feature_importance.csv", index=False)

    strategy_scores = _strategy_scores(
        feature_panel,
        classifier,
        features,
        source=source,
        family=family,
        representation=representation,
        oos_start=oos_start,
        min_feature_coverage=min_feature_coverage,
    )
    strategy_scores.to_parquet(artifact_dir / "strategy_scores.parquet", index=False)
    returns = _long_only_returns(
        strategy_scores,
        next_returns,
        top_k_values=top_k_values,
        cost_bps=cost_bps,
        capital_base=capital_base,
    )
    returns.insert(0, "strategy_source", strategy_source)
    returns.insert(1, "source", source)
    returns.insert(2, "family", family)
    returns.insert(3, "representation", representation)
    returns.to_csv(artifact_dir / "backtest_summary_long_only.csv", index=False)

    row = {
        "strategy_source": strategy_source,
        "source": source,
        "family": family,
        "representation": representation,
        "status": "ok",
        "features": len(features),
        "rows": len(dataset),
        "train_rows": len(train),
        "oos_rows": len(oos),
        "classes": dataset["collapsed_label"].nunique(),
        **{f"train_{key}": value for key, value in train_scores.items()},
        **{f"oos_{key}": value for key, value in oos_scores.items()},
    }
    row.update(extra_model_metadata or {})
    pd.DataFrame([row]).to_csv(artifact_dir / "model_results.csv", index=False)
    return [row], returns.to_dict("records"), importance.to_dict("records")


def _strategy_scores(
    feature_panel: pd.DataFrame,
    classifier: RapidsRandomForestClassifier,
    features: list[str],
    *,
    source: str,
    family: str,
    representation: str,
    oos_start: pd.Timestamp,
    min_feature_coverage: float,
) -> pd.DataFrame:
    oos_features = feature_panel.loc[pd.to_datetime(feature_panel["date"]).ge(oos_start), ["symbol", "date", *features]].copy()
    prediction_frame = build_family_prediction_frame(
        oos_features,
        features,
        min_feature_coverage=min_feature_coverage,
    )
    proba = classifier.predict_proba_frame(prediction_frame, features)
    scores = build_strategy_score_frame(
        source=source,
        family=family,
        prediction_frame=prediction_frame,
        probability_frame=proba,
        ae_familiarity_frame=None,
        apply_ae_to_exits=False,
    )
    scores["strategy_source"] = f"{source}.{family}.{representation}"
    return scores


def _long_only_returns(
    strategy_scores: pd.DataFrame,
    next_returns: pd.DataFrame,
    *,
    top_k_values: tuple[int, ...],
    cost_bps: float,
    capital_base: float,
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
        rows.append(
            shared_book_performance_metrics(
                returns,
                equity,
                weights,
                trades,
                framework="vectorized_shared_book",
                variant="long_only",
                top_k=top_k,
                cost_bps=cost_bps,
            )
        )
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


def _safe_filename(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value)).strip("._") or "family"


if __name__ == "__main__":
    main()
