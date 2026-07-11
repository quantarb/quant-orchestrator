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

from quant_orchestrator.platforms.ml_frameworks.rapids import RapidsRandomForestClassifier
from quant_orchestrator.research_tools.cross_sectional_return_autoencoder import (
    MatrixAutoencoderConfig,
    build_multi_horizon_return_tensor,
    build_symbol_reconstruction_feature_frame,
    reconstruct_multi_horizon_returns,
    train_matrix_autoencoder,
)
from quant_orchestrator.research_tools.ml_trading import (
    classification_probability_diagnostics,
    prepare_family_dataset,
)
from quant_orchestrator.research_tools.ml_trading_experiment import (
    MLTradingExperimentConfig,
    _build_oracle_trade_label_rows_sparse,
    _prepare_quant_warehouse_import,
    _warehouse_imports,
)


SOURCE = "cross_sectional_autoencoder"
FAMILY = "multi_horizon_return_reconstruction"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", default="artifacts/cross_sectional_autoencoder_classifier_1t")
    parser.add_argument("--min-market-cap", type=int, default=1_000_000_000_000)
    parser.add_argument("--provider", default="fmp")
    parser.add_argument("--start-date", default="1900-01-01")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--train-end", default="2020-12-31")
    parser.add_argument("--oos-start", default="2021-01-01")
    parser.add_argument("--oracle-frequency", default="YE")
    parser.add_argument("--oracle-k-min", type=int, default=1)
    parser.add_argument("--oracle-k-max", type=int, default=12)
    parser.add_argument("--min-feature-coverage", type=float, default=0.50)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--random-seed", type=int, default=20260702)
    args = parser.parse_args()

    started = perf_counter()
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(int(args.random_seed))
    _prepare_quant_warehouse_import("/home/jlee153232/PycharmProjects/quant-warehouse")
    Warehouse, _EventPairStore, BinaryTargetConfig, FamilyEvaluationConfig, *_rest = _warehouse_imports()
    screen_fmp_equity_universe = _rest[-1]
    warehouse = Warehouse()

    end_date = str(args.end_date).strip() or None
    feature_config = FamilyEvaluationConfig(
        provider=args.provider,
        market_cap_min=int(args.min_market_cap),
        start_date=args.start_date,
        end_date=end_date,
        max_features_per_family=1,
    )
    symbols, raw_universe, universe_eligibility, universe_source = _screen_price_universe(
        warehouse,
        feature_config,
        screen_fmp_equity_universe=screen_fmp_equity_universe,
    )
    symbols = tuple(str(symbol).strip().upper() for symbol in symbols if str(symbol).strip())

    price_frames = _load_price_frames(
        warehouse,
        symbols,
        provider=args.provider,
        start=args.start_date,
        end=end_date,
    )
    wide_close = pd.DataFrame({symbol: frame["close"] for symbol, frame in price_frames.items()}).sort_index()
    wide_close = wide_close.ffill()
    wide_close = wide_close.loc[:, wide_close.notna().any(axis=0)]
    if wide_close.empty:
        raise ValueError("No price data loaded for cross-sectional autoencoder classifier")

    tensor = build_multi_horizon_return_tensor(wide_close, min_finite_horizons=1)
    ae_fit = train_matrix_autoencoder(
        tensor,
        train_end=pd.Timestamp(args.train_end),
        config=MatrixAutoencoderConfig(
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            random_seed=int(args.random_seed),
        ),
    )
    reconstructed = reconstruct_multi_horizon_returns(ae_fit, tensor.values)
    feature_panel = build_symbol_reconstruction_feature_frame(tensor, reconstructed)
    feature_panel["symbol"] = feature_panel["symbol"].astype(str).str.upper()
    feature_panel["date"] = pd.to_datetime(feature_panel["date"], errors="coerce").dt.normalize()

    feature_cols = [col for col in feature_panel.columns if col not in {"symbol", "date"}]
    feature_metadata = pd.DataFrame(
        {
            "source": SOURCE,
            "family": FAMILY,
            "feature": feature_cols,
        }
    )

    target_config = BinaryTargetConfig(
        provider=args.provider,
        start_date=args.start_date,
        end_date=end_date,
        oracle_trade_k_by_frequency={
            str(args.oracle_frequency).upper(): tuple(range(int(args.oracle_k_min), int(args.oracle_k_max) + 1))
        },
    )
    label_rows, label_diagnostics, oracle_seconds, _oracle_unique_windows = (
        _build_oracle_trade_label_rows_sparse(
            symbols,
            target_config,
            warehouse=warehouse,
        )
    )
    dataset, classifier_features = prepare_family_dataset(
        feature_panel,
        feature_metadata,
        label_rows,
        source=SOURCE,
        family=FAMILY,
        min_feature_coverage=float(args.min_feature_coverage),
    )
    if dataset.empty:
        raise ValueError("CS-AE feature family produced no labeled classifier rows")

    train_end = pd.Timestamp(args.train_end)
    oos_start = pd.Timestamp(args.oos_start)
    train = dataset.loc[pd.to_datetime(dataset["date"]).le(train_end)].copy()
    oos = dataset.loc[pd.to_datetime(dataset["date"]).ge(oos_start)].copy()
    if len(train) < 250 or train["collapsed_label"].nunique() < 2:
        raise ValueError(
            "Not enough labeled train rows/classes for CS-AE classifier: "
            f"rows={len(train)}, classes={train['collapsed_label'].nunique()}"
        )

    classifier = RapidsRandomForestClassifier.fit(
        train,
        features=classifier_features,
        target_col="collapsed_label",
        random_state=int(args.random_seed),
        params=MLTradingExperimentConfig().rf_params,
    )
    train_proba = classifier.predict_proba_frame(train, classifier_features)
    train_scores = classification_probability_diagnostics(
        train,
        train_proba,
        target_col="collapsed_label",
        labels=classifier.encoder.classes_,
    )
    oos_proba = classifier.predict_proba_frame(oos, classifier_features) if not oos.empty else pd.DataFrame()
    oos_scores = (
        classification_probability_diagnostics(
            oos,
            oos_proba,
            target_col="collapsed_label",
            labels=classifier.encoder.classes_,
        )
        if not oos.empty and oos["collapsed_label"].nunique() > 1
        else _empty_scores(len(oos))
    )

    importances = _feature_importances(classifier, classifier_features)
    model_results = pd.DataFrame(
        [
            {
                "source": SOURCE,
                "family": FAMILY,
                "strategy_source": f"{SOURCE}.{FAMILY}",
                "status": "ok",
                "features": len(classifier_features),
                "symbols": len(symbols),
                "tensor_dates": len(tensor.dates),
                "rows": len(dataset),
                "train_rows": len(train),
                "oos_rows": len(oos),
                "classes": dataset["collapsed_label"].nunique(),
                "ae_fit_seconds": ae_fit.fit_seconds,
                "ae_device": ae_fit.device,
                "ae_hidden_dim": ae_fit.hidden_dim,
                "ae_latent_dim": ae_fit.latent_dim,
                "oracle_seconds": oracle_seconds,
                **{f"train_{key}": value for key, value in train_scores.items()},
                **{f"oos_{key}": value for key, value in oos_scores.items()},
                "elapsed_seconds": perf_counter() - started,
            }
        ]
    )

    feature_panel.to_parquet(artifact_dir / "feature_panel.parquet", index=False)
    dataset.to_parquet(artifact_dir / "classifier_dataset.parquet", index=False)
    label_rows.to_csv(artifact_dir / "label_rows.csv", index=False)
    label_diagnostics.to_csv(artifact_dir / "label_diagnostics.csv", index=False)
    model_results.to_csv(artifact_dir / "model_results.csv", index=False)
    importances.to_csv(artifact_dir / "feature_importance.csv", index=False)
    importances.head(20).to_csv(artifact_dir / "feature_importance_top20.csv", index=False)
    raw_universe.to_csv(artifact_dir / "raw_universe.csv", index=False)
    universe_eligibility.to_csv(artifact_dir / "universe_eligibility.csv", index=False)
    with (artifact_dir / "classifier.pkl").open("wb") as handle:
        pickle.dump(classifier, handle, protocol=pickle.HIGHEST_PROTOCOL)
    (artifact_dir / "metadata.json").write_text(
        json.dumps(
            {
                "source": SOURCE,
                "family": FAMILY,
                "universe_source": universe_source,
                "symbols": symbols,
                "horizons": tensor.horizons,
                "ae_losses_tail": ae_fit.losses[-10:],
                "artifact_dir": str(artifact_dir),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print("artifact_dir", artifact_dir)
    print(model_results.to_string(index=False))
    print(importances.head(20).to_string(index=False))


def _load_price_frames(warehouse, symbols, *, provider: str, start: str, end: str | None) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for raw_symbol in symbols:
        symbol = str(raw_symbol).strip().upper()
        if not symbol:
            continue
        prices = warehouse.read_prices(symbol, provider=provider, start=start, end=end)
        if prices is None or prices.empty or "close" not in prices.columns:
            continue
        frame = prices.copy()
        frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index, errors="coerce")).normalize()
        frame = frame.loc[frame.index.notna()].sort_index()
        frame = frame.loc[~frame.index.duplicated(keep="last")]
        frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
        frames[symbol] = frame
    return frames


def _screen_price_universe(
    warehouse,
    config,
    *,
    screen_fmp_equity_universe,
) -> tuple[tuple[str, ...], pd.DataFrame, pd.DataFrame, str]:
    try:
        symbols, raw_universe, eligibility, source = screen_fmp_equity_universe(
            config,
            warehouse=warehouse,
            required_sections=("prices",),
        )
        bad_asset_rejections = (
            eligibility.get("reason", pd.Series(dtype=object)).astype(str).str.startswith("asset_class:").mean()
            if not eligibility.empty
            else 0.0
        )
        if len(symbols) >= 2 and bad_asset_rejections < 0.50:
            return symbols, raw_universe, eligibility, source
    except Exception:
        pass

    profiles = warehouse.catalog.query_symbol_profiles(
        provider=config.provider,
        min_market_cap=config.market_cap_min,
        country=config.country,
        exchanges=config.exchanges,
        exclude_etf=True,
        exclude_fund=True,
        limit=config.screen_limit,
    )
    rows = []
    eligibility_rows = []
    symbols = []
    for profile in profiles:
        symbol = str(profile.symbol).strip().upper()
        rows.append(
            {
                "symbol": symbol,
                "name": profile.company_name,
                "market_cap": profile.market_cap,
                "exchange": profile.exchange,
                "country": profile.country,
                "sector": profile.sector,
                "industry": profile.industry,
            }
        )
        prices = warehouse.read_prices(symbol, provider=config.provider)
        ok = prices is not None and not prices.empty and "close" in prices.columns
        if ok:
            symbols.append(symbol)
        eligibility_rows.append(
            {
                "symbol": symbol,
                "eligible": bool(ok),
                "reason": "ok" if ok else "prices: empty",
                "screen_market_cap": profile.market_cap,
            }
        )
    raw_universe = pd.DataFrame(rows).drop_duplicates("symbol")
    eligibility = pd.DataFrame(eligibility_rows).drop_duplicates("symbol")
    return tuple(sorted(dict.fromkeys(symbols))), raw_universe, eligibility, f"catalog:{config.provider}:price_history"


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
