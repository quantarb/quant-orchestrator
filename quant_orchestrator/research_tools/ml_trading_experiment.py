from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import pickle
import re
import sys
import tempfile
from time import perf_counter
from typing import Literal

import numpy as np
import pandas as pd

from quant_orchestrator.platforms.backtesting_frameworks.shared_book import build_shared_book_weights
from quant_orchestrator.platforms.backtesting_frameworks.zipline.shared_book import (
    ZiplineSharedBookSummaryJob,
    run_zipline_shared_book_summary_jobs,
)
from quant_orchestrator.platforms.ml_frameworks.rapids import RapidsRandomForestClassifier
from quant_orchestrator.platforms.ml_frameworks.torch_autoencoder import (
    LatentAutoencoderConfig,
    LatentAutoencoderIndex,
)
from quant_orchestrator.research_tools.ml_trading import (
    build_family_prediction_frame,
    build_strategy_score_frame,
    prepare_family_dataset,
    write_ml_trading_artifact_files,
)
from quant_orchestrator.tracking import get_tracker


ExperimentMode = Literal["classifier", "classifier_ae"]


@dataclass(frozen=True)
class MLTradingExperimentConfig:
    experiment_name: str = "gpu_rf_shared_book_1t"
    mode: ExperimentMode = "classifier"
    min_market_cap: int = 1_000_000_000_000
    start_date: str = "1900-01-01"
    end_date: str | None = None
    train_end: str = "2019-12-31"
    oos_start: str = "2020-01-01"
    top_k_values: tuple[int, ...] = (5, 10, 20, 40)
    entry_threshold: float = 0.50
    exit_threshold: float = 0.50
    min_feature_coverage: float = 0.50
    max_features_per_family: int = 50
    min_train_rows_per_family: int = 250
    min_classes_per_family: int = 2
    capital_base: float = 1_000_000.0
    zipline_commission_per_share: float = 0.005
    zipline_slippage_bps: float = 5.0
    zipline_max_workers: int = 4
    random_seed: int = 20260702
    provider: str = "fmp"
    event_families: tuple[str, ...] = (
        "congress",
        "insider",
        "analyst_rating",
        "price_target",
        "earnings",
    )
    oracle_frequencies: tuple[str, ...] = ("YE",)
    oracle_k_min: int = 1
    oracle_k_max: int = 12
    rf_params: dict = field(
        default_factory=lambda: {
            "n_estimators": 300,
            "max_depth": 16,
            "max_features": "sqrt",
            "n_bins": 128,
            "n_streams": 8,
        }
    )
    ae_config: LatentAutoencoderConfig = field(default_factory=LatentAutoencoderConfig)
    quant_warehouse_root: str | None = "/home/jlee153232/PycharmProjects/quant-warehouse"
    log_mlflow: bool = True
    mlflow_experiment: str = "ml_trading"
    mlflow_tracking_uri: str | None = None


@dataclass(frozen=True)
class MLTradingExperimentResult:
    config: MLTradingExperimentConfig
    mlflow_run_id: str | None
    model_results: pd.DataFrame
    strategy_scores: pd.DataFrame
    backtest_summary: pd.DataFrame
    trade_log: pd.DataFrame
    analysis_markdown: str
    elapsed_seconds: float

    @property
    def metrics(self) -> dict[str, float | int | None]:
        best = _best_backtest_row(self.backtest_summary)
        return {
            "symbols": _safe_int(self.backtest_summary["score_symbols"].max())
            if "score_symbols" in self.backtest_summary
            else None,
            "trained_models": int((self.model_results["status"] == "ok").sum())
            if "status" in self.model_results
            else int(len(self.model_results)),
            "strategy_sources": int(self.strategy_scores["strategy_source"].nunique())
            if "strategy_source" in self.strategy_scores
            else 0,
            "best_sharpe": None if best is None else float(best["sharpe"]),
            "best_total_return": None if best is None else float(best["total_return"]),
            "best_max_drawdown": None if best is None else float(best["max_drawdown"]),
            "elapsed_seconds": float(self.elapsed_seconds),
        }


def run_ml_trading_experiment(
    config: MLTradingExperimentConfig,
) -> MLTradingExperimentResult:
    started = perf_counter()
    _prepare_quant_warehouse_import(config.quant_warehouse_root)
    np.random.seed(config.random_seed)

    (
        Warehouse,
        EventPairStore,
        BinaryTargetConfig,
        FamilyEvaluationConfig,
        build_collapsed_bullish_event_target_panel,
        build_fundamental_feature_panel,
        build_oracle_trade_target_panel,
        cap_features_by_quality,
        load_fmp_event_pairs,
        screen_fmp_equity_universe,
    ) = _warehouse_imports()

    train_end = pd.Timestamp(config.train_end)
    oos_start = pd.Timestamp(config.oos_start)
    warehouse = Warehouse()
    event_store = EventPairStore(backend=warehouse.backend, catalog=warehouse.catalog)
    feature_config = FamilyEvaluationConfig(
        provider=config.provider,
        market_cap_min=config.min_market_cap,
        start_date=config.start_date,
        end_date=config.end_date,
        max_features_per_family=config.max_features_per_family,
    )
    target_config = BinaryTargetConfig(
        provider=config.provider,
        start_date=config.start_date,
        end_date=config.end_date,
        event_families=config.event_families,
        oracle_trade_k_by_frequency={
            frequency: tuple(range(config.oracle_k_min, config.oracle_k_max + 1))
            for frequency in config.oracle_frequencies
        },
    )

    symbols, _raw_universe, universe_eligibility, universe_source = screen_fmp_equity_universe(
        feature_config,
        warehouse=warehouse,
    )
    raw_feature_panel, raw_feature_metadata, _feature_diagnostics, feature_timings = (
        build_fundamental_feature_panel(symbols, feature_config, warehouse=warehouse)
    )
    selected_features, selected_feature_metadata, _feature_quality = cap_features_by_quality(
        raw_feature_panel,
        raw_feature_metadata,
        max_features=config.max_features_per_family,
    )
    feature_panel = raw_feature_panel[["symbol", "date", *selected_features]].copy()
    feature_panel["symbol"] = feature_panel["symbol"].astype(str).str.upper()
    feature_panel["date"] = pd.to_datetime(feature_panel["date"], errors="coerce").dt.normalize()

    events, event_diagnostics, event_load_seconds = load_fmp_event_pairs(
        symbols,
        target_config,
        event_store=event_store,
        include_historical=True,
    )
    event_symbols = tuple(
        event_diagnostics.loc[event_diagnostics["combined_rows"].gt(0), "symbol"].sort_values()
    )
    feature_panel = feature_panel.loc[feature_panel["symbol"].isin(event_symbols)].copy()
    collapsed_event_panel, _collapsed_event_metadata = build_collapsed_bullish_event_target_panel(
        feature_panel[["symbol", "date"]],
        events,
        target_config,
    )
    oracle_panel, oracle_metadata, oracle_seconds = build_oracle_trade_target_panel(
        event_symbols,
        target_config,
        warehouse=warehouse,
    )
    label_rows, label_diagnostics = collapsed_label_rows(collapsed_event_panel, oracle_panel)
    model_results, models = _train_family_models(
        config,
        feature_panel,
        selected_feature_metadata,
        label_rows,
        train_end=train_end,
        oos_start=oos_start,
    )
    strategy_scores, mean_scores = _score_family_models(
        config,
        feature_panel,
        models,
        oos_start=oos_start,
    )
    price_frames = _load_price_frames(
        warehouse,
        event_symbols,
        provider=config.provider,
        start=config.start_date,
        end=config.end_date,
    )
    backtest_summary, trade_log, trade_generation_audit = _run_shared_book_backtests(
        config,
        strategy_scores,
        price_frames,
        oos_start=oos_start,
    )
    model_oos_summary = _model_oos_summary(model_results)
    analysis_markdown = _build_analysis(
        config,
        universe_source=universe_source,
        symbols=symbols,
        event_symbols=event_symbols,
        model_results=model_results,
        model_oos_summary=model_oos_summary,
        strategy_scores=strategy_scores,
        mean_scores=mean_scores,
        backtest_summary=backtest_summary,
        trade_generation_audit=trade_generation_audit,
        feature_timings=feature_timings,
        event_load_seconds=event_load_seconds,
        oracle_seconds=oracle_seconds,
        oracle_metadata_rows=len(oracle_metadata),
        label_diagnostics=label_diagnostics,
        eligible_symbols=int(universe_eligibility["eligible"].sum())
        if "eligible" in universe_eligibility
        else len(symbols),
    )
    elapsed_seconds = perf_counter() - started
    metrics = _experiment_metrics(
        model_results=model_results,
        strategy_scores=strategy_scores,
        backtest_summary=backtest_summary,
        symbols=symbols,
        event_symbols=event_symbols,
        elapsed_seconds=elapsed_seconds,
    )
    result = MLTradingExperimentResult(
        config=config,
        mlflow_run_id=None,
        model_results=model_results,
        strategy_scores=strategy_scores,
        backtest_summary=backtest_summary,
        trade_log=trade_log,
        analysis_markdown=analysis_markdown,
        elapsed_seconds=elapsed_seconds,
    )
    if config.log_mlflow:
        result = MLTradingExperimentResult(
            config=result.config,
            mlflow_run_id=log_ml_trading_result_to_mlflow(result, trained_models=models),
            model_results=result.model_results,
            strategy_scores=result.strategy_scores,
            backtest_summary=result.backtest_summary,
            trade_log=result.trade_log,
            analysis_markdown=result.analysis_markdown,
            elapsed_seconds=result.elapsed_seconds,
        )
    return result


def log_ml_trading_result_to_mlflow(result: MLTradingExperimentResult, *, trained_models: dict | None = None) -> str:
    tracker = get_tracker(tracking_uri=result.config.mlflow_tracking_uri)
    tags = {
        "quant_orchestrator.kind": "ml_trading",
        "quant_orchestrator.mode": result.config.mode,
        "quant_orchestrator.provider": result.config.provider,
        "quant_orchestrator.experiment_name": result.config.experiment_name,
    }
    with tracker.start_run(
        name=result.config.experiment_name,
        experiment=result.config.mlflow_experiment,
        tags=tags,
    ) as run:
        tracker.log_params(_config_params(result.config))
        tracker.log_metrics(_finite_metrics(result.metrics))
        with tempfile.TemporaryDirectory(prefix="quant-orchestrator-ml-trading-") as tmp_dir:
            paths = write_ml_trading_artifact_files(
                model_results=result.model_results,
                strategy_scores=result.strategy_scores,
                backtest_summary=result.backtest_summary,
                trade_log=result.trade_log,
                analysis_markdown=result.analysis_markdown,
                directory=tmp_dir,
            )
            for path in paths.values():
                tracker.log_artifact(str(path), artifact_path="ml_trading")
            if trained_models:
                model_root = Path(tmp_dir) / "models"
                _write_trained_model_artifacts(model_root, trained_models)
                tracker.log_artifact(str(model_root), artifact_path="ml_trading_models")
        return str(run.info.run_id)


def _write_trained_model_artifacts(directory: Path, trained_models: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    manifest = []
    for (source, family), payload in trained_models.items():
        strategy_source = f"{source}.{family}"
        model_dir = directory / _safe_filename(strategy_source)
        model_dir.mkdir(parents=True, exist_ok=True)
        features = list(payload.get("features", []))
        metadata = {
            "strategy_source": strategy_source,
            "source": str(source),
            "family": str(family),
            "features": features,
            "feature_count": len(features),
            "has_classifier": payload.get("classifier") is not None,
            "has_autoencoder": payload.get("autoencoder") is not None,
            "classifier_file": "classifier.pkl",
            "autoencoder_dir": "autoencoder" if payload.get("autoencoder") is not None else None,
        }
        classifier = payload.get("classifier")
        if classifier is not None:
            try:
                with (model_dir / "classifier.pkl").open("wb") as handle:
                    pickle.dump(classifier, handle, protocol=pickle.HIGHEST_PROTOCOL)
                metadata["classifier_serialized"] = True
            except Exception as exc:  # pragma: no cover - backend-specific serialization failures
                metadata["classifier_serialized"] = False
                metadata["classifier_serialization_error"] = repr(exc)
                (model_dir / "classifier_serialization_error.txt").write_text(repr(exc), encoding="utf-8")
        autoencoder = payload.get("autoencoder")
        if autoencoder is not None:
            _write_autoencoder_artifacts(model_dir / "autoencoder", autoencoder)
            metadata["autoencoder_metadata"] = autoencoder.metadata()
        (model_dir / "metadata.json").write_text(_json_dumps(metadata), encoding="utf-8")
        manifest.append(metadata)
    (directory / "manifest.json").write_text(_json_dumps({"models": manifest}), encoding="utf-8")


def _write_autoencoder_artifacts(directory: Path, autoencoder) -> None:
    import torch

    directory.mkdir(parents=True, exist_ok=True)
    state_dict = {
        key: value.detach().cpu()
        for key, value in autoencoder.model.state_dict().items()
    }
    torch.save(
        {
            "state_dict": state_dict,
            "in_dim": len(autoencoder.features),
            "hidden_dim": autoencoder.hidden_dim,
            "bottleneck_dim": autoencoder.bottleneck_dim,
            "model_factory": "quant_orchestrator.platforms.ml_frameworks.torch_autoencoder.latent_index.create_family_autoencoder",
        },
        directory / "model_state.pt",
    )
    np.savez_compressed(
        directory / "preprocessing.npz",
        center=autoencoder.center,
        scale=autoencoder.scale,
        lower=autoencoder.lower,
        upper=autoencoder.upper,
    )
    with (directory / "nearest_neighbors.pkl").open("wb") as handle:
        pickle.dump(autoencoder.nn_index, handle, protocol=pickle.HIGHEST_PROTOCOL)
    metadata = {
        "features": list(autoencoder.features),
        "metadata": autoencoder.metadata(),
        "architecture_diagnostics": list(getattr(autoencoder, "architecture_diagnostics", [])),
        "config": asdict(autoencoder.config),
        "latent_distance_cutoff": autoencoder.latent_distance_cutoff,
    }
    (directory / "metadata.json").write_text(_json_dumps(metadata), encoding="utf-8")


def collapsed_label_rows(
    collapsed_event_panel: pd.DataFrame,
    oracle_panel: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    event_target = "target_event_collapsed__bullish"
    event_activity = f"_target_activity__{event_target}"
    frames = []
    diagnostics = []

    event_frame = collapsed_event_panel.copy()
    if event_target in event_frame.columns:
        activity = pd.to_numeric(event_frame.get(event_activity, 0), errors="coerce").fillna(0)
        value = pd.to_numeric(event_frame[event_target], errors="coerce").fillna(0)
        bullish = event_frame.loc[value.eq(1), ["symbol", "date"]].copy()
        bullish["collapsed_label"] = "event_bullish"
        bullish["label_source"] = "event_collapsed"
        frames.append(bullish)
        diagnostics.append(
            {
                "source": "event_collapsed",
                "candidate_rows": int(activity.gt(0).sum()),
                "used_rows": len(bullish),
                "dropped_rows": int(activity.gt(0).sum()) - len(bullish),
                "note": "mirror/non-bullish event rows excluded",
            }
        )

    long_cols = sorted(
        c for c in oracle_panel.columns if c.startswith("target_oracle_trade_entry__") and c.endswith("_long")
    )
    short_cols = sorted(
        c for c in oracle_panel.columns if c.startswith("target_oracle_trade_entry__") and c.endswith("_short")
    )
    if long_cols and short_cols:
        oracle = oracle_panel[["symbol", "date", *long_cols, *short_cols]].copy()
        long_any = oracle[long_cols].apply(pd.to_numeric, errors="coerce").fillna(0).gt(0).any(axis=1)
        short_any = oracle[short_cols].apply(pd.to_numeric, errors="coerce").fillna(0).gt(0).any(axis=1)
        ambiguous = long_any & short_any
        long_rows = oracle.loc[long_any & ~short_any, ["symbol", "date"]].copy()
        long_rows["collapsed_label"] = "oracle_long"
        long_rows["label_source"] = "oracle_trade"
        short_rows = oracle.loc[short_any & ~long_any, ["symbol", "date"]].copy()
        short_rows["collapsed_label"] = "oracle_short"
        short_rows["label_source"] = "oracle_trade"
        frames.extend([long_rows, short_rows])
        diagnostics.append(
            {
                "source": "oracle_trade",
                "candidate_rows": int((long_any | short_any).sum()),
                "used_rows": len(long_rows) + len(short_rows),
                "dropped_rows": int(ambiguous.sum()),
                "note": "ambiguous long+short rows dropped after k collapse",
            }
        )

    labels = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=["symbol", "date", "collapsed_label", "label_source"])
    )
    labels["symbol"] = labels["symbol"].astype(str).str.upper()
    labels["date"] = pd.to_datetime(labels["date"], errors="coerce").dt.normalize()
    labels = labels.dropna(subset=["symbol", "date", "collapsed_label"]).drop_duplicates()
    return labels.sort_values(["date", "symbol", "collapsed_label"]).reset_index(drop=True), pd.DataFrame(diagnostics)


def _train_family_models(
    config: MLTradingExperimentConfig,
    feature_panel: pd.DataFrame,
    selected_feature_metadata: pd.DataFrame,
    label_rows: pd.DataFrame,
    *,
    train_end: pd.Timestamp,
    oos_start: pd.Timestamp,
) -> tuple[pd.DataFrame, dict]:
    model_rows = []
    models = {}
    for source, family in (
        selected_feature_metadata[["source", "family"]]
        .drop_duplicates()
        .sort_values(["source", "family"])
        .itertuples(index=False, name=None)
    ):
        family_frame, features = prepare_family_dataset(
            feature_panel,
            selected_feature_metadata,
            label_rows,
            source=str(source),
            family=str(family),
            min_feature_coverage=config.min_feature_coverage,
        )
        if family_frame.empty:
            model_rows.append(
                {"source": source, "family": family, "status": "skipped_empty", "features": len(features), "rows": 0}
            )
            continue
        train = family_frame.loc[pd.to_datetime(family_frame["date"]).le(train_end)].copy()
        oos = family_frame.loc[pd.to_datetime(family_frame["date"]).ge(oos_start)].copy()
        if len(train) < config.min_train_rows_per_family or train["collapsed_label"].nunique() < config.min_classes_per_family:
            model_rows.append(
                {
                    "source": source,
                    "family": family,
                    "status": "skipped_sparse_train",
                    "features": len(features),
                    "rows": len(family_frame),
                    "train_rows": len(train),
                    "oos_rows": len(oos),
                    "train_classes": train["collapsed_label"].nunique(),
                }
            )
            continue
        fit_started = perf_counter()
        classifier = RapidsRandomForestClassifier.fit(
            train,
            features=features,
            target_col="collapsed_label",
            random_state=config.random_seed,
            params=config.rf_params,
        )
        classifier_fit_seconds = perf_counter() - fit_started
        autoencoder = None
        ae_fit_seconds = np.nan
        ae_metadata = {}
        if config.mode == "classifier_ae":
            ae_started = perf_counter()
            autoencoder = LatentAutoencoderIndex.fit(train, features=features, config=config.ae_config)
            ae_fit_seconds = perf_counter() - ae_started
            ae_metadata = {} if autoencoder is None else autoencoder.metadata()
        models[(source, family)] = {
            "classifier": classifier,
            "autoencoder": autoencoder,
            "features": features,
        }
        train_scores = classifier.score(train, features=features, target_col="collapsed_label")
        oos_scores = (
            classifier.score(oos, features=features, target_col="collapsed_label")
            if not oos.empty and oos["collapsed_label"].nunique() > 1
            else {"rows": len(oos), "accuracy": np.nan, "balanced_accuracy": np.nan, "macro_f1": np.nan}
        )
        model_rows.append(
            {
                "source": source,
                "family": family,
                "status": "ok",
                "features": len(features),
                "rows": len(family_frame),
                "train_rows": len(train),
                "oos_rows": len(oos),
                "classes": family_frame["collapsed_label"].nunique(),
                "classifier_fit_seconds": classifier_fit_seconds,
                "ae_fit_seconds": ae_fit_seconds,
                **ae_metadata,
                **{f"train_{k}": v for k, v in train_scores.items()},
                **{f"oos_{k}": v for k, v in oos_scores.items()},
            }
        )
    return (
        pd.DataFrame(model_rows)
        .sort_values(["status", "oos_macro_f1", "oos_balanced_accuracy"], ascending=[True, False, False])
        .reset_index(drop=True),
        models,
    )


def _score_family_models(
    config: MLTradingExperimentConfig,
    feature_panel: pd.DataFrame,
    models: dict,
    *,
    oos_start: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred_frames = []
    for (source, family), payload in models.items():
        pred_input = build_family_prediction_frame(
            feature_panel,
            payload["features"],
            min_feature_coverage=config.min_feature_coverage,
        )
        if pred_input.empty:
            continue
        proba = payload["classifier"].predict_proba_frame(pred_input, payload["features"])
        ae_frame = (
            payload["autoencoder"].familiarity(pred_input)
            if config.mode == "classifier_ae" and payload["autoencoder"] is not None
            else None
        )
        pred_frames.append(
            build_strategy_score_frame(
                source=str(source),
                family=str(family),
                prediction_frame=pred_input,
                probability_frame=proba,
                ae_familiarity_frame=ae_frame,
                apply_ae_to_exits=True,
            )
        )
    single_model_scores = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()
    if single_model_scores.empty:
        return single_model_scores, pd.DataFrame()
    single_model_scores = single_model_scores.loc[pd.to_datetime(single_model_scores["date"]).ge(oos_start)].copy()
    mean_scores = (
        single_model_scores.groupby(["symbol", "date"], as_index=False)
        .agg(
            long_score=("long_score", "mean"),
            short_score=("short_score", "mean"),
            long_exit_score=("long_exit_score", "mean"),
            short_exit_score=("short_exit_score", "mean"),
            classifier_long_score=("classifier_long_score", "mean"),
            classifier_short_score=("classifier_short_score", "mean"),
            ae_familiarity=("ae_familiarity", "mean"),
            ae_recon_error=("ae_recon_error", "mean"),
            ae_latent_distance=("ae_latent_distance", "mean"),
            model_count=("strategy_source", "nunique"),
        )
    )
    mean_scores["source"] = "ensemble"
    mean_scores["family"] = "mean"
    mean_scores["strategy_source"] = "ensemble_mean"
    mean_scores["net_score"] = mean_scores["long_score"] - mean_scores["short_score"]
    score_cols = [
        "strategy_source",
        "source",
        "family",
        "symbol",
        "date",
        "long_score",
        "short_score",
        "long_exit_score",
        "short_exit_score",
        "classifier_long_score",
        "classifier_short_score",
        "ae_familiarity",
        "ae_recon_error",
        "ae_latent_distance",
        "net_score",
        "model_count",
    ]
    return pd.concat([mean_scores[score_cols], single_model_scores[score_cols]], ignore_index=True), mean_scores


def _run_shared_book_backtests(
    config: MLTradingExperimentConfig,
    strategy_scores: pd.DataFrame,
    price_frames: dict[str, pd.DataFrame],
    *,
    oos_start: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if strategy_scores.empty or not price_frames:
        return pd.DataFrame(), pd.DataFrame(), {}
    wide_close = pd.DataFrame({symbol: frame["close"] for symbol, frame in price_frames.items()}).sort_index().ffill()
    next_returns = wide_close.pct_change().shift(-1)
    oos_dates = pd.DatetimeIndex(sorted(set(strategy_scores["date"]).intersection(next_returns.index)))
    oos_dates = oos_dates[oos_dates >= oos_start]
    trade_symbols = tuple(sorted(set(strategy_scores["symbol"]).intersection(next_returns.columns)))
    if len(oos_dates) == 0 or not trade_symbols:
        return pd.DataFrame(), pd.DataFrame(), {}
    price_window_start = pd.Timestamp(oos_dates.min()) - pd.Timedelta(days=20)
    price_window_end = pd.Timestamp(oos_dates.max())
    price_frame_subset = {
        symbol: frame.loc[
            (pd.DatetimeIndex(frame.index) >= price_window_start)
            & (pd.DatetimeIndex(frame.index) <= price_window_end)
        ].copy()
        for symbol, frame in price_frames.items()
        if symbol in trade_symbols
    }
    trade_logs = []
    zipline_jobs = []
    strategy_source_order = ["ensemble_mean"] + sorted(
        source for source in strategy_scores["strategy_source"].dropna().unique().tolist() if source != "ensemble_mean"
    )
    for strategy_source in strategy_source_order:
        score_frame = strategy_scores.loc[strategy_scores["strategy_source"].eq(strategy_source)].copy()
        if score_frame.empty:
            continue
        source_value = score_frame["source"].iloc[0]
        family_value = score_frame["family"].iloc[0]
        for variant in ["long_only", "short_only", "long_short"]:
            for top_k in config.top_k_values:
                weights, trades = build_shared_book_weights(
                    score_frame,
                    trade_symbols,
                    oos_dates,
                    top_k=top_k,
                    variant=variant,
                    entry_threshold=config.entry_threshold,
                    exit_threshold=config.exit_threshold,
                    long_exit_score_col="long_exit_score",
                    short_exit_score_col="short_exit_score",
                )
                trades = trades.assign(
                    strategy_source=strategy_source,
                    source=source_value,
                    family=family_value,
                    variant=variant,
                    top_k=top_k,
                )
                trade_logs.append(trades)
                metadata = {
                    "strategy_source": strategy_source,
                    "source": source_value,
                    "family": family_value,
                    "variant": variant,
                    "top_k": top_k,
                    "score_rows": len(score_frame),
                    "score_symbols": score_frame["symbol"].nunique(),
                    "score_dates": score_frame["date"].nunique(),
                    "signal_events": len(trades),
                    "commission_per_share": config.zipline_commission_per_share,
                    "slippage_bps": config.zipline_slippage_bps,
                    "avg_gross_exposure": float(weights.abs().sum(axis=1).mean()),
                    "avg_net_exposure": float(weights.sum(axis=1).mean()),
                    "fully_invested_days": float(weights.abs().sum(axis=1).ge(0.999).mean()),
                    "cash_days": float(weights.abs().sum(axis=1).eq(0).mean()),
                }
                if "ae_familiarity" in score_frame:
                    metadata.update(
                        {
                            "ae_familiarity_mean": float(score_frame["ae_familiarity"].mean()),
                            "ae_familiarity_p25": float(score_frame["ae_familiarity"].quantile(0.25)),
                            "ae_familiarity_p75": float(score_frame["ae_familiarity"].quantile(0.75)),
                        }
                    )
                zipline_jobs.append(
                    ZiplineSharedBookSummaryJob(
                        price_frames=price_frame_subset,
                        target_weights=weights,
                        metadata=metadata,
                        capital_base=config.capital_base,
                        commission_per_share=config.zipline_commission_per_share,
                        slippage_bps=config.zipline_slippage_bps,
                    )
                )
    backtest_summary = run_zipline_shared_book_summary_jobs(
        zipline_jobs,
        max_workers=config.zipline_max_workers,
    )
    usable_trade_logs = [
        frame.dropna(axis=1, how="all")
        for frame in trade_logs
        if frame is not None and not frame.empty and not frame.dropna(axis=1, how="all").empty
    ]
    trade_log = pd.concat(usable_trade_logs, ignore_index=True) if usable_trade_logs else pd.DataFrame()
    audit = {
        "strategy_sources": len(strategy_source_order),
        "variants": 3,
        "top_k_values": len(config.top_k_values),
        "zipline_jobs": len(zipline_jobs),
        "score_rows_total": int(strategy_scores.groupby("strategy_source").size().sum()),
        "signal_events_total": int(sum(len(frame) for frame in trade_logs)),
        "max_trade_cap": None,
    }
    return backtest_summary.sort_values(["strategy_source", "variant", "top_k"]).reset_index(drop=True), trade_log, audit


def _load_price_frames(warehouse, symbols, *, provider: str, start: str, end: str | None) -> dict[str, pd.DataFrame]:
    frames = {}
    for symbol in symbols:
        prices = warehouse.read_prices(symbol, provider=provider, start=start, end=end)
        if prices is None or prices.empty:
            continue
        frame = prices.rename(columns=str.lower).copy()
        required = ["open", "high", "low", "close", "volume"]
        if not set(required).issubset(frame.columns):
            continue
        frame = frame[required].apply(pd.to_numeric, errors="coerce")
        frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index, errors="coerce")).normalize()
        frame = frame.dropna(subset=["open", "high", "low", "close"]).sort_index()
        if not frame.empty:
            frames[str(symbol).upper()] = frame
    return frames


def _model_oos_summary(model_results: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "source",
        "family",
        "oos_rows",
        "oos_accuracy",
        "oos_balanced_accuracy",
        "oos_macro_f1",
        "ae_latent_index_rows",
        "ae_train_latent_distance_mean",
        "ae_train_latent_distance_p95",
        "ae_train_error_mean",
        "ae_train_error_p95",
    ]
    out = model_results.loc[model_results["status"].eq("ok"), [col for col in cols if col in model_results.columns]].copy()
    if not out.empty:
        out["strategy_source"] = out.apply(lambda row: f"{row['source']}.{row['family']}", axis=1)
    return out


def _build_analysis(
    config: MLTradingExperimentConfig,
    *,
    universe_source: str,
    symbols,
    event_symbols,
    model_results: pd.DataFrame,
    model_oos_summary: pd.DataFrame,
    strategy_scores: pd.DataFrame,
    mean_scores: pd.DataFrame,
    backtest_summary: pd.DataFrame,
    trade_generation_audit: dict,
    feature_timings: dict,
    event_load_seconds: float,
    oracle_seconds: float,
    oracle_metadata_rows: int,
    label_diagnostics: pd.DataFrame,
    eligible_symbols: int,
) -> str:
    cap_text = _market_cap_text(config.min_market_cap)
    lines = [
        "## Written Analysis",
        "",
        f"- Experiment artifact name: `{config.experiment_name}`.",
        f"- Mode: `{config.mode}`.",
        f"- Universe: {len(symbols)} FMP {cap_text}+ symbols; {len(event_symbols)} had event coverage; universe_source={universe_source}; eligible_symbols={eligible_symbols}.",
        f"- Training window: all available rows through {config.train_end}. Out-of-sample model and trading window starts {config.oos_start}.",
        f"- Trained feature-family models: {int((model_results['status'] == 'ok').sum()) if 'status' in model_results else len(model_results)}.",
        f"- Strategy sources traded: {strategy_scores['strategy_source'].nunique() if not strategy_scores.empty else 0} total.",
        f"- Ensemble OOS prediction rows: {len(mean_scores):,} across {mean_scores['symbol'].nunique() if not mean_scores.empty else 0} symbols and {mean_scores['date'].nunique() if not mean_scores.empty else 0} dates.",
        f"- Event load seconds: {event_load_seconds:.3f}; oracle seconds: {oracle_seconds:.3f}; oracle metadata rows: {oracle_metadata_rows}.",
        f"- Feature timings: {feature_timings}.",
        f"- Strategy variants: long_only, short_only, long_short with top_k={list(config.top_k_values)}.",
        f"- Execution: native Zipline shared-book engine, ${config.zipline_commission_per_share:.3f}/share commission, {config.zipline_slippage_bps:.1f} bps slippage.",
    ]
    if config.mode == "classifier_ae":
        lines.extend(
            [
                f"- AE epochs={config.ae_config.epochs}; familiarity mode=latent nearest-neighbor reciprocal distance; metric={config.ae_config.nn_metric}; train-distance percentile={config.ae_config.familiarity_quantile}.",
                f"- Total indexed training latent rows: {int(model_results.get('ae_latent_index_rows', pd.Series(dtype=float)).fillna(0).sum()):,}.",
            ]
        )
    if trade_generation_audit:
        lines.append(
            f"- Trade generation audit: {trade_generation_audit.get('zipline_jobs', 0)} Zipline jobs, "
            f"{trade_generation_audit.get('score_rows_total', 0):,} score rows, "
            f"{trade_generation_audit.get('signal_events_total', 0):,} signal events, "
            f"max_trade_cap={trade_generation_audit.get('max_trade_cap')}."
        )
    if not label_diagnostics.empty:
        used = int(label_diagnostics["used_rows"].sum()) if "used_rows" in label_diagnostics else 0
        lines.append(f"- Collapsed label rows used before feature inner join: {used:,}.")
    if not model_oos_summary.empty:
        best_model = model_oos_summary.sort_values("oos_macro_f1", ascending=False).iloc[0]
        lines.append(
            f"- Best OOS classifier family: {best_model['strategy_source']} with macro_f1={best_model['oos_macro_f1']:.4f}."
        )
    best = _best_backtest_row(backtest_summary)
    if best is not None:
        lines.extend(
            [
                "",
                "Best native Zipline trading row by Sharpe:",
                f"- {best['strategy_source']} / {best['variant']} top_k={int(best['top_k'])}: "
                f"total_return={best['total_return']:.2%}, sharpe={best['sharpe']:.2f}, "
                f"max_drawdown={best['max_drawdown']:.2%}, avg_gross={best['avg_gross_exposure']:.2f}, "
                f"avg_net={best['avg_net_exposure']:.2f}.",
            ]
        )
        ensemble = (
            backtest_summary.loc[backtest_summary["strategy_source"].eq("ensemble_mean")]
            .sort_values(["sharpe", "total_return"], ascending=False)
            .head(1)
        )
        if not ensemble.empty:
            row = ensemble.iloc[0]
            lines.append(
                f"- Best ensemble row: {row['variant']} top_k={int(row['top_k'])}: "
                f"total_return={row['total_return']:.2%}, sharpe={row['sharpe']:.2f}, "
                f"max_drawdown={row['max_drawdown']:.2%}."
            )
    lines.extend(
        [
            "",
            "Interpretation:",
            "- Dagster should run this job; notebooks should load the saved artifacts and inspect results.",
            "- MLflow is the source of truth for run params, high-level metrics, and artifact files.",
        ]
    )
    return "\n".join(lines)


def _experiment_metrics(
    *,
    model_results: pd.DataFrame,
    strategy_scores: pd.DataFrame,
    backtest_summary: pd.DataFrame,
    symbols,
    event_symbols,
    elapsed_seconds: float,
) -> dict[str, float | int | None]:
    best = _best_backtest_row(backtest_summary)
    return {
        "symbols": int(len(symbols)),
        "event_symbols": int(len(event_symbols)),
        "trained_models": int((model_results["status"] == "ok").sum()) if "status" in model_results else int(len(model_results)),
        "strategy_sources": int(strategy_scores["strategy_source"].nunique()) if not strategy_scores.empty else 0,
        "best_sharpe": None if best is None else float(best["sharpe"]),
        "best_total_return": None if best is None else float(best["total_return"]),
        "best_max_drawdown": None if best is None else float(best["max_drawdown"]),
        "elapsed_seconds": float(elapsed_seconds),
    }


def _best_backtest_row(backtest_summary: pd.DataFrame):
    if backtest_summary.empty:
        return None
    return backtest_summary.sort_values(["sharpe", "total_return"], ascending=False).iloc[0]


def _config_params(config: MLTradingExperimentConfig) -> dict:
    params = asdict(config)
    params["top_k_values"] = list(config.top_k_values)
    params["event_families"] = list(config.event_families)
    params["oracle_frequencies"] = list(config.oracle_frequencies)
    return params


def _finite_metrics(metrics: dict) -> dict[str, float]:
    out = {}
    for key, value in metrics.items():
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric):
            out[key] = numeric
    return out


def _json_dumps(payload: dict) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, default=_json_default)


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return repr(value)


def _safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return cleaned.strip("._") or "model"


def _safe_int(value) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(value)


def _market_cap_text(value: int) -> str:
    if value >= 1_000_000_000_000:
        return f"{value // 1_000_000_000_000}T"
    if value >= 1_000_000_000:
        return f"{value // 1_000_000_000}B"
    return str(value)


def _prepare_quant_warehouse_import(path: str | None) -> None:
    if not path:
        return
    cleaned = str(Path(path).expanduser().resolve())
    sys.path[:] = [entry for entry in sys.path if str(Path(entry).expanduser().resolve()) != cleaned]
    sys.path.insert(0, cleaned)
    loaded_path = getattr(sys.modules.get("quant_warehouse"), "__file__", "")
    if loaded_path and not str(Path(loaded_path).expanduser().resolve()).startswith(cleaned):
        for module_name in list(sys.modules):
            if module_name == "quant_warehouse" or module_name.startswith("quant_warehouse."):
                sys.modules.pop(module_name, None)


def _warehouse_imports():
    from quant_warehouse.platforms.data_providers.fmp.target_engineering.event_pairs import EventPairStore
    from quant_warehouse.research_tools import (
        BinaryTargetConfig,
        FamilyEvaluationConfig,
        build_collapsed_bullish_event_target_panel,
        build_fundamental_feature_panel,
        build_oracle_trade_target_panel,
        cap_features_by_quality,
        load_fmp_event_pairs,
        screen_fmp_equity_universe,
    )
    from quant_warehouse.warehouse.api import Warehouse

    return (
        Warehouse,
        EventPairStore,
        BinaryTargetConfig,
        FamilyEvaluationConfig,
        build_collapsed_bullish_event_target_panel,
        build_fundamental_feature_panel,
        build_oracle_trade_target_panel,
        cap_features_by_quality,
        load_fmp_event_pairs,
        screen_fmp_equity_universe,
    )
