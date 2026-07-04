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

from quant_orchestrator.platforms.backtesting_frameworks.shared_book import (
    build_shared_book_weights,
    run_shared_book_backtest,
    shared_book_performance_metrics,
)
from quant_orchestrator.backtests.ml_score import (
    run_ml_score_signal_backtest,
)
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
    classification_probability_diagnostics,
    metric_correlation_summary,
    model_vs_trading_summary,
    prepare_family_dataset,
    write_ml_trading_artifact_files,
)
from quant_orchestrator.tracking import get_tracker


ExperimentMode = Literal["classifier", "classifier_ae"]
FeatureRepresentation = Literal["raw", "ae_only", "raw_plus_ae"]
TargetLabelMode = Literal["oracle_only", "event_and_oracle"]


@dataclass(frozen=True)
class MLTradingExperimentConfig:
    experiment_name: str = "gpu_rf_shared_book_1t"
    mode: ExperimentMode = "classifier"
    min_market_cap: int = 1_000_000_000_000
    symbols: tuple[str, ...] | None = None
    start_date: str = "1900-01-01"
    end_date: str | None = None
    train_end: str = "2019-12-31"
    oos_start: str = "2020-01-01"
    score_start: str | None = None
    top_k_values: tuple[int, ...] = (5, 10, 20, 40)
    entry_threshold: float = 0.50
    exit_threshold: float = 0.50
    min_feature_coverage: float = 0.50
    max_features_per_family: int = 50
    min_train_rows_per_family: int = 250
    min_classes_per_family: int = 2
    strategy_sources: tuple[str, ...] | None = None
    target_label_mode: TargetLabelMode = "oracle_only"
    feature_representations: tuple[FeatureRepresentation, ...] = ("raw",)
    capital_base: float = 1_000_000.0
    zipline_commission_per_share: float = 0.005
    zipline_slippage_bps: float = 5.0
    zipline_max_workers: int = 4
    run_zipline_backtests: bool = True
    include_yearly_vectorized_diagnostics: bool = True
    backtesting_py_symbol_cases_per_side: int = 10
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
    model_vs_trading: pd.DataFrame
    metric_correlations: pd.DataFrame
    yearly_backtest_summary: pd.DataFrame
    symbol_strategy_summary: pd.DataFrame
    symbol_robustness_summary: pd.DataFrame
    backtesting_py_symbol_validation: pd.DataFrame
    phase_timings: pd.DataFrame
    analysis_markdown: str
    elapsed_seconds: float

    @property
    def metrics(self) -> dict[str, float | int | None]:
        best = _best_backtest_row(self.backtest_summary)
        return {
            "symbols": _safe_int(self.backtest_summary["score_symbols"].max())
            if "score_symbols" in self.backtest_summary
            else int(self.strategy_scores["symbol"].nunique())
            if "symbol" in self.strategy_scores
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
    phase_started = started
    phase_rows: list[dict[str, object]] = []

    def mark_phase(phase: str, **metadata: object) -> None:
        nonlocal phase_started
        now = perf_counter()
        row = {
            "phase": phase,
            "seconds": float(now - phase_started),
            "elapsed_seconds": float(now - started),
        }
        row.update(metadata)
        phase_rows.append(row)
        phase_started = now

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
    mark_phase("setup")

    train_end = pd.Timestamp(config.train_end)
    oos_start = pd.Timestamp(config.oos_start)
    score_start = pd.Timestamp(config.score_start) if config.score_start else oos_start
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

    if config.symbols:
        symbols = tuple(dict.fromkeys(str(symbol).strip().upper() for symbol in config.symbols if str(symbol).strip()))
        _raw_universe = pd.DataFrame({"symbol": symbols})
        universe_eligibility = pd.DataFrame({"symbol": symbols, "eligible": True})
        universe_source = "config:symbols"
    else:
        symbols, _raw_universe, universe_eligibility, universe_source = screen_fmp_equity_universe(
            feature_config,
            warehouse=warehouse,
        )
    mark_phase("screen_universe", symbols=len(symbols))
    raw_feature_panel, raw_feature_metadata, _feature_diagnostics, feature_timings = (
        build_fundamental_feature_panel(symbols, feature_config, warehouse=warehouse)
    )
    mark_phase(
        "build_feature_panel",
        raw_feature_rows=len(raw_feature_panel),
        raw_feature_columns=len(raw_feature_panel.columns) if not raw_feature_panel.empty else 0,
    )
    selected_features, selected_feature_metadata, _feature_quality = cap_features_by_quality(
        raw_feature_panel,
        raw_feature_metadata,
        max_features=config.max_features_per_family,
    )
    selected_features, selected_feature_metadata = _filter_selected_strategy_sources(
        selected_features,
        selected_feature_metadata,
        strategy_sources=config.strategy_sources,
    )
    feature_panel = raw_feature_panel[["symbol", "date", *selected_features]].copy()
    feature_panel["symbol"] = feature_panel["symbol"].astype(str).str.upper()
    feature_panel["date"] = pd.to_datetime(feature_panel["date"], errors="coerce").dt.normalize()
    mark_phase("cap_features", selected_features=len(selected_features), selected_families=selected_feature_metadata["family"].nunique())

    event_symbols: tuple[str, ...] = ()
    event_load_seconds = 0.0
    collapsed_event_panel = pd.DataFrame()
    if config.target_label_mode == "event_and_oracle":
        events, event_diagnostics, event_load_seconds = load_fmp_event_pairs(
            symbols,
            target_config,
            event_store=event_store,
            include_historical=True,
        )
        mark_phase("load_event_pairs", event_rows=len(events), event_load_seconds=float(event_load_seconds))
        event_symbols = tuple(
            event_diagnostics.loc[event_diagnostics["combined_rows"].gt(0), "symbol"].sort_values()
        )
        collapsed_event_panel, _collapsed_event_metadata = build_collapsed_bullish_event_target_panel(
            feature_panel[["symbol", "date"]],
            events,
            target_config,
        )
    else:
        mark_phase("load_event_pairs", event_rows=0, event_load_seconds=0.0, skipped=True)
    oracle_label_rows, oracle_label_diagnostics, oracle_seconds = _build_oracle_trade_label_rows_sparse(
        symbols,
        target_config,
        warehouse=warehouse,
    )
    event_label_rows, event_label_diagnostics = (
        collapsed_label_rows(collapsed_event_panel, pd.DataFrame())
        if config.target_label_mode == "event_and_oracle"
        else (
            pd.DataFrame(columns=["symbol", "date", "collapsed_label", "label_source"]),
            pd.DataFrame(
                [
                    {
                        "source": "event_collapsed",
                        "candidate_rows": 0,
                        "used_rows": 0,
                        "dropped_rows": 0,
                        "note": "event labels skipped; oracle_only target_label_mode",
                    }
                ]
            ),
        )
    )
    label_rows = _combine_label_rows(event_label_rows, oracle_label_rows)
    label_diagnostics = pd.concat(
        [event_label_diagnostics, oracle_label_diagnostics],
        ignore_index=True,
    )
    mark_phase("build_labels", label_rows=len(label_rows), oracle_seconds=float(oracle_seconds))
    model_results, models = _train_family_models(
        config,
        feature_panel,
        selected_feature_metadata,
        label_rows,
        train_end=train_end,
        oos_start=oos_start,
    )
    mark_phase("train_family_models", trained_models=int((model_results["status"] == "ok").sum()) if "status" in model_results else len(model_results))
    strategy_scores, mean_scores = _score_family_models(
        config,
        feature_panel,
        models,
        score_start=score_start,
    )
    mark_phase("score_family_models", score_rows=len(strategy_scores), strategy_sources=strategy_scores["strategy_source"].nunique() if not strategy_scores.empty else 0)
    price_frames = _load_price_frames(
        warehouse,
        symbols,
        provider=config.provider,
        start=config.start_date,
        end=config.end_date,
    )
    mark_phase("load_price_frames", price_symbols=len(price_frames))
    backtest_summary, trade_log, yearly_backtest_summary, trade_generation_audit = _run_shared_book_backtests(
        config,
        strategy_scores,
        price_frames,
        oos_start=oos_start,
    )
    mark_phase(
        "shared_book_backtests",
        zipline_rows=len(backtest_summary),
        yearly_rows=len(yearly_backtest_summary),
        run_zipline_backtests=bool(config.run_zipline_backtests),
    )
    symbol_strategy_summary, symbol_robustness_summary = _run_symbol_rule_diagnostics(
        config,
        strategy_scores,
        price_frames,
        oos_start=oos_start,
    )
    mark_phase("symbol_rule_diagnostics", symbol_rows=len(symbol_strategy_summary), robustness_rows=len(symbol_robustness_summary))
    backtesting_py_symbol_validation = _run_backtesting_py_symbol_validation(
        config,
        strategy_scores,
        price_frames,
        symbol_strategy_summary,
        oos_start=oos_start,
    )
    mark_phase("backtesting_py_symbol_validation", validation_rows=len(backtesting_py_symbol_validation))
    model_oos_summary = _model_oos_summary(model_results)
    model_vs_trading = model_vs_trading_summary(model_results, backtest_summary)
    metric_correlations = metric_correlation_summary(
        model_vs_trading,
        x_cols=(
            "oos_accuracy",
            "oos_balanced_accuracy",
            "oos_macro_f1",
            "oos_log_loss",
            "oos_brier_macro",
            "oos_expected_calibration_error",
            "oos_mean_confidence",
        ),
        y_cols=("sharpe", "total_return", "max_drawdown", "win_rate"),
    )
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
        model_vs_trading=model_vs_trading,
        metric_correlations=metric_correlations,
        yearly_backtest_summary=yearly_backtest_summary,
        symbol_strategy_summary=symbol_strategy_summary,
        symbol_robustness_summary=symbol_robustness_summary,
        backtesting_py_symbol_validation=backtesting_py_symbol_validation,
        trade_generation_audit=trade_generation_audit,
        feature_timings=feature_timings,
        event_load_seconds=event_load_seconds,
        oracle_seconds=oracle_seconds,
        oracle_metadata_rows=len(oracle_label_diagnostics),
        label_diagnostics=label_diagnostics,
        eligible_symbols=int(universe_eligibility["eligible"].sum())
        if "eligible" in universe_eligibility
        else len(symbols),
    )
    elapsed_seconds = perf_counter() - started
    phase_timings = pd.DataFrame(phase_rows)
    result = MLTradingExperimentResult(
        config=config,
        mlflow_run_id=None,
        model_results=model_results,
        strategy_scores=strategy_scores,
        backtest_summary=backtest_summary,
        trade_log=trade_log,
        model_vs_trading=model_vs_trading,
        metric_correlations=metric_correlations,
        yearly_backtest_summary=yearly_backtest_summary,
        symbol_strategy_summary=symbol_strategy_summary,
        symbol_robustness_summary=symbol_robustness_summary,
        backtesting_py_symbol_validation=backtesting_py_symbol_validation,
        phase_timings=phase_timings,
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
            model_vs_trading=result.model_vs_trading,
            metric_correlations=result.metric_correlations,
            yearly_backtest_summary=result.yearly_backtest_summary,
            symbol_strategy_summary=result.symbol_strategy_summary,
            symbol_robustness_summary=result.symbol_robustness_summary,
            backtesting_py_symbol_validation=result.backtesting_py_symbol_validation,
            phase_timings=result.phase_timings,
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
                model_vs_trading=result.model_vs_trading,
                metric_correlations=result.metric_correlations,
                yearly_backtest_summary=result.yearly_backtest_summary,
                symbol_strategy_summary=result.symbol_strategy_summary,
                symbol_robustness_summary=result.symbol_robustness_summary,
                backtesting_py_symbol_validation=result.backtesting_py_symbol_validation,
                phase_timings=result.phase_timings,
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
            "base_family": payload.get("base_family", str(family)),
            "representation": payload.get("representation", "raw"),
            "features": features,
            "raw_features": list(payload.get("raw_features", features)),
            "feature_count": len(features),
            "has_classifier": payload.get("classifier") is not None,
            "has_autoencoder": payload.get("autoencoder") is not None,
            "has_feature_autoencoder": payload.get("feature_autoencoder") is not None,
            "classifier_file": "classifier.pkl",
            "autoencoder_dir": "autoencoder" if payload.get("autoencoder") is not None else None,
            "feature_autoencoder_dir": "feature_autoencoder" if payload.get("feature_autoencoder") is not None else None,
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
        feature_autoencoder = payload.get("feature_autoencoder")
        if feature_autoencoder is not None and feature_autoencoder is not autoencoder:
            _write_autoencoder_artifacts(model_dir / "feature_autoencoder", feature_autoencoder)
            metadata["feature_autoencoder_metadata"] = feature_autoencoder.metadata()
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


def _filter_selected_strategy_sources(
    selected_features: list[str],
    selected_feature_metadata: pd.DataFrame,
    *,
    strategy_sources: tuple[str, ...] | None,
) -> tuple[list[str], pd.DataFrame]:
    if not strategy_sources:
        return selected_features, selected_feature_metadata
    wanted = {str(source).strip() for source in strategy_sources if str(source).strip()}
    if not wanted or selected_feature_metadata.empty:
        return selected_features, selected_feature_metadata
    metadata = selected_feature_metadata.copy()
    metadata["strategy_source"] = metadata["source"].astype(str) + "." + metadata["family"].astype(str)
    metadata = metadata.loc[metadata["strategy_source"].isin(wanted)].drop(columns=["strategy_source"])
    features = [feature for feature in selected_features if feature in set(metadata["feature"].astype(str))]
    return features, metadata.reset_index(drop=True)


def _build_oracle_trade_label_rows_sparse(
    symbols,
    config,
    *,
    warehouse,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Build collapsed oracle labels without materializing dense per-k target columns."""

    from quant_warehouse.platforms.data_providers.fmp.target_engineering.strategy_solver import (
        solve_side_trades_by_frequency_batched_multi_k,
    )

    started = perf_counter()
    price_frames: dict[str, pd.DataFrame] = {}
    for raw_symbol in symbols:
        symbol = str(raw_symbol).strip().upper()
        if not symbol:
            continue
        prices = warehouse.read_prices(
            symbol,
            provider=config.provider,
            start=config.start_date,
            end=config.end_date,
        )
        if prices is None or prices.empty:
            continue
        price_frames[symbol] = prices

    long_dates: set[tuple[str, pd.Timestamp]] = set()
    short_dates: set[tuple[str, pd.Timestamp]] = set()
    k_by_frequency = config.oracle_trade_k_by_frequency or {"YE": tuple(range(1, 13))}
    for raw_frequency, raw_ks in k_by_frequency.items():
        frequency = str(raw_frequency or "").strip().upper()
        ks = tuple(dict.fromkeys(int(k) for k in raw_ks if int(k) > 0))
        if not frequency or not ks:
            continue
        trades_by_k = solve_side_trades_by_frequency_batched_multi_k(
            price_frames,
            ks=ks,
            freq=frequency,
            min_profit_pct=float(config.oracle_trade_min_profit_pct),
            long_entry_price_col=config.oracle_trade_long_entry_price_col,
            long_exit_price_col=config.oracle_trade_long_exit_price_col,
            short_entry_price_col=config.oracle_trade_short_entry_price_col,
            short_exit_price_col=config.oracle_trade_short_exit_price_col,
        )
        for trades_by_symbol in trades_by_k.values():
            for raw_symbol, trades in trades_by_symbol.items():
                symbol = str(raw_symbol).strip().upper()
                for trade in trades or []:
                    entry_date = getattr(trade.get("entry_row"), "name", None)
                    date = pd.to_datetime(entry_date, errors="coerce")
                    if pd.isna(date):
                        continue
                    key = (symbol, pd.Timestamp(date).normalize())
                    side = str(trade.get("side") or "").strip().lower()
                    if side == "long":
                        long_dates.add(key)
                    elif side == "short":
                        short_dates.add(key)

    ambiguous = long_dates & short_dates
    rows = [
        {"symbol": symbol, "date": date, "collapsed_label": "oracle_long", "label_source": "oracle_trade"}
        for symbol, date in sorted(long_dates - ambiguous)
    ]
    rows.extend(
        {
            "symbol": symbol,
            "date": date,
            "collapsed_label": "oracle_short",
            "label_source": "oracle_trade",
        }
        for symbol, date in sorted(short_dates - ambiguous)
    )
    labels = pd.DataFrame(rows, columns=["symbol", "date", "collapsed_label", "label_source"])
    diagnostics = pd.DataFrame(
        [
            {
                "source": "oracle_trade",
                "candidate_rows": len(long_dates | short_dates),
                "used_rows": len(labels),
                "dropped_rows": len(ambiguous),
                "note": "sparse collapsed oracle labels; ambiguous long+short rows dropped after k collapse",
            }
        ]
    )
    return labels, diagnostics, perf_counter() - started


def _combine_label_rows(*frames: pd.DataFrame) -> pd.DataFrame:
    usable = [frame for frame in frames if frame is not None and not frame.empty]
    if not usable:
        return pd.DataFrame(columns=["symbol", "date", "collapsed_label", "label_source"])
    labels = pd.concat(usable, ignore_index=True)
    labels["symbol"] = labels["symbol"].astype(str).str.upper()
    labels["date"] = pd.to_datetime(labels["date"], errors="coerce").dt.normalize()
    return (
        labels.dropna(subset=["symbol", "date", "collapsed_label"])
        .drop_duplicates()
        .sort_values(["date", "symbol", "collapsed_label"])
        .reset_index(drop=True)
    )


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
    representations = _normalized_feature_representations(config.feature_representations)
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
        autoencoder = None
        ae_fit_seconds = np.nan
        ae_metadata = {}
        needs_autoencoder = config.mode == "classifier_ae" or any(rep in {"ae_only", "raw_plus_ae"} for rep in representations)
        if needs_autoencoder:
            ae_started = perf_counter()
            autoencoder = LatentAutoencoderIndex.fit(train, features=features, config=config.ae_config)
            ae_fit_seconds = perf_counter() - ae_started
            ae_metadata = {} if autoencoder is None else autoencoder.metadata()
        for representation in representations:
            if representation in {"ae_only", "raw_plus_ae"} and autoencoder is None:
                model_rows.append(
                    {
                        "source": source,
                        "family": _represented_family_name(str(family), representation),
                        "base_family": family,
                        "representation": representation,
                        "status": "skipped_autoencoder_unavailable",
                        "features": len(features),
                        "rows": len(family_frame),
                        "train_rows": len(train),
                        "oos_rows": len(oos),
                    }
                )
                continue
            train_rep, train_features = _apply_feature_representation(train, features, autoencoder=autoencoder, representation=representation)
            oos_rep, _ = _apply_feature_representation(oos, features, autoencoder=autoencoder, representation=representation)
            family_rep = _represented_family_name(str(family), representation)
            fit_started = perf_counter()
            classifier = RapidsRandomForestClassifier.fit(
                train_rep,
                features=train_features,
                target_col="collapsed_label",
                random_state=config.random_seed,
                params=config.rf_params,
            )
            classifier_fit_seconds = perf_counter() - fit_started
            models[(source, family_rep)] = {
                "classifier": classifier,
                "autoencoder": autoencoder if config.mode == "classifier_ae" else None,
                "feature_autoencoder": autoencoder,
                "features": train_features,
                "raw_features": features,
                "base_family": str(family),
                "representation": representation,
            }
            train_proba = classifier.predict_proba_frame(train_rep, train_features)
            train_scores = classification_probability_diagnostics(
                train_rep,
                train_proba,
                target_col="collapsed_label",
                labels=classifier.encoder.classes_,
            )
            if not oos_rep.empty and oos_rep["collapsed_label"].nunique() > 1:
                oos_proba = classifier.predict_proba_frame(oos_rep, train_features)
                oos_scores = classification_probability_diagnostics(
                    oos_rep,
                    oos_proba,
                    target_col="collapsed_label",
                    labels=classifier.encoder.classes_,
                )
            else:
                oos_scores = {
                    "rows": len(oos_rep),
                    "accuracy": np.nan,
                    "balanced_accuracy": np.nan,
                    "macro_f1": np.nan,
                    "log_loss": np.nan,
                    "brier_macro": np.nan,
                    "expected_calibration_error": np.nan,
                    "mean_confidence": np.nan,
                }
            model_rows.append(
                {
                    "source": source,
                    "family": family_rep,
                    "base_family": family,
                    "representation": representation,
                    "strategy_source": f"{source}.{family_rep}",
                    "status": "ok",
                    "features": len(train_features),
                    "raw_features": len(features),
                    "ae_features": len([col for col in train_features if str(col).startswith("ae_")]),
                    "rows": len(family_frame),
                    "train_rows": len(train_rep),
                    "oos_rows": len(oos_rep),
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


def _normalized_feature_representations(representations: tuple[FeatureRepresentation, ...]) -> tuple[FeatureRepresentation, ...]:
    allowed = {"raw", "ae_only", "raw_plus_ae"}
    cleaned = []
    for representation in representations or ("raw",):
        value = str(representation)
        if value not in allowed:
            raise ValueError(f"unknown feature representation {value!r}; expected one of {sorted(allowed)}")
        if value not in cleaned:
            cleaned.append(value)
    return tuple(cleaned) or ("raw",)


def _represented_family_name(family: str, representation: str) -> str:
    return str(family) if representation == "raw" else f"{family}__{representation}"


def _apply_feature_representation(
    frame: pd.DataFrame,
    raw_features: list[str],
    *,
    autoencoder,
    representation: str,
) -> tuple[pd.DataFrame, list[str]]:
    if representation == "raw":
        return frame.copy(), list(raw_features)
    if autoencoder is None:
        return pd.DataFrame(), []
    base_cols = [col for col in ("symbol", "date", "collapsed_label", "label_source") if col in frame.columns]
    out = frame[base_cols].copy()
    ae_features = autoencoder.transform_features(frame, prefix="ae").reset_index(drop=True)
    ae_feature_cols = list(ae_features.columns)
    out = pd.concat([out.reset_index(drop=True), ae_features.astype("float32")], axis=1)
    if representation == "ae_only":
        return out, ae_feature_cols
    if representation == "raw_plus_ae":
        raw = frame[list(raw_features)].reset_index(drop=True).astype("float32")
        out = pd.concat([out, raw], axis=1)
        return out, [*list(raw_features), *ae_feature_cols]
    raise ValueError(f"unknown feature representation {representation!r}")


def _score_family_models(
    config: MLTradingExperimentConfig,
    feature_panel: pd.DataFrame,
    models: dict,
    *,
    score_start: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred_frames = []
    for (source, family), payload in models.items():
        pred_input = build_family_prediction_frame(
            feature_panel,
            payload.get("raw_features", payload["features"]),
            min_feature_coverage=config.min_feature_coverage,
        )
        if pred_input.empty:
            continue
        pred_rep, pred_features = _apply_feature_representation(
            pred_input,
            payload.get("raw_features", payload["features"]),
            autoencoder=payload.get("feature_autoencoder"),
            representation=payload.get("representation", "raw"),
        )
        proba = payload["classifier"].predict_proba_frame(pred_rep, pred_features)
        ae_frame = (
            payload["autoencoder"].familiarity(pred_input)
            if config.mode == "classifier_ae" and payload["autoencoder"] is not None
            else None
        )
        pred_frames.append(
            build_strategy_score_frame(
                source=str(source),
                family=str(family),
                prediction_frame=pred_rep,
                probability_frame=proba,
                ae_familiarity_frame=ae_frame,
                apply_ae_to_exits=False,
            )
        )
    single_model_scores = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()
    if single_model_scores.empty:
        return single_model_scores, pd.DataFrame()
    single_model_scores = single_model_scores.loc[pd.to_datetime(single_model_scores["date"]).ge(score_start)].copy()
    mean_scores = (
        single_model_scores.groupby(["symbol", "date"], as_index=False)
        .agg(
            long_score=("long_score", "mean"),
            short_score=("short_score", "mean"),
            long_exit_score=("long_exit_score", "mean"),
            short_exit_score=("short_exit_score", "mean"),
            classifier_long_score=("classifier_long_score", "mean"),
            classifier_short_score=("classifier_short_score", "mean"),
            long_agree_count=("long_agree_count", "sum"),
            short_agree_count=("short_agree_count", "sum"),
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
        "long_agree_count",
        "short_agree_count",
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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    if strategy_scores.empty or not price_frames:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}
    wide_close = pd.DataFrame({symbol: frame["close"] for symbol, frame in price_frames.items()}).sort_index().ffill()
    next_returns = wide_close.pct_change().shift(-1)
    oos_dates = pd.DatetimeIndex(sorted(set(strategy_scores["date"]).intersection(next_returns.index)))
    oos_dates = oos_dates[oos_dates >= oos_start]
    trade_symbols = tuple(sorted(set(strategy_scores["symbol"]).intersection(next_returns.columns)))
    if len(oos_dates) == 0 or not trade_symbols:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}
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
    yearly_rows = []
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
                if config.include_yearly_vectorized_diagnostics:
                    yearly_rows.extend(
                        _yearly_vectorized_backtest_rows(
                            weights,
                            trades,
                            next_returns,
                            metadata=metadata,
                            cost_bps=config.zipline_slippage_bps + 0.5,
                            capital_base=config.capital_base,
                        )
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
    backtest_summary = (
        run_zipline_shared_book_summary_jobs(
            zipline_jobs,
            max_workers=config.zipline_max_workers,
        )
        if config.run_zipline_backtests
        else pd.DataFrame()
    )
    usable_trade_logs = [
        frame.dropna(axis=1, how="all")
        for frame in trade_logs
        if frame is not None and not frame.empty and not frame.dropna(axis=1, how="all").empty
    ]
    trade_log = pd.concat(usable_trade_logs, ignore_index=True) if usable_trade_logs else pd.DataFrame()
    yearly_summary = pd.DataFrame(yearly_rows)
    audit = {
        "strategy_sources": len(strategy_source_order),
        "variants": 3,
        "top_k_values": len(config.top_k_values),
        "zipline_jobs": len(zipline_jobs),
        "run_zipline_backtests": bool(config.run_zipline_backtests),
        "yearly_vectorized_rows": len(yearly_summary),
        "score_rows_total": int(strategy_scores.groupby("strategy_source").size().sum()),
        "signal_events_total": int(sum(len(frame) for frame in trade_logs)),
        "max_trade_cap": None,
    }
    return (
        backtest_summary.sort_values(["strategy_source", "variant", "top_k"]).reset_index(drop=True)
        if not backtest_summary.empty
        else backtest_summary,
        trade_log,
        yearly_summary.sort_values(["strategy_source", "variant", "top_k", "year"]).reset_index(drop=True)
        if not yearly_summary.empty
        else yearly_summary,
        audit,
    )


def _yearly_vectorized_backtest_rows(
    weights: pd.DataFrame,
    trades: pd.DataFrame,
    next_returns: pd.DataFrame,
    *,
    metadata: dict,
    cost_bps: float,
    capital_base: float,
) -> list[dict[str, object]]:
    aligned_returns = next_returns.reindex(index=weights.index, columns=weights.columns).fillna(0.0)
    rows = []
    for year, year_weights in weights.groupby(weights.index.year):
        year_returns_input = aligned_returns.reindex(index=year_weights.index, columns=year_weights.columns).fillna(0.0)
        year_trades = (
            trades.loc[pd.to_datetime(trades["date"], errors="coerce").dt.year.eq(int(year))].copy()
            if not trades.empty and "date" in trades.columns
            else trades
        )
        returns, equity, _turnover = run_shared_book_backtest(
            year_weights,
            year_returns_input,
            cost_bps=cost_bps,
            capital_base=capital_base,
        )
        row = shared_book_performance_metrics(
            returns,
            equity,
            year_weights,
            year_trades,
            framework="vectorized_shared_book_yearly",
            variant=str(metadata["variant"]),
            top_k=int(metadata["top_k"]),
            cost_bps=cost_bps,
        )
        row.update(
            {
                "year": int(year),
                "strategy_source": metadata["strategy_source"],
                "source": metadata["source"],
                "family": metadata["family"],
                "score_rows": metadata["score_rows"],
                "score_symbols": metadata["score_symbols"],
                "score_dates": metadata["score_dates"],
            }
        )
        rows.append(row)
    return rows


def _run_symbol_rule_diagnostics(
    config: MLTradingExperimentConfig,
    strategy_scores: pd.DataFrame,
    price_frames: dict[str, pd.DataFrame],
    *,
    oos_start: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if strategy_scores.empty or not price_frames:
        return pd.DataFrame(), pd.DataFrame()
    wide_close = pd.DataFrame({symbol: frame["close"] for symbol, frame in price_frames.items()}).sort_index().ffill()
    next_returns = wide_close.pct_change().shift(-1)
    trade_symbols = tuple(sorted(set(strategy_scores["symbol"]).intersection(next_returns.columns)))
    strategy_source_order = ["ensemble_mean"] + sorted(
        source for source in strategy_scores["strategy_source"].dropna().unique().tolist() if source != "ensemble_mean"
    )
    rows = []
    returns_matrix = next_returns.loc[next_returns.index >= oos_start, list(trade_symbols)].fillna(0.0)
    if returns_matrix.empty:
        return pd.DataFrame(), pd.DataFrame()
    buy_hold_metrics = _single_symbol_metrics_matrix(returns_matrix, returns_matrix.ne(0).astype(float))
    for strategy_source in strategy_source_order:
        score_frame = strategy_scores.loc[strategy_scores["strategy_source"].eq(strategy_source)].copy()
        if score_frame.empty:
            continue
        source_value = score_frame["source"].iloc[0]
        family_value = score_frame["family"].iloc[0]
        score_frame["date"] = pd.to_datetime(score_frame["date"], errors="coerce").dt.normalize()
        long_scores = (
            score_frame.pivot_table(index="date", columns="symbol", values="long_score", aggfunc="mean")
            .reindex(index=returns_matrix.index, columns=returns_matrix.columns)
            .fillna(0.0)
        )
        short_scores = (
            score_frame.pivot_table(index="date", columns="symbol", values="short_score", aggfunc="mean")
            .reindex(index=returns_matrix.index, columns=returns_matrix.columns)
            .fillna(0.0)
        )
        long_agree_counts = short_agree_counts = model_counts = None
        if {"long_agree_count", "short_agree_count", "model_count"}.issubset(score_frame.columns):
            long_agree_counts = (
                score_frame.pivot_table(index="date", columns="symbol", values="long_agree_count", aggfunc="mean")
                .reindex(index=returns_matrix.index, columns=returns_matrix.columns)
                .fillna(0.0)
            )
            short_agree_counts = (
                score_frame.pivot_table(index="date", columns="symbol", values="short_agree_count", aggfunc="mean")
                .reindex(index=returns_matrix.index, columns=returns_matrix.columns)
                .fillna(0.0)
            )
            model_counts = (
                score_frame.pivot_table(index="date", columns="symbol", values="model_count", aggfunc="mean")
                .reindex(index=returns_matrix.index, columns=returns_matrix.columns)
                .fillna(0.0)
            )
        for variant in ("long_only", "short_only", "long_short"):
            positions = _single_symbol_positions_matrix(
                long_scores,
                short_scores,
                long_agree_counts=long_agree_counts,
                short_agree_counts=short_agree_counts,
                model_counts=model_counts,
                variant=variant,
                entry_threshold=config.entry_threshold,
                exit_threshold=config.exit_threshold,
            )
            shifted_positions = positions.shift(1).fillna(0.0)
            turnover = shifted_positions.diff().abs().fillna(shifted_positions.abs())
            costs = turnover * ((config.zipline_slippage_bps + 0.5) / 10_000.0)
            strategy_returns = shifted_positions * returns_matrix - costs
            metrics = _single_symbol_metrics_matrix(strategy_returns, turnover)
            for symbol in returns_matrix.columns:
                strategy_total = metrics["total_return"].get(symbol, np.nan)
                buy_hold_total = buy_hold_metrics["total_return"].get(symbol, np.nan)
                rows.append(
                    {
                        "strategy_source": strategy_source,
                        "source": source_value,
                        "family": family_value,
                        "symbol": symbol,
                        "variant": variant,
                        "days": int(returns_matrix[symbol].notna().sum()),
                        "strategy_total_return": float(strategy_total),
                        "strategy_sharpe": float(metrics["sharpe"].get(symbol, np.nan)),
                        "strategy_max_drawdown": float(metrics["max_drawdown"].get(symbol, np.nan)),
                        "strategy_win_rate": float(metrics["win_rate"].get(symbol, np.nan)),
                        "trades": int(metrics["trades"].get(symbol, 0)),
                        "buy_hold_total_return": float(buy_hold_total),
                        "buy_hold_sharpe": float(buy_hold_metrics["sharpe"].get(symbol, np.nan)),
                        "buy_hold_max_drawdown": float(buy_hold_metrics["max_drawdown"].get(symbol, np.nan)),
                        "excess_total_return": float(strategy_total - buy_hold_total),
                        "beats_buy_hold": bool(strategy_total > buy_hold_total),
                        "active_day_rate": float(shifted_positions[symbol].ne(0).mean()),
                        "long_day_rate": float(shifted_positions[symbol].gt(0).mean()),
                        "short_day_rate": float(shifted_positions[symbol].lt(0).mean()),
                    }
                )
    symbol_summary = pd.DataFrame(rows)
    if symbol_summary.empty:
        return symbol_summary, pd.DataFrame()
    robustness = (
        symbol_summary.groupby(["strategy_source", "source", "family", "variant"], as_index=False)
        .agg(
            symbols=("symbol", "nunique"),
            median_strategy_total_return=("strategy_total_return", "median"),
            mean_strategy_total_return=("strategy_total_return", "mean"),
            median_buy_hold_total_return=("buy_hold_total_return", "median"),
            mean_excess_total_return=("excess_total_return", "mean"),
            median_excess_total_return=("excess_total_return", "median"),
            beat_buy_hold_rate=("beats_buy_hold", "mean"),
            median_strategy_sharpe=("strategy_sharpe", "median"),
            median_buy_hold_sharpe=("buy_hold_sharpe", "median"),
            median_max_drawdown=("strategy_max_drawdown", "median"),
            median_active_day_rate=("active_day_rate", "median"),
            total_trades=("trades", "sum"),
        )
        .sort_values(["beat_buy_hold_rate", "median_excess_total_return", "median_strategy_sharpe"], ascending=False)
        .reset_index(drop=True)
    )
    return symbol_summary.sort_values(["strategy_source", "variant", "symbol"]).reset_index(drop=True), robustness


def _single_symbol_positions_matrix(
    long_scores: pd.DataFrame,
    short_scores: pd.DataFrame,
    *,
    long_agree_counts: pd.DataFrame | None = None,
    short_agree_counts: pd.DataFrame | None = None,
    model_counts: pd.DataFrame | None = None,
    variant: str,
    entry_threshold: float,
    exit_threshold: float,
) -> pd.DataFrame:
    positions = np.zeros(long_scores.shape, dtype="float64")
    current = np.zeros(long_scores.shape[1], dtype="float64")
    long_values = long_scores.to_numpy(dtype="float64", copy=False)
    short_values = short_scores.to_numpy(dtype="float64", copy=False)
    long_agree_values = long_agree_counts.to_numpy(dtype="float64", copy=False) if long_agree_counts is not None else None
    short_agree_values = short_agree_counts.to_numpy(dtype="float64", copy=False) if short_agree_counts is not None else None
    model_count_values = model_counts.to_numpy(dtype="float64", copy=False) if model_counts is not None else None
    for row_idx in range(long_values.shape[0]):
        long_row = long_values[row_idx]
        short_row = short_values[row_idx]
        if model_count_values is not None and long_agree_values is not None and short_agree_values is not None:
            model_row = model_count_values[row_idx]
            long_ok = (model_row > 0) & (long_agree_values[row_idx] == model_row)
            short_ok = (model_row > 0) & (short_agree_values[row_idx] == model_row)
        else:
            long_ok = long_row >= short_row
            short_ok = short_row > long_row
        if variant == "long_only":
            current = np.where((current > 0) & ~long_ok, 0.0, current)
            current = np.where((current == 0) & (long_row > entry_threshold) & long_ok, 1.0, current)
        elif variant == "short_only":
            current = np.where((current < 0) & ~short_ok, 0.0, current)
            current = np.where((current == 0) & (short_row > entry_threshold) & short_ok, -1.0, current)
        elif variant == "long_short":
            current = np.where((current > 0) & ~long_ok, 0.0, current)
            current = np.where((current < 0) & ~short_ok, 0.0, current)
            current = np.where(
                (current == 0) & (long_row > entry_threshold) & long_ok,
                1.0,
                np.where((current == 0) & (short_row > entry_threshold) & short_ok, -1.0, current),
            )
        else:
            raise ValueError(f"unknown variant {variant!r}")
        positions[row_idx] = current
    return pd.DataFrame(positions, index=long_scores.index, columns=long_scores.columns)


def _single_symbol_metrics_matrix(strategy_returns: pd.DataFrame, turnover: pd.DataFrame) -> dict[str, pd.Series]:
    clean = strategy_returns.astype("float64").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    equity = (1.0 + clean).cumprod(axis=0)
    total_return = equity.iloc[-1] - 1.0 if len(equity) else pd.Series(dtype="float64")
    std = clean.std(axis=0).replace(0.0, np.nan)
    sharpe = clean.mean(axis=0).div(std) * np.sqrt(252)
    drawdown = equity.div(equity.cummax(axis=0)).sub(1.0)
    return {
        "total_return": total_return,
        "sharpe": sharpe,
        "max_drawdown": drawdown.min(axis=0),
        "win_rate": clean.gt(0).mean(axis=0),
        "trades": turnover.gt(0).sum(axis=0),
    }


def _run_backtesting_py_symbol_validation(
    config: MLTradingExperimentConfig,
    strategy_scores: pd.DataFrame,
    price_frames: dict[str, pd.DataFrame],
    symbol_strategy_summary: pd.DataFrame,
    *,
    oos_start: pd.Timestamp,
) -> pd.DataFrame:
    cases_per_side = int(config.backtesting_py_symbol_cases_per_side)
    if cases_per_side <= 0 or symbol_strategy_summary.empty:
        return pd.DataFrame()
    candidates = pd.concat(
        [
            symbol_strategy_summary.sort_values(
                ["excess_total_return", "strategy_sharpe"],
                ascending=False,
            ).head(cases_per_side),
            symbol_strategy_summary.sort_values(
                ["excess_total_return", "strategy_sharpe"],
                ascending=True,
            ).head(cases_per_side),
        ],
        ignore_index=True,
    ).drop_duplicates(["strategy_source", "symbol", "variant"])
    rows = []
    for case in candidates.itertuples(index=False):
        strategy_source = str(case.strategy_source)
        symbol = str(case.symbol).upper()
        variant = str(case.variant)
        prices = price_frames.get(symbol)
        if prices is None or prices.empty:
            continue
        score_frame = strategy_scores.loc[
            strategy_scores["strategy_source"].eq(strategy_source)
            & strategy_scores["symbol"].astype(str).str.upper().eq(symbol)
        ].copy()
        if score_frame.empty:
            continue
        price_frame = prices.loc[pd.DatetimeIndex(prices.index) >= oos_start].copy()
        if len(price_frame) < 3:
            continue
        try:
            result = run_ml_score_signal_backtest(
                price_frame,
                score_frame,
                symbol=symbol,
                strategy_source=strategy_source,
                variant=variant,
                entry_threshold=config.entry_threshold,
                exit_threshold=config.exit_threshold,
                cash=config.capital_base,
                commission_bps=0.5,
                spread_bps=config.zipline_slippage_bps,
            )
        except Exception as exc:  # pragma: no cover - backend/runtime specific
            rows.append(
                {
                    "framework": "backtesting_py_signal",
                    "strategy_source": strategy_source,
                    "symbol": symbol,
                    "variant": variant,
                    "status": "error",
                    "error": repr(exc),
                }
            )
            continue
        row = dict(result.summary)
        row.update(
            {
                "status": "ok",
                "vectorized_strategy_total_return": float(case.strategy_total_return),
                "vectorized_buy_hold_total_return": float(case.buy_hold_total_return),
                "vectorized_excess_total_return": float(case.excess_total_return),
                "vectorized_strategy_sharpe": float(case.strategy_sharpe)
                if pd.notna(case.strategy_sharpe)
                else np.nan,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["status", "excess_total_return"], ascending=[True, False]).reset_index(drop=True) if rows else pd.DataFrame()


def _single_symbol_positions(
    scores: pd.DataFrame,
    *,
    variant: str,
    entry_threshold: float,
    exit_threshold: float,
) -> pd.Series:
    long_score = pd.to_numeric(scores["long_score"], errors="coerce").fillna(0.0)
    short_score = pd.to_numeric(scores["short_score"], errors="coerce").fillna(0.0)
    if {"long_agree_count", "short_agree_count", "model_count"}.issubset(scores.columns):
        model_count = pd.to_numeric(scores["model_count"], errors="coerce").fillna(0.0)
        long_agree = pd.to_numeric(scores["long_agree_count"], errors="coerce").fillna(0.0)
        short_agree = pd.to_numeric(scores["short_agree_count"], errors="coerce").fillna(0.0)
    else:
        model_count = long_agree = short_agree = None
    position = []
    current = 0.0
    for idx, (long_value, short_value) in enumerate(zip(long_score, short_score)):
        if model_count is not None and long_agree is not None and short_agree is not None:
            models = float(model_count.iloc[idx])
            long_ok = models > 0 and float(long_agree.iloc[idx]) == models
            short_ok = models > 0 and float(short_agree.iloc[idx]) == models
        else:
            long_ok = long_value >= short_value
            short_ok = short_value > long_value
        if variant == "long_only":
            if current > 0 and not long_ok:
                current = 0.0
            elif current == 0 and long_value > entry_threshold and long_ok:
                current = 1.0
        elif variant == "short_only":
            if current < 0 and not short_ok:
                current = 0.0
            elif current == 0 and short_value > entry_threshold and short_ok:
                current = -1.0
        elif variant == "long_short":
            if current > 0 and not long_ok:
                current = 0.0
            elif current < 0 and not short_ok:
                current = 0.0
            if current == 0 and long_value > entry_threshold and long_ok:
                current = 1.0
            elif current == 0 and short_value > entry_threshold and short_ok:
                current = -1.0
        else:
            raise ValueError(f"unknown variant {variant!r}")
        position.append(current)
    return pd.Series(position, index=scores.index, dtype="float64")


def _single_symbol_metrics(strategy_returns: pd.Series, buy_hold_returns: pd.Series, *, trades: int) -> dict[str, float | int]:
    clean = pd.to_numeric(strategy_returns, errors="coerce").fillna(0.0)
    equity = (1.0 + clean).cumprod()
    total_return = float(equity.iloc[-1] - 1.0) if len(equity) else np.nan
    sharpe = float(clean.mean() / clean.std() * np.sqrt(252)) if len(clean) and clean.std() else np.nan
    drawdown = equity / equity.cummax() - 1.0 if len(equity) else pd.Series(dtype="float64")
    return {
        "total_return": total_return,
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()) if len(drawdown) else np.nan,
        "win_rate": float(clean.gt(0).mean()) if len(clean) else np.nan,
        "trades": int(trades),
    }


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
    model_vs_trading: pd.DataFrame,
    metric_correlations: pd.DataFrame,
    yearly_backtest_summary: pd.DataFrame,
    symbol_strategy_summary: pd.DataFrame,
    symbol_robustness_summary: pd.DataFrame,
    backtesting_py_symbol_validation: pd.DataFrame,
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
        f"- Feature representations: {list(config.feature_representations)}.",
        f"- Universe: {len(symbols)} FMP {cap_text}+ symbols; {len(event_symbols)} had event coverage; universe_source={universe_source}; eligible_symbols={eligible_symbols}.",
        f"- Training window: all available rows through {config.train_end}. Out-of-sample model and trading window starts {config.oos_start}.",
        f"- Trained feature-family models: {int((model_results['status'] == 'ok').sum()) if 'status' in model_results else len(model_results)}.",
        f"- Strategy sources traded: {strategy_scores['strategy_source'].nunique() if not strategy_scores.empty else 0} total.",
        f"- Ensemble OOS prediction rows: {len(mean_scores):,} across {mean_scores['symbol'].nunique() if not mean_scores.empty else 0} symbols and {mean_scores['date'].nunique() if not mean_scores.empty else 0} dates.",
        f"- Event load seconds: {event_load_seconds:.3f}; oracle seconds: {oracle_seconds:.3f}; oracle metadata rows: {oracle_metadata_rows}.",
        f"- Feature timings: {feature_timings}.",
        f"- Strategy variants: long_only, short_only, long_short with top_k={list(config.top_k_values)}.",
        f"- Execution: native Zipline shared-book engine, ${config.zipline_commission_per_share:.3f}/share commission, {config.zipline_slippage_bps:.1f} bps slippage.",
        "- Diagnostics: model-vs-trading, probability calibration, yearly vectorized shared-book summaries, per-symbol buy-and-hold robustness, and sampled backtesting.py single-symbol validations are saved as MLflow artifacts.",
    ]
    if "representation" in model_results.columns and not model_results.empty:
        rep_counts = model_results.loc[model_results["status"].eq("ok")].groupby("representation").size().to_dict()
        lines.append(f"- Trained representation counts: {rep_counts}.")
        if any(rep in set(config.feature_representations) for rep in ("ae_only", "raw_plus_ae")):
            ae_feature_mean = pd.to_numeric(model_results.get("ae_features", pd.Series(dtype=float)), errors="coerce").mean()
            raw_feature_mean = pd.to_numeric(model_results.get("raw_features", pd.Series(dtype=float)), errors="coerce").mean()
            lines.append(
                f"- AE-derived classifier features are Orchestrator experiment features only; mean raw_features={raw_feature_mean:.1f}, mean ae_features={ae_feature_mean:.1f}."
            )
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
            f"{trade_generation_audit.get('yearly_vectorized_rows', 0):,} yearly diagnostic rows, "
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
    if not model_vs_trading.empty:
        top_family = model_vs_trading.sort_values(["sharpe", "total_return"], ascending=False).iloc[0]
        lines.append(
            f"- Best single-family trading row: {top_family['strategy_source']} / {top_family['variant']} "
            f"top_k={int(top_family['top_k'])}: sharpe={top_family['sharpe']:.2f}, "
            f"macro_f1={top_family.get('oos_macro_f1', np.nan):.4f}, "
            f"calibration_error={top_family.get('oos_expected_calibration_error', np.nan):.4f}."
        )
    if not metric_correlations.empty:
        macro = metric_correlations.loc[
            metric_correlations["x"].eq("oos_macro_f1") & metric_correlations["y"].eq("sharpe")
        ]
        if not macro.empty:
            row = macro.iloc[0]
            lines.append(
                f"- Model/trading correlation check: oos_macro_f1 vs Sharpe has spearman={row['spearman']:.3f} "
                f"across {int(row['rows'])} family models."
            )
    if not yearly_backtest_summary.empty:
        yearly_best = (
            yearly_backtest_summary.sort_values(["strategy_source", "year", "sharpe"], ascending=[True, True, False])
            .groupby(["strategy_source", "year"], as_index=False)
            .head(1)
        )
        stable = (
            yearly_best.groupby("strategy_source", as_index=False)
            .agg(years=("year", "nunique"), positive_year_rate=("total_return", lambda s: float(pd.to_numeric(s, errors="coerce").gt(0).mean())))
            .sort_values(["positive_year_rate", "years"], ascending=False)
            .head(1)
        )
        if not stable.empty:
            row = stable.iloc[0]
            lines.append(
                f"- Yearly diagnostic stability leader: {row['strategy_source']} had positive-return best rows in "
                f"{row['positive_year_rate']:.1%} of {int(row['years'])} years."
            )
    if not symbol_robustness_summary.empty:
        broad = symbol_robustness_summary.sort_values(
            ["beat_buy_hold_rate", "median_excess_total_return", "median_strategy_sharpe"],
            ascending=False,
        ).iloc[0]
        lines.append(
            f"- Per-symbol robustness leader: {broad['strategy_source']} / {broad['variant']} beat buy-and-hold on "
            f"{broad['beat_buy_hold_rate']:.1%} of {int(broad['symbols'])} symbols with median excess return "
            f"{broad['median_excess_total_return']:.2%}."
        )
    if not backtesting_py_symbol_validation.empty and "status" in backtesting_py_symbol_validation.columns:
        ok = backtesting_py_symbol_validation.loc[backtesting_py_symbol_validation["status"].eq("ok")]
        if not ok.empty:
            agreement = (
                np.sign(pd.to_numeric(ok["excess_total_return"], errors="coerce"))
                == np.sign(pd.to_numeric(ok["vectorized_excess_total_return"], errors="coerce"))
            ).mean()
            lines.append(
                f"- backtesting.py validation sampled {len(ok)} top/bottom symbol cases; excess-return sign agreed with the vectorized scan on {agreement:.1%}."
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
    params["feature_representations"] = list(config.feature_representations)
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
