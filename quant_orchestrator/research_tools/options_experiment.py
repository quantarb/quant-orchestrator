from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from types import SimpleNamespace
from pathlib import Path
import pickle
from time import perf_counter
from typing import Any, Literal

import numpy as np
import pandas as pd
from quant_warehouse.platforms.data_providers.thetadata.settlement import (
    OptionSettlement,
    settle_option_exit,
)

from quant_orchestrator.platforms.backtesting_frameworks.reporting import (
    NormalizedBacktestReport,
    normalize_trade_windows as normalize_report_trade_windows,
    report_trade_windows,
)
from quant_orchestrator.research_tools.option_trade_execution import (
    OptionTradeExecutionBatch,
    OptionTradeExecutor,
    execute_rule_trade_windows,
)
from quant_orchestrator.tracking import get_tracker


@dataclass(frozen=True)
class SharedSplitConfig:
    insample_end: str = "2024-12-31"

    @property
    def insample_end_ts(self) -> pd.Timestamp:
        return pd.Timestamp(self.insample_end).normalize()

    @property
    def oos_start_ts(self) -> pd.Timestamp:
        return self.insample_end_ts + pd.Timedelta(days=1)


@dataclass(frozen=True)
class OptionRetrievalConfig:
    option_universe: Literal["filtered", "full_chain_actions"] = "filtered"
    min_dte: int = 30
    max_dte: int = 90
    target_dte: int = 45
    max_abs_moneyness: float = 0.15
    min_entry_mid: float = 0.25
    max_entry_spread_pct: float = 0.20
    max_candidates_per_trade: int = 40
    require_affordable: bool = True
    min_affordable_contracts: int = 1
    exit_lookback_days: int = 7
    price_eval_candidates: bool = True
    chain_columns: tuple[str, ...] = (
        "snapshot_date",
        "contract_symbol",
        "expiration",
        "strike",
        "option_type",
        "bid",
        "ask",
        "mid",
        "delta",
        "gamma",
        "theta",
        "vega",
        "rho",
        "iv",
        "implied_volatility",
        "underlying_price",
        "volume",
        "open_interest",
    )


@dataclass(frozen=True)
class OptopsyExecutionConfig:
    capital: float = 100_000.0
    quantity: int = 1
    max_positions: int = 5
    multiplier: int = 100
    selector: str = "first"
    sizing_mode: str = "portfolio_fraction"
    allocation_fraction: float | None = None
    top_k_column: str = "top_k"


@dataclass(frozen=True)
class OptionMvBasketConfig:
    enabled: bool = True
    risk_aversion: float = 1.0
    max_weight: float = 0.35
    max_legs: int = 4
    min_predicted_weight: float = 0.02


@dataclass(frozen=True)
class OptionWindowBuildConfig:
    variant: str = "long_short"
    top_k: int = 5
    entry_threshold: float = 0.5
    exit_threshold: float = 0.5
    top_family_count: int = 8
    ranking_framework: str | None = "zipline_shared_book_native"
    min_signal_events: int = 1
    min_ae_familiarity: float | None = None
    max_trades: int | None = None


@dataclass(frozen=True)
class OracleOptionExperimentConfig:
    experiment_name: str = "oracle_1t_option_retrieval_ranking"
    symbols: tuple[str, ...] = ("AAPL", "AMZN", "AVGO", "GOOG", "GOOGL", "META", "MSFT", "NVDA", "TSLA")
    price_start: str = "2018-01-01"
    price_end: str | None = "2026-06-24"
    k_params: dict[str, list[int]] = field(default_factory=lambda: {"YE": [1]})
    min_profit_pct: float = 0.01
    categorical_trade_context_columns: tuple[str, ...] = ()
    split: SharedSplitConfig = field(default_factory=SharedSplitConfig)
    retrieval: OptionRetrievalConfig = field(default_factory=OptionRetrievalConfig)
    execution: OptopsyExecutionConfig = field(default_factory=OptopsyExecutionConfig)
    mv_basket: OptionMvBasketConfig = field(default_factory=OptionMvBasketConfig)
    random_seed: int = 20260702
    option_execution_workers: int = 1
    log_mlflow: bool = True
    mlflow_experiment: str = "options_trading"
    mlflow_tracking_uri: str | None = None
    artifact_dir: str | None = None
    quant_warehouse_root: str | None = "/home/jlee153232/PycharmProjects/quant-warehouse"


@dataclass(frozen=True)
class OracleOptionExperimentResult:
    config: OracleOptionExperimentConfig
    mlflow_run_id: str | None
    coverage: pd.DataFrame
    oracle_trades: pd.DataFrame
    option_panel: pd.DataFrame
    train_panel: pd.DataFrame
    eval_panel: pd.DataFrame
    selector_summary: pd.DataFrame
    optopsy_summary: pd.DataFrame
    symbol_summary: pd.DataFrame
    source_family_summary: pd.DataFrame
    feature_coverage: pd.DataFrame
    selected_option_trades: pd.DataFrame
    selected_option_paths: pd.DataFrame
    metrics: dict[str, float]
    artifact_paths: dict[str, Path]
    analysis_markdown: str
    elapsed_seconds: float
    trade_status: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass(frozen=True)
class OptionWindowDataset:
    windows_by_group: dict[str, pd.DataFrame]
    source_groups: dict[str, tuple[str, ...]]
    window_summary: pd.DataFrame
    source_ranking: pd.DataFrame


@dataclass(frozen=True)
class OptionExperimentArtifacts:
    base_dir: Path
    coverage: pd.DataFrame
    trade_windows: pd.DataFrame
    option_panel: pd.DataFrame
    train_panel: pd.DataFrame
    eval_panel: pd.DataFrame
    selector_summary: pd.DataFrame
    optopsy_summary: pd.DataFrame
    symbol_summary: pd.DataFrame
    source_family_summary: pd.DataFrame
    feature_coverage: pd.DataFrame
    selected_option_trades: pd.DataFrame
    selected_option_paths: pd.DataFrame
    metrics: dict[str, Any]
    config: dict[str, Any]
    analysis_markdown: str


@dataclass(frozen=True)
class OptionRuntimeEstimate:
    baseline_elapsed_seconds: float
    baseline_trade_windows: int
    baseline_option_rows: int
    target_trade_windows: int
    target_option_rows: int
    estimated_seconds_by_windows: float
    estimated_seconds_by_option_rows: float
    estimated_seconds_conservative: float
    max_seconds: float
    runnable_within_budget: bool


def run_oracle_option_experiment(config: OracleOptionExperimentConfig) -> OracleOptionExperimentResult:
    started = perf_counter()
    _prepare_quant_warehouse_import(config.quant_warehouse_root)
    (
        Warehouse,
        LabelBuildSpec,
        build_trade_results,
        read_option_chain_arctic,
        option_chain_coverage,
        build_option_contract_features,
        option_ranker_feature_columns,
        build_option_mean_variance_labels,
    ) = _warehouse_imports()
    warehouse = Warehouse()
    coverage = option_chain_coverage(config.symbols)
    eligible_symbols = tuple(
        coverage.loc[coverage["snapshot_day_count"].fillna(0).gt(0), "symbol"].astype(str).str.upper().tolist()
    )
    price_frames = _load_price_frames(
        warehouse,
        eligible_symbols,
        start=config.price_start,
        end=config.price_end,
    )
    option_price_frames = _load_price_frames(
        warehouse,
        eligible_symbols,
        start=config.price_start,
        end=config.price_end,
        adjustment="unadjusted",
    )
    label_spec = LabelBuildSpec(
        k_params=config.k_params,
        min_profit_pct=config.min_profit_pct,
        buy_execution="high",
        sell_execution="low",
        short_execution="low",
        cover_execution="high",
        start_date=config.price_start,
        end_date=config.price_end,
    )
    trade_result = build_trade_results(eligible_symbols, spec=label_spec, price_frames=price_frames)
    oracle_trades = _normalize_oracle_trades(pd.DataFrame(trade_result.completed_trades))
    return _run_option_experiment_from_trades(
        config=config,
        started=started,
        coverage=coverage,
        trades=oracle_trades,
        price_frames=price_frames,
        option_price_frames=option_price_frames,
        read_option_chain_arctic=read_option_chain_arctic,
        build_option_contract_features=build_option_contract_features,
        option_ranker_feature_columns=option_ranker_feature_columns,
        build_option_mean_variance_labels=build_option_mean_variance_labels,
    )


def run_trade_window_option_experiment(
    config: OracleOptionExperimentConfig,
    trade_windows: pd.DataFrame,
) -> OracleOptionExperimentResult:
    """Run the option selector from externally generated equity trade windows."""

    started = perf_counter()
    _prepare_quant_warehouse_import(config.quant_warehouse_root)
    (
        Warehouse,
        _LabelBuildSpec,
        _build_trade_results,
        read_option_chain_arctic,
        option_chain_coverage,
        build_option_contract_features,
        option_ranker_feature_columns,
        build_option_mean_variance_labels,
    ) = _warehouse_imports()
    normalized_trades = _normalize_trade_windows(trade_windows)
    if normalized_trades.empty:
        raise ValueError("trade_windows did not contain any valid option-tradable entry/exit rows")
    symbols = tuple(sorted(set(normalized_trades["symbol"].astype(str).str.upper()).intersection(config.symbols or ())))
    if not symbols:
        symbols = tuple(sorted(normalized_trades["symbol"].astype(str).str.upper().unique().tolist()))
    coverage = option_chain_coverage(symbols)
    eligible_symbols = tuple(
        coverage.loc[coverage["snapshot_day_count"].fillna(0).gt(0), "symbol"].astype(str).str.upper().tolist()
    )
    normalized_trades = normalized_trades.loc[normalized_trades["symbol"].isin(eligible_symbols)].copy()
    warehouse = Warehouse()
    price_frames = _load_price_frames(
        warehouse,
        eligible_symbols,
        start=config.price_start,
        end=config.price_end,
    )
    option_price_frames = _load_price_frames(
        warehouse,
        eligible_symbols,
        start=config.price_start,
        end=config.price_end,
        adjustment="unadjusted",
    )
    return _run_option_experiment_from_trades(
        config=config,
        started=started,
        coverage=coverage,
        trades=normalized_trades,
        price_frames=price_frames,
        option_price_frames=option_price_frames,
        read_option_chain_arctic=read_option_chain_arctic,
        build_option_contract_features=build_option_contract_features,
        option_ranker_feature_columns=option_ranker_feature_columns,
        build_option_mean_variance_labels=build_option_mean_variance_labels,
    )


def run_backtest_report_option_experiment(
    config: OracleOptionExperimentConfig,
    report: NormalizedBacktestReport,
) -> OracleOptionExperimentResult:
    """Run option execution from any framework's normalized backtest report."""

    return run_trade_window_option_experiment(config, report_trade_windows(report))


def run_trade_window_option_execution(
    config: OracleOptionExperimentConfig,
    trade_windows: pd.DataFrame,
    *,
    selector_name: str = "rule_atm_90d",
) -> OracleOptionExperimentResult:
    """Execute option equivalents from equity trade windows without training candidate labels.

    This is the fast path for production-style option backtests. It loads the
    full option chain on each equity entry date, selects the contract on entry
    information only, then loads/prices only the selected contract forward to
    the equity exit/option expiration settlement date.
    """

    started = perf_counter()
    _prepare_quant_warehouse_import(config.quant_warehouse_root)
    (
        Warehouse,
        _LabelBuildSpec,
        _build_trade_results,
        read_option_chain_arctic,
        option_chain_coverage,
        build_option_contract_features,
        _option_ranker_feature_columns,
        _build_option_mean_variance_labels,
    ) = _warehouse_imports()
    normalized_trades = _normalize_trade_windows(trade_windows)
    if normalized_trades.empty:
        raise ValueError("trade_windows did not contain any valid option-tradable entry/exit rows")
    symbols = tuple(sorted(set(normalized_trades["symbol"].astype(str).str.upper()).intersection(config.symbols or ())))
    if not symbols:
        symbols = tuple(sorted(normalized_trades["symbol"].astype(str).str.upper().unique().tolist()))
    coverage = option_chain_coverage(symbols)
    eligible_symbols = tuple(
        coverage.loc[coverage["snapshot_day_count"].fillna(0).gt(0), "symbol"].astype(str).str.upper().tolist()
    )
    normalized_trades = normalized_trades.loc[normalized_trades["symbol"].isin(eligible_symbols)].copy()
    warehouse = Warehouse()
    price_frames = _load_price_frames(
        warehouse,
        eligible_symbols,
        start=config.price_start,
        end=config.price_end,
    )
    option_price_frames = _load_price_frames(
        warehouse,
        eligible_symbols,
        start=config.price_start,
        end=config.price_end,
        adjustment="unadjusted",
    )
    def retriever_factory() -> _OptionRetriever:
        return _OptionRetriever(
            config.retrieval,
            execution=config.execution,
            price_frames=price_frames,
            option_price_frames=option_price_frames,
            read_option_chain_arctic=read_option_chain_arctic,
            build_option_contract_features=build_option_contract_features,
        )

    phase_started = perf_counter()
    execution_batch = OptionTradeExecutor(
        retriever_factory,
        selector_name=selector_name,
        workers=max(1, int(config.option_execution_workers)),
    ).execute(normalized_trades)
    selected_option_trades = execution_batch.selected_option_trades
    selected_option_paths = execution_batch.selected_option_paths
    trade_status = execution_batch.trade_status
    metrics: dict[str, float] = {
        "phase_entry_selection_and_selected_pricing_seconds": float(perf_counter() - phase_started),
        "trade_windows": float(len(normalized_trades)),
        "selected_option_trades": float(len(selected_option_trades)),
        "selected_option_paths": float(len(selected_option_paths)),
        "option_execution_workers": float(max(1, int(config.option_execution_workers))),
    }
    status_counts = trade_status["status"].value_counts() if not trade_status.empty and "status" in trade_status else pd.Series(dtype=int)
    for status, count in status_counts.items():
        metrics[f"trade_status_{status}"] = float(count)
    selected_by_selector = {selector_name: selected_option_trades}
    selector_summary, symbol_summary = _summarize_selected_by_selector(selected_by_selector)
    optopsy_summary = _run_optopsy_selector_backtests(selected_by_selector, config.execution)
    feature_coverage = _option_feature_coverage(
        selected_option_trades,
        train_panel=pd.DataFrame(),
        eval_panel=selected_option_trades,
    )
    metrics.update(execution_batch.metrics)
    metrics.update(_option_feature_coverage_metrics(feature_coverage))
    elapsed_seconds = perf_counter() - started
    metrics["elapsed_seconds"] = float(elapsed_seconds)
    analysis_markdown = _build_execution_analysis(
        config,
        trade_windows=normalized_trades,
        selected_option_trades=selected_option_trades,
        selector_summary=selector_summary,
        optopsy_summary=optopsy_summary,
        trade_status=trade_status,
        metrics=metrics,
    )
    artifact_paths = write_oracle_option_artifacts(
        config=config,
        coverage=coverage,
        oracle_trades=normalized_trades,
        option_panel=selected_option_trades,
        train_panel=pd.DataFrame(),
        eval_panel=selected_option_trades,
        selector_summary=selector_summary,
        optopsy_summary=optopsy_summary,
        symbol_summary=symbol_summary,
        source_family_summary=pd.DataFrame(),
        feature_coverage=feature_coverage,
        selected_option_trades=selected_option_trades,
        selected_option_paths=selected_option_paths,
        model=None,
        metrics=metrics,
        analysis_markdown=analysis_markdown,
        directory=Path(config.artifact_dir) if config.artifact_dir else None,
    )
    status_path = artifact_paths["metrics"].with_name("trade_status.csv")
    trade_status.to_csv(status_path, index=False)
    artifact_paths["trade_status"] = status_path
    mlflow_run_id = None
    if config.log_mlflow:
        mlflow_run_id = _log_mlflow(config, artifact_paths, metrics)
    return OracleOptionExperimentResult(
        config=config,
        mlflow_run_id=mlflow_run_id,
        coverage=coverage,
        oracle_trades=normalized_trades,
        option_panel=selected_option_trades,
        train_panel=pd.DataFrame(),
        eval_panel=selected_option_trades,
        selector_summary=selector_summary,
        optopsy_summary=optopsy_summary,
        symbol_summary=symbol_summary,
        source_family_summary=pd.DataFrame(),
        feature_coverage=feature_coverage,
        selected_option_trades=selected_option_trades,
        selected_option_paths=selected_option_paths,
        metrics=metrics,
        artifact_paths=artifact_paths,
        analysis_markdown=analysis_markdown,
        elapsed_seconds=elapsed_seconds,
        trade_status=trade_status,
    )


def run_backtest_report_option_execution(
    config: OracleOptionExperimentConfig,
    report: NormalizedBacktestReport,
    *,
    selector_name: str = "rule_atm_90d",
) -> OracleOptionExperimentResult:
    """Fast option execution from any framework's normalized backtest report."""

    return run_trade_window_option_execution(config, report_trade_windows(report), selector_name=selector_name)


def build_classifier_signal_trade_windows(
    strategy_scores: pd.DataFrame,
    *,
    strategy_sources: tuple[str, ...] | None = ("ensemble_mean",),
    variant: str = "long_short",
    top_k: int = 5,
    entry_threshold: float = 0.5,
    exit_threshold: float = 0.5,
    min_ae_familiarity: float | None = None,
    max_trades: int | None = None,
) -> pd.DataFrame:
    """Convert classifier score rows into option-tradable equity windows.

    The input must be the full scored trading calendar for the strategy
    universe. Models are trained on events, but the strategy must be scored and
    replayed on every trading day so exits and new entries are not skipped.
    """

    if strategy_scores.empty:
        return pd.DataFrame()
    if variant not in {"long_only", "short_only", "long_short"}:
        raise ValueError(f"unknown variant {variant!r}")
    if int(top_k) <= 0:
        raise ValueError("top_k must be positive")
    required = {"strategy_source", "source", "family", "symbol", "date", "long_score", "short_score", "long_exit_score", "short_exit_score"}
    missing = required - set(strategy_scores.columns)
    if missing:
        raise KeyError(f"strategy_scores missing required columns: {sorted(missing)}")

    scores = strategy_scores.copy()
    scores["symbol"] = scores["symbol"].astype(str).str.upper()
    scores["date"] = pd.to_datetime(scores["date"], errors="coerce").dt.normalize()
    scores = scores.dropna(subset=["date", "symbol"])
    if strategy_sources is not None:
        wanted = {str(source) for source in strategy_sources}
        scores = scores.loc[scores["strategy_source"].astype(str).isin(wanted)].copy()
    if scores.empty:
        return pd.DataFrame()
    scores = _planner_score_frame_for_sources(scores, strategy_sources=strategy_sources)

    rows: list[dict[str, Any]] = []
    grouped = scores.sort_values(["strategy_source", "date", "symbol"]).groupby("strategy_source", sort=True)
    for strategy_source, source_scores in grouped:
        source_value = str(source_scores["source"].iloc[0])
        family_value = str(source_scores["family"].iloc[0])
        by_date = {date: frame.set_index("symbol", drop=False) for date, frame in source_scores.groupby("date", sort=True)}
        positions: dict[str, dict[str, Any]] = {}
        for date in sorted(by_date):
            day = by_date[date]
            for symbol, position in list(positions.items()):
                if symbol not in day.index:
                    continue
                row = day.loc[symbol]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                long_exit = _nan_float(row["long_exit_score"])
                short_exit = _nan_float(row["short_exit_score"])
                if {"long_agree_count", "short_agree_count", "model_count"}.issubset(day.columns):
                    model_count = _nan_float(row["model_count"])
                    long_agree = _nan_float(row["long_agree_count"])
                    short_agree = _nan_float(row["short_agree_count"])
                    should_exit = (
                        position["side"] == "long"
                        and pd.notna(model_count)
                        and model_count > 0
                        and long_agree < model_count
                    ) or (
                        position["side"] == "short"
                        and pd.notna(model_count)
                        and model_count > 0
                        and short_agree < model_count
                    )
                else:
                    should_exit = (
                        position["side"] == "long"
                        and pd.notna(short_exit)
                        and pd.notna(long_exit)
                        and short_exit > long_exit
                    ) or (
                        position["side"] == "short"
                        and pd.notna(long_exit)
                        and pd.notna(short_exit)
                        and long_exit >= short_exit
                    )
                if should_exit:
                    rows.append(_classifier_trade_window_row(position, exit_date=pd.Timestamp(date), exit_score=short_exit if position["side"] == "long" else long_exit))
                    del positions[symbol]
            open_slots = max(0, int(top_k) - len(positions))
            if open_slots <= 0:
                continue
            candidates = _classifier_entry_candidates(
                day,
                held_symbols=set(positions),
                variant=variant,
                entry_threshold=float(entry_threshold),
                min_ae_familiarity=min_ae_familiarity,
            )
            for candidate in candidates[:open_slots]:
                symbol = candidate["symbol"]
                positions[symbol] = {
                    **candidate,
                    "entry_date": pd.Timestamp(date),
                    "strategy_source": str(strategy_source),
                    "source": source_value,
                    "family": family_value,
                    "variant": variant,
                    "top_k": int(top_k),
                }
        if by_date:
            final_date = pd.Timestamp(max(by_date))
            for position in positions.values():
                rows.append(_classifier_trade_window_row(position, exit_date=final_date, exit_score=np.nan))

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.loc[pd.to_datetime(out["exit_date"]).gt(pd.to_datetime(out["entry_date"]))].copy()
    out = out.sort_values(["entry_date", "strategy_source", "symbol", "side"]).reset_index(drop=True)
    if max_trades is not None:
        out = out.head(int(max_trades)).copy()
    out["trade_id"] = out.apply(
        lambda row: (
            f"{row['strategy_source']}|{row['variant']}|{row['symbol']}|"
            f"{pd.Timestamp(row['entry_date']).date()}|{row['side']}"
        ),
        axis=1,
    )
    return out.reset_index(drop=True)


def build_option_window_dataset(
    strategy_scores: pd.DataFrame,
    *,
    backtest_summary: pd.DataFrame | None = None,
    config: OptionWindowBuildConfig = OptionWindowBuildConfig(),
    include_groups: tuple[str, ...] = ("individual_feature_families", "all_feature_families"),
) -> OptionWindowDataset:
    """Build comparable option-trade-window datasets from classifier scores."""

    if strategy_scores.empty:
        empty_summary = pd.DataFrame(columns=["group", "sources", "windows", "symbols", "min_entry_date", "max_entry_date"])
        return OptionWindowDataset({}, {}, empty_summary, pd.DataFrame())
    all_sources = tuple(sorted(src for src in strategy_scores["strategy_source"].dropna().astype(str).unique() if src != "ensemble_mean"))
    ranking = (
        rank_option_window_strategy_sources(
            backtest_summary,
            variant=config.variant,
            top_k=config.top_k,
            framework=config.ranking_framework,
            min_signal_events=config.min_signal_events,
        )
        if backtest_summary is not None and not backtest_summary.empty
        else pd.DataFrame()
    )
    top_sources = tuple(ranking["strategy_source"].head(config.top_family_count).astype(str).tolist()) if not ranking.empty else all_sources[: config.top_family_count]
    candidate_groups = {"all_feature_families": all_sources}
    if "ensemble_mean" in include_groups:
        candidate_groups["ensemble_mean"] = ("ensemble_mean",)
    if "top_feature_families" in include_groups:
        candidate_groups["top_feature_families"] = top_sources
    if "individual_feature_families" in include_groups:
        for source in all_sources:
            candidate_groups[f"individual__{source}"] = (source,)
    source_groups = {
        name: sources
        for name, sources in candidate_groups.items()
        if name in include_groups or (name.startswith("individual__") and "individual_feature_families" in include_groups)
    }
    windows_by_group: dict[str, pd.DataFrame] = {}
    summary_rows = []
    for name, sources in source_groups.items():
        windows = build_classifier_signal_trade_windows(
            strategy_scores,
            strategy_sources=sources,
            variant=config.variant,
            top_k=config.top_k,
            entry_threshold=config.entry_threshold,
            exit_threshold=config.exit_threshold,
            min_ae_familiarity=config.min_ae_familiarity,
            max_trades=config.max_trades,
        )
        windows_by_group[name] = windows
        summary_rows.append(
            {
                "group": name,
                "sources": len(sources),
                "windows": len(windows),
                "symbols": int(windows["symbol"].nunique()) if not windows.empty else 0,
                "min_entry_date": windows["entry_date"].min() if not windows.empty else pd.NaT,
                "max_entry_date": windows["entry_date"].max() if not windows.empty else pd.NaT,
            }
        )
    return OptionWindowDataset(
        windows_by_group=windows_by_group,
        source_groups=source_groups,
        window_summary=pd.DataFrame(summary_rows),
        source_ranking=ranking,
    )


def rank_option_window_strategy_sources(
    backtest_summary: pd.DataFrame,
    *,
    variant: str = "long_short",
    top_k: int = 5,
    framework: str | None = None,
    exclude_ensemble: bool = True,
    min_signal_events: int = 1,
) -> pd.DataFrame:
    """Rank equity strategy sources before converting them into option windows."""

    if backtest_summary.empty:
        return pd.DataFrame()
    required = {"strategy_source", "source", "family", "variant", "top_k", "signal_events", "sharpe", "total_return", "max_drawdown"}
    missing = required - set(backtest_summary.columns)
    if missing:
        raise KeyError(f"backtest_summary missing required columns: {sorted(missing)}")
    out = backtest_summary.copy()
    out = out.loc[out["variant"].astype(str).eq(str(variant)) & pd.to_numeric(out["top_k"], errors="coerce").eq(int(top_k))].copy()
    if framework is not None:
        out = out.loc[out["framework"].astype(str).eq(str(framework))].copy()
    if exclude_ensemble:
        out = out.loc[out["strategy_source"].astype(str).ne("ensemble_mean")].copy()
    out["signal_events"] = pd.to_numeric(out["signal_events"], errors="coerce").fillna(0)
    out = out.loc[out["signal_events"].ge(int(min_signal_events))].copy()
    for col in ("sharpe", "total_return", "max_drawdown"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.sort_values(["sharpe", "total_return"], ascending=[False, False]).reset_index(drop=True)


def _classifier_entry_candidates(
    day: pd.DataFrame,
    *,
    held_symbols: set[str],
    variant: str,
    entry_threshold: float,
    min_ae_familiarity: float | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in day.reset_index(drop=True).itertuples(index=False):
        symbol = str(getattr(raw, "symbol")).upper()
        if symbol in held_symbols:
            continue
        ae_familiarity = _nan_float(getattr(raw, "ae_familiarity", np.nan))
        if min_ae_familiarity is not None and (pd.isna(ae_familiarity) or ae_familiarity < float(min_ae_familiarity)):
            continue
        long_score = _nan_float(getattr(raw, "long_score", np.nan))
        short_score = _nan_float(getattr(raw, "short_score", np.nan))
        long_direction = _nan_float(getattr(raw, "long_exit_score", long_score))
        short_direction = _nan_float(getattr(raw, "short_exit_score", short_score))
        classifier_long = pd.notna(long_direction) and pd.notna(short_direction) and long_direction >= short_direction
        classifier_short = pd.notna(long_direction) and pd.notna(short_direction) and short_direction > long_direction
        model_count = _nan_float(getattr(raw, "model_count", np.nan))
        long_agree = _nan_float(getattr(raw, "long_agree_count", np.nan))
        short_agree = _nan_float(getattr(raw, "short_agree_count", np.nan))
        if pd.notna(model_count) and model_count > 0 and pd.notna(long_agree) and pd.notna(short_agree):
            classifier_long = bool(long_agree == model_count)
            classifier_short = bool(short_agree == model_count)
        base = {
            "symbol": symbol,
            "ae_familiarity": ae_familiarity,
            "ae_recon_error": _nan_float(getattr(raw, "ae_recon_error", np.nan)),
            "ae_latent_distance": _nan_float(getattr(raw, "ae_latent_distance", np.nan)),
        }
        if variant in {"long_only", "long_short"} and classifier_long and _entry_score_ok(long_score, entry_threshold):
            rows.append({**base, "side": "long", "entry_score": long_score, "opposite_score": short_score})
        if variant in {"short_only", "long_short"} and classifier_short and _entry_score_ok(short_score, entry_threshold):
            rows.append({**base, "side": "short", "entry_score": short_score, "opposite_score": long_score})
    return sorted(rows, key=lambda row: (row["entry_score"], -row.get("opposite_score", 0.0)), reverse=True)


def _planner_score_frame_for_sources(scores: pd.DataFrame, *, strategy_sources: tuple[str, ...] | None) -> pd.DataFrame:
    sources = tuple(scores["strategy_source"].dropna().astype(str).unique().tolist())
    if len(sources) <= 1:
        return scores.copy()
    numeric_aggs = {
        out_col: (in_col, "mean")
        for out_col, in_col in {
            "long_score": "long_score",
            "short_score": "short_score",
            "long_exit_score": "long_exit_score",
            "short_exit_score": "short_exit_score",
            "ae_familiarity": "ae_familiarity",
            "ae_recon_error": "ae_recon_error",
            "ae_latent_distance": "ae_latent_distance",
        }.items()
        if in_col in scores.columns
    }
    optional = {
        "classifier_long_score": ("classifier_long_score", "mean"),
        "classifier_short_score": ("classifier_short_score", "mean"),
        "long_agree_count": ("long_agree_count", "sum"),
        "short_agree_count": ("short_agree_count", "sum"),
        "model_count": ("strategy_source", "nunique"),
    }
    for col, agg in optional.items():
        if agg[0] in scores.columns:
            numeric_aggs[col] = agg
    out = scores.groupby(["symbol", "date"], as_index=False).agg(**numeric_aggs)
    label = "ensemble_mean" if strategy_sources is None else "ensemble_" + str(len(sources)) + "_sources"
    out["source"] = "ensemble"
    out["family"] = "mean"
    out["strategy_source"] = label
    out["net_score"] = out["long_score"] - out["short_score"]
    return out


def _entry_score_ok(value: float, threshold: float) -> bool:
    return bool(pd.notna(value) and np.isfinite(float(value)) and float(value) > float(threshold))


def _option_action_for_signal(side: str, option_type: str) -> str | None:
    side = str(side).lower()
    option_type = "call" if str(option_type).lower().startswith("c") else "put"
    if side == "long" and option_type == "call":
        return "buy_call"
    if side == "long" and option_type == "put":
        return "sell_put"
    if side == "short" and option_type == "call":
        return "sell_call"
    if side == "short" and option_type == "put":
        return "buy_put"
    return None


def _classifier_trade_window_row(position: dict[str, Any], *, exit_date: pd.Timestamp, exit_score: float) -> dict[str, Any]:
    entry_date = pd.Timestamp(position["entry_date"]).normalize()
    exit_date = pd.Timestamp(exit_date).normalize()
    planned_holding_days = max(0, int((exit_date - entry_date).days))
    return {
        "symbol": str(position["symbol"]).upper(),
        "side": str(position["side"]),
        "entry_date": entry_date,
        "exit_date": exit_date,
        "strategy_source": str(position["strategy_source"]),
        "source": str(position["source"]),
        "family": str(position["family"]),
        "source_family": f"{position['source']}.{position['family']}",
        "variant": str(position["variant"]),
        "top_k": int(position["top_k"]),
        "equity_signal_score": _nan_float(position.get("entry_score")),
        "equity_exit_signal_score": _nan_float(exit_score),
        "planned_holding_days": planned_holding_days,
        "ae_familiarity": _nan_float(position.get("ae_familiarity")),
        "ae_recon_error": _nan_float(position.get("ae_recon_error")),
        "ae_latent_distance": _nan_float(position.get("ae_latent_distance")),
    }


def _run_option_experiment_from_trades(
    *,
    config: OracleOptionExperimentConfig,
    started: float,
    coverage: pd.DataFrame,
    trades: pd.DataFrame,
    price_frames: dict[str, pd.DataFrame],
    option_price_frames: dict[str, pd.DataFrame] | None = None,
    read_option_chain_arctic,
    build_option_contract_features,
    option_ranker_feature_columns,
    build_option_mean_variance_labels,
) -> OracleOptionExperimentResult:
    oracle_trades = _normalize_trade_windows(trades)
    retriever = _OptionRetriever(
        config.retrieval,
        execution=config.execution,
        price_frames=price_frames,
        option_price_frames=option_price_frames,
        read_option_chain_arctic=read_option_chain_arctic,
        build_option_contract_features=build_option_contract_features,
    )
    phase_metrics: dict[str, float] = {}
    phase_started = perf_counter()
    if config.retrieval.price_eval_candidates:
        option_panel = _build_option_panel(oracle_trades, retriever, price_exit=True)
        phase_metrics["phase_option_panel_seconds"] = float(perf_counter() - phase_started)
        phase_started = perf_counter()
        if config.mv_basket.enabled:
            option_panel = _add_mean_variance_basket_labels(
                option_panel,
                config.mv_basket,
                build_option_mean_variance_labels=build_option_mean_variance_labels,
            )
        phase_metrics["phase_mv_labels_seconds"] = float(perf_counter() - phase_started)
        phase_started = perf_counter()
        train_panel, eval_panel = _split_option_panel(option_panel, config.split)
    else:
        train_trades, eval_trades = _split_trade_windows(oracle_trades, config.split)
        train_panel = _build_option_panel(train_trades, retriever, price_exit=True)
        eval_panel = _build_option_panel(eval_trades, retriever, price_exit=False)
        phase_metrics["phase_option_panel_seconds"] = float(perf_counter() - phase_started)
        phase_started = perf_counter()
        if config.mv_basket.enabled:
            train_panel = _add_mean_variance_basket_labels(
                train_panel,
                config.mv_basket,
                build_option_mean_variance_labels=build_option_mean_variance_labels,
            )
        phase_metrics["phase_mv_labels_seconds"] = float(perf_counter() - phase_started)
        phase_started = perf_counter()
        option_panel = pd.concat([train_panel, eval_panel], ignore_index=True) if not eval_panel.empty else train_panel.copy()
    phase_metrics["phase_split_seconds"] = float(perf_counter() - phase_started)
    phase_started = perf_counter()
    model, model_metrics, eval_scored = _train_option_ranker(
        train_panel,
        eval_panel,
        config=config,
        option_ranker_feature_columns=option_ranker_feature_columns,
    )
    phase_metrics["phase_return_ranker_seconds"] = float(perf_counter() - phase_started)
    phase_started = perf_counter()
    mv_model, mv_metrics, eval_scored = _train_mv_basket_ranker(
        train_panel,
        eval_scored,
        config=config,
        option_ranker_feature_columns=option_ranker_feature_columns,
    )
    phase_metrics["phase_mv_ranker_seconds"] = float(perf_counter() - phase_started)
    model_payload = {"single_leg_ranker": model, "mv_basket_ranker": mv_model}
    model_metrics.update(phase_metrics)
    model_metrics.update(mv_metrics)
    phase_started = perf_counter()
    selected_by_selector = _select_options_by_selector(eval_scored, mv_basket_config=config.mv_basket)
    selected_by_selector, selected_option_paths = _price_selected_by_selector(selected_by_selector, retriever)
    selected_option_trades = _selected_by_selector_to_frame(selected_by_selector)
    selector_summary, symbol_summary = _summarize_selected_by_selector(selected_by_selector)
    source_family_summary = _source_family_diagnostics(eval_scored, selected_by_selector)
    feature_coverage = _option_feature_coverage(option_panel, train_panel=train_panel, eval_panel=eval_scored)
    model_metrics.update(retriever.metrics())
    model_metrics.update(_option_feature_coverage_metrics(feature_coverage))
    model_metrics["phase_summaries_seconds"] = float(perf_counter() - phase_started)
    phase_started = perf_counter()
    optopsy_summary = _run_optopsy_selector_backtests(selected_by_selector, config.execution)
    model_metrics["phase_optopsy_seconds"] = float(perf_counter() - phase_started)
    elapsed_seconds = perf_counter() - started
    model_metrics["elapsed_seconds"] = float(elapsed_seconds)
    phase_started = perf_counter()
    analysis_markdown = _build_analysis(
        config,
        oracle_trades=oracle_trades,
        option_panel=option_panel,
        train_panel=train_panel,
        eval_panel=eval_scored,
        selector_summary=selector_summary,
        optopsy_summary=optopsy_summary,
        source_family_summary=source_family_summary,
        feature_coverage=feature_coverage,
        metrics=model_metrics,
    )
    model_metrics["phase_analysis_seconds"] = float(perf_counter() - phase_started)
    artifact_paths = write_oracle_option_artifacts(
        config=config,
        coverage=coverage,
        oracle_trades=oracle_trades,
        option_panel=option_panel,
        train_panel=train_panel,
        eval_panel=eval_scored,
        selector_summary=selector_summary,
        optopsy_summary=optopsy_summary,
        symbol_summary=symbol_summary,
        source_family_summary=source_family_summary,
        feature_coverage=feature_coverage,
        selected_option_trades=selected_option_trades,
        selected_option_paths=selected_option_paths,
        model=model_payload,
        metrics=model_metrics,
        analysis_markdown=analysis_markdown,
        directory=Path(config.artifact_dir) if config.artifact_dir else None,
    )
    mlflow_run_id = None
    if config.log_mlflow:
        mlflow_run_id = _log_mlflow(config, artifact_paths, model_metrics)
    return OracleOptionExperimentResult(
        config=config,
        mlflow_run_id=mlflow_run_id,
        coverage=coverage,
        oracle_trades=oracle_trades,
        option_panel=option_panel,
        train_panel=train_panel,
        eval_panel=eval_scored,
        selector_summary=selector_summary,
        optopsy_summary=optopsy_summary,
        symbol_summary=symbol_summary,
        source_family_summary=source_family_summary,
        feature_coverage=feature_coverage,
        selected_option_trades=selected_option_trades,
        selected_option_paths=selected_option_paths,
        metrics=model_metrics,
        artifact_paths=artifact_paths,
        analysis_markdown=analysis_markdown,
        elapsed_seconds=elapsed_seconds,
    )


class _OptionRetriever:
    def __init__(
        self,
        config: OptionRetrievalConfig,
        *,
        execution: OptopsyExecutionConfig | None = None,
        price_frames: dict[str, pd.DataFrame],
        option_price_frames: dict[str, pd.DataFrame] | None = None,
        read_option_chain_arctic,
        build_option_contract_features,
    ):
        self.config = config
        self.execution = execution or OptopsyExecutionConfig()
        self.price_frames = price_frames
        self.option_price_frames = option_price_frames or {}
        self.read_option_chain_arctic = read_option_chain_arctic
        self.build_option_contract_features = build_option_contract_features
        self._chain_cache: dict[tuple[str, pd.Timestamp], pd.DataFrame] = {}
        self._raw_chain_cache: dict[tuple[str, pd.Timestamp], pd.DataFrame] = {}
        self._underlying_cache: dict[tuple[str, pd.Timestamp], float | None] = {}
        self._nearest_exit_snapshot_cache: dict[tuple[str, pd.Timestamp], pd.Timestamp | None] = {}
        self._chain_quote_cache: dict[tuple[str, pd.Timestamp], dict[str, tuple[float, float, float]]] = {}
        self._contract_path_cache: dict[tuple[str, str, pd.Timestamp, pd.Timestamp], pd.DataFrame] = {}
        self.chain_read_count = 0
        self.chain_cache_hits = 0
        self.raw_chain_cache_hits = 0
        self.nearest_exit_cache_hits = 0
        self.chain_quote_cache_hits = 0
        self.quote_chain_read_count = 0
        self.quote_chain_duplicate_rows_dropped = 0
        self.contract_path_read_count = 0
        self.contract_path_cache_hits = 0
        self.chain_duplicate_rows_dropped = 0
        self.underlying_lookup_count = 0
        self.underlying_cache_hits = 0
        self.underlying_adjusted_fallback_count = 0
        self.affordability_checked_count = 0
        self.affordability_filtered_count = 0

    def retrieve(self, trade: pd.Series, *, price_exit: bool = True) -> pd.DataFrame:
        symbol = str(trade["symbol"]).upper()
        side = str(trade["side"]).lower()
        entry_date = pd.Timestamp(trade["entry_date"]).normalize()
        equity_exit_date = pd.Timestamp(trade["exit_date"]).normalize()
        realized_holding_days = max(0, int((equity_exit_date - entry_date).days))
        entry_chain = self._load_day_chain(symbol, entry_date)
        spot = self._chain_underlying_price(entry_chain)
        if spot is None:
            spot = self._underlying_close(symbol, entry_date)
        if entry_chain.empty or spot is None or spot <= 0:
            return pd.DataFrame()
        if self.config.option_universe == "full_chain_actions":
            return self._retrieve_full_chain_actions(
                trade=trade,
                symbol=symbol,
                side=side,
                entry_date=entry_date,
                equity_exit_date=equity_exit_date,
                realized_holding_days=realized_holding_days,
                spot=float(spot),
                entry_chain=entry_chain,
                price_exit=price_exit,
            )
        option_type = "call" if side == "long" else "put"
        candidates = entry_chain.loc[entry_chain["option_type"].astype(str).str.lower().str.startswith(option_type[0])].copy()
        candidates["underlying_spot_entry"] = spot
        candidates["moneyness"] = candidates["strike"] / spot - 1.0
        candidates["abs_moneyness"] = candidates["moneyness"].abs()
        cfg = self.config
        candidates = candidates.loc[
            candidates["dte"].between(cfg.min_dte, cfg.max_dte)
            & candidates["mid"].ge(cfg.min_entry_mid)
            & candidates["spread_pct"].le(cfg.max_entry_spread_pct)
            & candidates["abs_moneyness"].le(cfg.max_abs_moneyness)
        ].copy()
        if candidates.empty:
            return candidates
        candidates = self._filter_affordable_candidates(candidates, trade)
        if candidates.empty:
            return candidates
        candidates["dte_gap"] = (candidates["dte"] - cfg.target_dte).abs()
        candidates["liquidity_score"] = (
            candidates.get("volume", pd.Series(0, index=candidates.index)).fillna(0)
            + candidates.get("open_interest", pd.Series(0, index=candidates.index)).fillna(0) / 100.0
        )
        candidates = candidates.sort_values(
            ["dte_gap", "abs_moneyness", "spread_pct", "liquidity_score"],
            ascending=[True, True, True, False],
        ).head(cfg.max_candidates_per_trade)
        candidates = self.build_option_contract_features(
            candidates,
            underlying_price=spot,
            target_dte=cfg.target_dte,
            compute_model_greeks=True,
        ).df
        rows = []
        for row in candidates.itertuples(index=False):
            entry_mid = float(row.mid)
            entry_price = float(getattr(row, "ask", entry_mid))
            row_payload = {
                    "trade_id": trade.get("trade_id", f"{symbol}|{entry_date.date()}|{side}"),
                    "symbol": symbol,
                    "side": side,
                    "equity_signal_side": side,
                    "entry_date": entry_date,
                    "equity_exit_date": equity_exit_date,
                    "option_exit_date": pd.NaT,
                    "expiration": pd.Timestamp(row.expiration).normalize(),
                    "contract_symbol": row.contract_symbol,
                    "option_type": option_type,
                    "option_action": "buy_call" if option_type == "call" else "buy_put",
                    "strike": float(row.strike),
                    "entry_mid": entry_mid,
                    "exit_mid": np.nan,
                    "entry_price": entry_price,
                    "exit_price": np.nan,
                    "option_pnl": np.nan,
                    "return_denominator": entry_price,
                    "option_return": np.nan,
                    "allocation_fraction": _nan_float(getattr(row, "allocation_fraction", np.nan)),
                    "allocation_budget": _nan_float(getattr(row, "allocation_budget", np.nan)),
                    "affordable_contracts": _nan_float(getattr(row, "affordable_contracts", np.nan)),
                    "realized_holding_days": realized_holding_days,
                    "realized_underlying_trade_return": pd.to_numeric(trade.get("ret_dec"), errors="coerce"),
                }
            for source_col, target_col in (
                ("planned_holding_days", "planned_holding_days"),
                ("equity_signal_score", "equity_signal_score"),
                ("equity_signal_rank", "equity_signal_rank"),
            ):
                value = trade.get(source_col)
                if value is not None and not pd.isna(value):
                    row_payload[target_col] = _nan_float(value)
            for col in ("source_family", "strategy_source", "source", "family"):
                value = trade.get(col)
                if value is not None and not pd.isna(value):
                    row_payload[col] = str(value)
            row_payload.update(_row_feature_payload(row))
            if price_exit:
                priced = self._price_option_payload(row_payload)
                if priced is None:
                    continue
                row_payload = priced
            rows.append(row_payload)
        return pd.DataFrame(rows)

    def _retrieve_full_chain_actions(
        self,
        *,
        trade: pd.Series,
        symbol: str,
        side: str,
        entry_date: pd.Timestamp,
        equity_exit_date: pd.Timestamp,
        realized_holding_days: int,
        spot: float,
        entry_chain: pd.DataFrame,
        price_exit: bool = True,
    ) -> pd.DataFrame:
        candidates = entry_chain.copy()
        candidates["option_type"] = candidates["option_type"].astype(str).str.lower().str.strip()
        candidates = candidates.loc[candidates["option_type"].str.startswith(("c", "p"))].copy()
        if candidates.empty:
            return candidates
        candidates["option_type"] = np.where(candidates["option_type"].str.startswith("c"), "call", "put")
        candidates["underlying_spot_entry"] = float(spot)
        candidates["moneyness"] = pd.to_numeric(candidates["strike"], errors="coerce") / float(spot) - 1.0
        candidates["abs_moneyness"] = candidates["moneyness"].abs()
        for col in ("bid", "ask", "mid", "strike"):
            candidates[col] = pd.to_numeric(candidates[col], errors="coerce")
        if "dte" not in candidates.columns:
            candidates["dte"] = (
                pd.to_datetime(candidates["expiration"], errors="coerce").dt.normalize() - entry_date
            ).dt.days
        candidates = candidates.loc[
            candidates["contract_symbol"].astype(str).ne("")
            & candidates["expiration"].notna()
            & candidates["strike"].gt(0)
            & candidates["bid"].ge(0)
            & candidates["ask"].gt(0)
            & candidates["ask"].ge(candidates["bid"])
            & pd.to_numeric(candidates["dte"], errors="coerce").gt(0)
        ].copy()
        if candidates.empty:
            return candidates
        cfg = self.config
        candidates["dte_gap"] = (pd.to_numeric(candidates["dte"], errors="coerce") - cfg.target_dte).abs()
        candidates["liquidity_score"] = (
            candidates.get("volume", pd.Series(0, index=candidates.index)).fillna(0)
            + candidates.get("open_interest", pd.Series(0, index=candidates.index)).fillna(0) / 100.0
        )
        candidates = self.build_option_contract_features(
            candidates,
            underlying_price=spot,
            target_dte=cfg.target_dte,
            compute_model_greeks=True,
        ).df
        rows = []
        for row in candidates.itertuples(index=False):
            option_type = "call" if str(row.option_type).lower().startswith("c") else "put"
            option_action = _option_action_for_signal(side, option_type)
            if option_action is None:
                continue
            entry_bid = _nan_float(getattr(row, "bid", np.nan))
            entry_ask = _nan_float(getattr(row, "ask", np.nan))
            if option_action.startswith("buy_"):
                entry_price = entry_ask
                denominator = entry_price
            else:
                entry_price = entry_bid
                denominator = float(row.strike) if option_action == "sell_put" else float(spot)
            if not np.isfinite(entry_price) or not np.isfinite(denominator) or denominator <= 0:
                continue
            row_payload = {
                "trade_id": trade.get("trade_id", f"{symbol}|{entry_date.date()}|{side}"),
                "symbol": symbol,
                "side": side,
                "equity_signal_side": side,
                "entry_date": entry_date,
                "equity_exit_date": equity_exit_date,
                "option_exit_date": pd.NaT,
                "expiration": pd.Timestamp(row.expiration).normalize(),
                "contract_symbol": row.contract_symbol,
                "option_type": option_type,
                "option_action": option_action,
                "strike": float(row.strike),
                "entry_mid": _nan_float(getattr(row, "mid", np.nan)),
                "exit_mid": np.nan,
                "entry_bid": entry_bid,
                "entry_ask": entry_ask,
                "exit_bid": np.nan,
                "exit_ask": np.nan,
                "entry_price": float(entry_price),
                "exit_price": np.nan,
                "option_pnl": np.nan,
                "return_denominator": float(denominator),
                "option_return": np.nan,
                "realized_holding_days": realized_holding_days,
                "realized_underlying_trade_return": pd.to_numeric(trade.get("ret_dec"), errors="coerce"),
            }
            for source_col, target_col in (
                ("planned_holding_days", "planned_holding_days"),
                ("equity_signal_score", "equity_signal_score"),
                ("equity_signal_rank", "equity_signal_rank"),
            ):
                value = trade.get(source_col)
                if value is not None and not pd.isna(value):
                    row_payload[target_col] = _nan_float(value)
            for col in ("source_family", "strategy_source", "source", "family", "top_k", "variant"):
                value = trade.get(col)
                if value is not None and not pd.isna(value):
                    row_payload[col] = value if col == "top_k" else str(value)
            row_payload.update(_row_feature_payload(row))
            if price_exit:
                priced = self._price_option_payload(row_payload)
                if priced is None:
                    continue
                row_payload = priced
            rows.append(row_payload)
        return pd.DataFrame(rows)

    def select_rule_atm_entry(self, trade: pd.Series) -> pd.DataFrame:
        symbol = str(trade["symbol"]).upper()
        side = str(trade["side"]).lower()
        entry_date = pd.Timestamp(trade["entry_date"]).normalize()
        equity_exit_date = pd.Timestamp(trade["exit_date"]).normalize()
        realized_holding_days = max(0, int((equity_exit_date - entry_date).days))
        entry_chain = self._load_day_chain_raw(symbol, entry_date)
        spot = self._chain_underlying_price(entry_chain)
        if spot is None:
            spot = self._underlying_close(symbol, entry_date)
        if entry_chain.empty or spot is None or spot <= 0:
            return pd.DataFrame()
        selected = self._select_rule_atm_from_native_chain(
            trade=trade,
            symbol=symbol,
            side=side,
            entry_date=entry_date,
            equity_exit_date=equity_exit_date,
            realized_holding_days=realized_holding_days,
            spot=float(spot),
            entry_chain=entry_chain,
        )
        if selected.empty:
            return selected
        featured = self.build_option_contract_features(
            selected,
            underlying_price=float(spot),
            target_dte=self.config.target_dte,
            compute_model_greeks=True,
        ).df
        if featured.empty:
            return selected
        return featured.reset_index(drop=True)

    def _select_rule_atm_from_native_chain(
        self,
        *,
        trade: pd.Series,
        symbol: str,
        side: str,
        entry_date: pd.Timestamp,
        equity_exit_date: pd.Timestamp,
        realized_holding_days: int,
        spot: float,
        entry_chain: pd.DataFrame,
    ) -> pd.DataFrame:
        candidates = entry_chain.copy()
        if "option_type" not in candidates.columns:
            return pd.DataFrame()
        candidates["option_type"] = candidates["option_type"].astype(str).str.lower().str.strip()
        option_type = "call" if side == "long" else "put"
        option_action = "buy_call" if side == "long" else "buy_put"
        candidates = candidates.loc[candidates["option_type"].str.startswith(option_type[0])].copy()
        if candidates.empty:
            return candidates
        candidates["option_type"] = option_type
        candidates["option_action"] = option_action
        for col in ("strike", "bid", "ask", "mid", "volume", "open_interest"):
            if col not in candidates.columns:
                candidates[col] = np.nan
            candidates[col] = pd.to_numeric(candidates[col], errors="coerce")
        if "expiration" not in candidates.columns:
            candidates["expiration"] = pd.NaT
        candidates["expiration"] = pd.to_datetime(candidates["expiration"], errors="coerce").dt.normalize()
        if "snapshot_date" not in candidates.columns:
            candidates["snapshot_date"] = entry_date
        candidates["snapshot_date"] = pd.to_datetime(candidates["snapshot_date"], errors="coerce").dt.normalize()
        missing_mid = candidates["mid"].isna() & candidates["bid"].notna() & candidates["ask"].notna()
        candidates.loc[missing_mid, "mid"] = (candidates.loc[missing_mid, "bid"] + candidates.loc[missing_mid, "ask"]) / 2.0
        if "dte" not in candidates.columns:
            candidates["dte"] = (candidates["expiration"] - entry_date).dt.days
        else:
            candidates["dte"] = pd.to_numeric(candidates["dte"], errors="coerce")
            missing_dte = candidates["dte"].isna() & candidates["expiration"].notna()
            candidates.loc[missing_dte, "dte"] = (candidates.loc[missing_dte, "expiration"] - entry_date).dt.days
        candidates["underlying_spot_entry"] = float(spot)
        candidates["moneyness"] = candidates["strike"] / float(spot) - 1.0
        candidates["abs_moneyness"] = candidates["moneyness"].abs()
        candidates["spread"] = candidates["ask"] - candidates["bid"]
        if "spread_pct" not in candidates.columns:
            candidates["spread_pct"] = candidates["spread"] / candidates["mid"].replace(0, np.nan)
        else:
            candidates["spread_pct"] = pd.to_numeric(candidates["spread_pct"], errors="coerce")
            missing_spread = candidates["spread_pct"].isna()
            candidates.loc[missing_spread, "spread_pct"] = (
                candidates.loc[missing_spread, "spread"] / candidates.loc[missing_spread, "mid"].replace(0, np.nan)
            )
        candidates["dte_gap"] = (candidates["dte"] - int(self.config.target_dte)).abs()
        candidates["liquidity_score"] = candidates["volume"].fillna(0.0) + candidates["open_interest"].fillna(0.0) / 100.0
        candidates = candidates.loc[
            candidates["contract_symbol"].astype(str).ne("")
            & candidates["expiration"].notna()
            & candidates["strike"].gt(0)
            & candidates["bid"].ge(0)
            & candidates["ask"].gt(0)
            & candidates["ask"].ge(candidates["bid"])
            & candidates["mid"].gt(0)
            & candidates["dte"].gt(0)
        ].copy()
        if candidates.empty:
            return candidates
        if self.config.option_universe != "full_chain_actions":
            candidates = candidates.loc[
                candidates["dte"].between(self.config.min_dte, self.config.max_dte)
                & candidates["mid"].ge(self.config.min_entry_mid)
                & candidates["spread_pct"].le(self.config.max_entry_spread_pct)
                & candidates["abs_moneyness"].le(self.config.max_abs_moneyness)
            ].copy()
            if candidates.empty:
                return candidates
            candidates = self._filter_affordable_candidates(candidates, trade)
            if candidates.empty:
                return candidates
        candidates["option_return"] = np.nan
        candidates["fixed_near_atm_score"] = _rule_score(
            candidates,
            {"dte_gap": -1.0, "abs_moneyness": -1.0, "spread_pct": -1.0},
        )
        selected = _choose_top_per_trade(
            self._native_candidates_to_execution_rows(
                candidates,
                trade=trade,
                symbol=symbol,
                side=side,
                equity_exit_date=equity_exit_date,
                realized_holding_days=realized_holding_days,
            ),
            "fixed_near_atm_score",
        )
        return selected.reset_index(drop=True)

    def _native_candidates_to_execution_rows(
        self,
        candidates: pd.DataFrame,
        *,
        trade: pd.Series,
        symbol: str,
        side: str,
        equity_exit_date: pd.Timestamp,
        realized_holding_days: int,
    ) -> pd.DataFrame:
        rows = []
        for row in candidates.itertuples(index=False):
            entry_ask = _nan_float(getattr(row, "ask", np.nan))
            entry_bid = _nan_float(getattr(row, "bid", np.nan))
            entry_mid = _nan_float(getattr(row, "mid", np.nan))
            entry_price = entry_ask
            if not np.isfinite(entry_price) or entry_price <= 0:
                continue
            row_payload = {
                "trade_id": trade.get("trade_id", f"{symbol}|{pd.Timestamp(getattr(row, 'snapshot_date')).date()}|{side}"),
                "symbol": symbol,
                "side": side,
                "equity_signal_side": side,
                "entry_date": pd.Timestamp(getattr(row, "snapshot_date")).normalize(),
                "equity_exit_date": equity_exit_date,
                "option_exit_date": pd.NaT,
                "expiration": pd.Timestamp(getattr(row, "expiration")).normalize(),
                "contract_symbol": getattr(row, "contract_symbol"),
                "option_type": getattr(row, "option_type"),
                "option_action": getattr(row, "option_action"),
                "strike": _nan_float(getattr(row, "strike", np.nan)),
                "entry_mid": entry_mid,
                "exit_mid": np.nan,
                "entry_bid": entry_bid,
                "entry_ask": entry_ask,
                "exit_bid": np.nan,
                "exit_ask": np.nan,
                "entry_price": float(entry_price),
                "exit_price": np.nan,
                "option_pnl": np.nan,
                "return_denominator": float(entry_price),
                "option_return": np.nan,
                "realized_holding_days": realized_holding_days,
                "realized_underlying_trade_return": pd.to_numeric(trade.get("ret_dec"), errors="coerce"),
            }
            for source_col, target_col in (
                ("planned_holding_days", "planned_holding_days"),
                ("equity_signal_score", "equity_signal_score"),
                ("equity_signal_rank", "equity_signal_rank"),
            ):
                value = trade.get(source_col)
                if value is not None and not pd.isna(value):
                    row_payload[target_col] = _nan_float(value)
            for col in ("source_family", "strategy_source", "source", "family", "top_k", "variant"):
                value = trade.get(col)
                if value is not None and not pd.isna(value):
                    row_payload[col] = value if col == "top_k" else str(value)
            row_payload.update(_row_feature_payload(row))
            rows.append(row_payload)
        return pd.DataFrame(rows)

    def price_selected_options(self, selected: pd.DataFrame) -> pd.DataFrame:
        priced, _paths = self.price_selected_options_with_paths(selected)
        return priced

    def price_selected_options_with_paths(
        self,
        selected: pd.DataFrame,
        *,
        selector_name: str | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if selected.empty:
            return selected.copy(), pd.DataFrame()
        rows = []
        path_frames = []
        for payload in selected.to_dict("records"):
            priced, path = self._price_selected_option_payload_with_path(payload, selector_name=selector_name)
            if priced is not None:
                rows.append(priced)
            if path is not None and not path.empty:
                path_frames.append(path)
        priced_frame = pd.DataFrame(rows) if rows else pd.DataFrame(columns=selected.columns)
        path_frame = pd.concat(path_frames, ignore_index=True) if path_frames else pd.DataFrame()
        return priced_frame, path_frame

    def _price_selected_option_payload_with_path(
        self,
        payload: dict[str, Any],
        *,
        selector_name: str | None,
    ) -> tuple[dict[str, Any] | None, pd.DataFrame]:
        priced = self._price_option_payload(payload)
        if priced is None:
            return None, pd.DataFrame()
        path = self._selected_contract_path(payload, priced=priced, selector_name=selector_name)
        stats = _selected_contract_path_stats(path)
        if stats:
            priced.update(stats)
        return priced, path

    def _price_option_payload(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        symbol = str(payload.get("symbol", "")).upper()
        equity_exit_date = pd.Timestamp(payload.get("equity_exit_date")).normalize()
        option_type = "call" if str(payload.get("option_type", "")).lower().startswith("c") else "put"
        row = SimpleNamespace(**payload)
        settlement = self._price_exit_settlement(symbol, equity_exit_date, row, option_type)
        if settlement is None:
            return None
        exit_snapshot_date = settlement.snapshot_date
        exit_bid = settlement.bid
        exit_ask = settlement.ask
        exit_mid = settlement.mid
        option_action = str(payload.get("option_action", ""))
        entry_price = _nan_float(payload.get("entry_price", np.nan))
        if not np.isfinite(entry_price):
            if option_action.startswith("buy_"):
                entry_price = _nan_float(payload.get("entry_ask", payload.get("entry_mid", np.nan)))
            else:
                entry_price = _nan_float(payload.get("entry_bid", payload.get("entry_mid", np.nan)))
        denominator = _nan_float(payload.get("return_denominator", np.nan))
        if not np.isfinite(denominator) or denominator <= 0:
            if option_action == "sell_put":
                denominator = _nan_float(payload.get("strike", np.nan))
            elif option_action == "sell_call":
                denominator = _nan_float(payload.get("underlying_spot_entry", np.nan))
            else:
                denominator = entry_price
        if option_action.startswith("buy_"):
            exit_price = float(exit_bid)
            pnl = exit_price - float(entry_price)
        else:
            exit_price = float(exit_ask)
            pnl = float(entry_price) - exit_price
        if not np.isfinite(entry_price) or not np.isfinite(exit_price) or not np.isfinite(denominator) or denominator <= 0:
            return None
        priced = dict(payload)
        priced.update(
            {
                "option_exit_date": pd.Timestamp(exit_snapshot_date).normalize(),
                "exit_mid": float(exit_mid),
                "exit_bid": float(exit_bid),
                "exit_ask": float(exit_ask),
                "entry_price": float(entry_price),
                "exit_price": float(exit_price),
                "exit_price_source": settlement.price_source,
                "option_pnl": float(pnl),
                "return_denominator": float(denominator),
                "option_return": float(pnl) / float(denominator),
            }
        )
        return priced

    def _selected_contract_path(
        self,
        payload: dict[str, Any],
        *,
        priced: dict[str, Any],
        selector_name: str | None,
    ) -> pd.DataFrame:
        symbol = str(payload.get("symbol", "")).upper()
        contract_symbol = str(payload.get("contract_symbol", ""))
        entry_date = pd.Timestamp(payload.get("entry_date")).normalize()
        expiration = pd.Timestamp(payload.get("expiration")).normalize()
        equity_exit_date = pd.Timestamp(payload.get("equity_exit_date")).normalize()
        target_exit = min(equity_exit_date, expiration)
        path = self._load_contract_price_path(symbol, contract_symbol, entry_date, target_exit)
        option_action = str(payload.get("option_action", ""))
        if path.empty:
            path = pd.DataFrame()
        else:
            path = path.copy()
        final_date = pd.Timestamp(priced["option_exit_date"]).normalize()
        if path.empty or not pd.to_datetime(path["snapshot_date"], errors="coerce").dt.normalize().eq(final_date).any():
            path = pd.concat(
                [
                    path,
                    pd.DataFrame(
                        [
                            {
                                "snapshot_date": final_date,
                                "contract_symbol": contract_symbol,
                                "bid": priced.get("exit_bid", np.nan),
                                "ask": priced.get("exit_ask", np.nan),
                                "mid": priced.get("exit_mid", np.nan),
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
        if path.empty:
            return path
        path["snapshot_date"] = pd.to_datetime(path["snapshot_date"], errors="coerce").dt.normalize()
        for col in ("bid", "ask", "mid"):
            if col not in path.columns:
                path[col] = np.nan
            path[col] = pd.to_numeric(path[col], errors="coerce")
        quote_col = "bid" if option_action.startswith("buy_") else "ask"
        path["mark_price"] = path[quote_col].where(path[quote_col].notna(), path["mid"])
        entry_price = _nan_float(priced.get("entry_price", np.nan))
        denominator = _nan_float(priced.get("return_denominator", np.nan))
        if option_action.startswith("buy_"):
            path["path_pnl"] = path["mark_price"] - entry_price
        else:
            path["path_pnl"] = entry_price - path["mark_price"]
        path["path_return"] = path["path_pnl"] / denominator if np.isfinite(denominator) and denominator > 0 else np.nan
        path["selector"] = selector_name
        path["trade_id"] = payload.get("trade_id")
        path["symbol"] = symbol
        path["side"] = payload.get("side")
        path["option_action"] = option_action
        path["option_type"] = payload.get("option_type")
        path["entry_date"] = entry_date
        path["equity_exit_date"] = equity_exit_date
        path["option_exit_date"] = final_date
        path["expiration"] = expiration
        path["strike"] = _nan_float(payload.get("strike", np.nan))
        path["entry_price"] = entry_price
        path["return_denominator"] = denominator
        path["expired_before_equity_exit"] = bool(expiration < equity_exit_date)
        return path.sort_values(["snapshot_date", "contract_symbol"]).reset_index(drop=True)

    def _load_contract_price_path(
        self,
        symbol: str,
        contract_symbol: str,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
    ) -> pd.DataFrame:
        key = (
            str(symbol).upper(),
            str(contract_symbol),
            pd.Timestamp(start_date).normalize(),
            pd.Timestamp(end_date).normalize(),
        )
        cached = self._contract_path_cache.get(key)
        if cached is not None:
            self.contract_path_cache_hits += 1
            return cached.copy()
        self.contract_path_read_count += 1
        columns = list(dict.fromkeys([*self.config.chain_columns, "snapshot_date", "contract_symbol", "bid", "ask", "mid"]))
        chain = self.read_option_chain_arctic(
            key[0],
            start_date=key[2],
            end_date=key[3],
            columns=columns,
        )
        if chain.empty or "contract_symbol" not in chain.columns:
            out = pd.DataFrame()
        else:
            out = chain.loc[chain["contract_symbol"].astype(str).eq(key[1])].copy()
            if not out.empty:
                if "snapshot_date" not in out.columns:
                    out["snapshot_date"] = out.index
                out["snapshot_date"] = pd.to_datetime(out["snapshot_date"], errors="coerce").dt.normalize()
                for col in ("bid", "ask", "mid"):
                    if col not in out.columns:
                        out[col] = np.nan
                    out[col] = pd.to_numeric(out[col], errors="coerce")
                missing_mid = out["mid"].isna() & out["bid"].notna() & out["ask"].notna()
                out.loc[missing_mid, "mid"] = (out.loc[missing_mid, "bid"] + out.loc[missing_mid, "ask"]) / 2.0
                out = (
                    out.loc[out["snapshot_date"].notna()]
                    .sort_values("snapshot_date")
                    .drop_duplicates("snapshot_date", keep="last")
                    .reset_index(drop=True)
                )
        self._contract_path_cache[key] = out.copy()
        return out

    def _filter_affordable_candidates(self, candidates: pd.DataFrame, trade: pd.Series) -> pd.DataFrame:
        if candidates.empty or not bool(self.config.require_affordable):
            return candidates
        work = candidates.copy()
        self.affordability_checked_count += int(len(work))
        fraction = _allocation_fraction_for_trade(
            trade,
            default_fraction=_default_option_allocation_fraction(self.execution),
            top_k_column=str(self.execution.top_k_column),
        )
        budget = float(self.execution.capital) * float(fraction)
        min_contracts = max(1, int(self.config.min_affordable_contracts))
        multiplier = max(1, int(self.execution.multiplier))
        max_entry_mid = budget / float(min_contracts * multiplier) if budget > 0 else 0.0
        work["allocation_fraction"] = float(fraction)
        work["allocation_budget"] = float(budget)
        work["max_affordable_entry_mid"] = float(max_entry_mid)
        entry_mid = pd.to_numeric(work["mid"], errors="coerce")
        work["affordable_contracts"] = np.floor(float(budget) / (entry_mid * float(multiplier))).replace([np.inf, -np.inf], np.nan)
        affordable = entry_mid.le(max_entry_mid) & work["affordable_contracts"].ge(min_contracts)
        self.affordability_filtered_count += int((~affordable).sum())
        return work.loc[affordable].copy()

    def _price_exit(self, symbol: str, equity_exit_date: pd.Timestamp, row, option_type: str) -> tuple[pd.Timestamp, float] | None:
        settlement = self._price_exit_settlement(symbol, equity_exit_date, row, option_type)
        if settlement is None:
            return None
        return settlement.snapshot_date, settlement.mid

    def _price_exit_prices(self, symbol: str, equity_exit_date: pd.Timestamp, row, option_type: str) -> tuple[pd.Timestamp, float, float, float] | None:
        settlement = self._price_exit_settlement(symbol, equity_exit_date, row, option_type)
        if settlement is None:
            return None
        return settlement.snapshot_date, settlement.bid, settlement.ask, settlement.mid

    def _price_exit_settlement(
        self,
        symbol: str,
        equity_exit_date: pd.Timestamp,
        row,
        option_type: str,
    ) -> OptionSettlement | None:
        return settle_option_exit(
            symbol=symbol,
            contract_symbol=str(row.contract_symbol),
            option_type=option_type,
            strike=float(row.strike),
            expiration=pd.Timestamp(row.expiration).normalize(),
            equity_exit_date=pd.Timestamp(equity_exit_date).normalize(),
            quote_loader=self._chain_quote,
            underlying_close_loader=self._underlying_close,
            entry_date=getattr(row, "entry_date", None),
            exit_lookback_days=self.config.exit_lookback_days,
        )

    def _load_nearest_exit_chain(self, symbol: str, target_date: pd.Timestamp) -> tuple[pd.Timestamp | None, pd.DataFrame]:
        snapshot = self._nearest_exit_snapshot(symbol, target_date)
        if snapshot is None:
            return None, pd.DataFrame()
        return snapshot, self._load_day_chain(symbol, snapshot)

    def _nearest_exit_snapshot(self, symbol: str, target_date: pd.Timestamp) -> pd.Timestamp | None:
        key = (str(symbol).upper(), pd.Timestamp(target_date).normalize())
        if key in self._nearest_exit_snapshot_cache:
            self.nearest_exit_cache_hits += 1
            return self._nearest_exit_snapshot_cache[key]
        for date in pd.bdate_range(key[1], key[1] - pd.Timedelta(days=self.config.exit_lookback_days), freq="-1B"):
            snapshot = pd.Timestamp(date).normalize()
            chain = self._load_day_chain(key[0], snapshot)
            if not chain.empty:
                self._nearest_exit_snapshot_cache[key] = snapshot
                return snapshot
        self._nearest_exit_snapshot_cache[key] = None
        return None

    def _chain_quote(self, symbol: str, snapshot_date: pd.Timestamp, contract_symbol: str) -> tuple[float, float, float] | None:
        key = (str(symbol).upper(), pd.Timestamp(snapshot_date).normalize())
        quote_map = self._chain_quote_cache.get(key)
        if quote_map is not None:
            self.chain_quote_cache_hits += 1
        else:
            quote_map = self._load_day_quote_map(key[0], key[1])
            self._chain_quote_cache[key] = quote_map
        return quote_map.get(str(contract_symbol))

    def _load_day_quote_map(
        self,
        symbol: str,
        date: pd.Timestamp,
    ) -> dict[str, tuple[float, float, float]]:
        self.quote_chain_read_count += 1
        chain = self.read_option_chain_arctic(
            str(symbol).upper(),
            start_date=pd.Timestamp(date).normalize(),
            end_date=pd.Timestamp(date).normalize(),
            columns=["snapshot_date", "contract_symbol", "bid", "ask", "mid", "underlying_price"],
        )
        if chain.empty or "contract_symbol" not in chain.columns:
            return {}
        chain_spot = self._chain_underlying_price(chain)
        if chain_spot is not None:
            self._underlying_cache[(str(symbol).upper(), pd.Timestamp(date).normalize())] = chain_spot
        work = chain[
            [
                col
                for col in ("snapshot_date", "contract_symbol", "bid", "ask", "mid")
                if col in chain.columns
            ]
        ].copy()
        if "bid" not in work.columns:
            work["bid"] = np.nan
        if "ask" not in work.columns:
            work["ask"] = np.nan
        if "mid" not in work.columns:
            work["mid"] = np.nan
        work["contract_symbol"] = work["contract_symbol"].astype(str)
        work["bid"] = pd.to_numeric(work["bid"], errors="coerce")
        work["ask"] = pd.to_numeric(work["ask"], errors="coerce")
        work["mid"] = pd.to_numeric(work["mid"], errors="coerce")
        missing_mid = work["mid"].isna() & work["bid"].notna() & work["ask"].notna()
        work.loc[missing_mid, "mid"] = (work.loc[missing_mid, "bid"] + work.loc[missing_mid, "ask"]) / 2.0
        work["bid"] = work["bid"].where(work["bid"].notna(), work["mid"])
        work["ask"] = work["ask"].where(work["ask"].notna(), work["mid"])
        work = work.loc[work["contract_symbol"].ne("") & work["mid"].notna()].copy()
        before = len(work)
        if "snapshot_date" in work.columns:
            work["_snapshot_sort"] = pd.to_datetime(work["snapshot_date"], errors="coerce")
            work = work.sort_values(["contract_symbol", "_snapshot_sort"])
        else:
            work = work.sort_values(["contract_symbol"])
        work = work.drop_duplicates("contract_symbol", keep="last")
        self.quote_chain_duplicate_rows_dropped += int(before - len(work))
        return {
            str(row.contract_symbol): (float(row.bid), float(row.ask), float(row.mid))
            for row in work.itertuples(index=False)
        }

    def _load_day_chain(self, symbol: str, date: pd.Timestamp) -> pd.DataFrame:
        key = (str(symbol).upper(), pd.Timestamp(date).normalize())
        cached = self._chain_cache.get(key)
        if cached is not None:
            self.chain_cache_hits += 1
            return cached.copy()
        self.chain_read_count += 1
        chain = self.read_option_chain_arctic(
            key[0],
            start_date=key[1],
            end_date=key[1],
            columns=list(self.config.chain_columns),
        )
        if chain.empty:
            self._chain_cache[key] = chain.copy()
            return chain
        chain = self._dedupe_entry_chain_contracts(chain)
        chain_spot = self._chain_underlying_price(chain)
        if chain_spot is not None:
            self._underlying_cache[key] = chain_spot
        featured = self.build_option_contract_features(
            chain,
            underlying_price=chain_spot if chain_spot is not None else self._underlying_close(key[0], key[1]),
            target_dte=self.config.target_dte,
            compute_model_greeks=False,
        ).df
        self._chain_cache[key] = featured.copy()
        return featured

    def _load_day_chain_raw(self, symbol: str, date: pd.Timestamp) -> pd.DataFrame:
        key = (str(symbol).upper(), pd.Timestamp(date).normalize())
        cached = self._raw_chain_cache.get(key)
        if cached is not None:
            self.raw_chain_cache_hits += 1
            return cached.copy()
        self.chain_read_count += 1
        chain = self.read_option_chain_arctic(
            key[0],
            start_date=key[1],
            end_date=key[1],
            columns=list(self.config.chain_columns),
        )
        if not chain.empty:
            chain = self._dedupe_entry_chain_contracts(chain)
            chain_spot = self._chain_underlying_price(chain)
            if chain_spot is not None:
                self._underlying_cache[key] = chain_spot
        self._raw_chain_cache[key] = chain.copy()
        return chain

    def _dedupe_entry_chain_contracts(self, chain: pd.DataFrame) -> pd.DataFrame:
        if chain.empty or "contract_symbol" not in chain.columns:
            return chain
        before = len(chain)
        work = chain.copy()
        if "snapshot_date" in work.columns:
            work["_snapshot_sort"] = pd.to_datetime(work["snapshot_date"], errors="coerce")
            sort_cols = ["contract_symbol", "_snapshot_sort"]
        else:
            sort_cols = ["contract_symbol"]
        work = work.sort_values(sort_cols).drop_duplicates("contract_symbol", keep="last")
        if "_snapshot_sort" in work.columns:
            work = work.drop(columns=["_snapshot_sort"])
        self.chain_duplicate_rows_dropped += int(before - len(work))
        return work.reset_index(drop=True)

    @staticmethod
    def _chain_underlying_price(chain: pd.DataFrame) -> float | None:
        if chain is None or chain.empty or "underlying_price" not in chain.columns:
            return None
        values = pd.to_numeric(chain["underlying_price"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        values = values.dropna()
        values = values.loc[values > 0]
        if values.empty:
            return None
        return float(values.median())

    def _underlying_close(self, symbol: str, date: pd.Timestamp) -> float | None:
        key = (str(symbol).upper(), pd.Timestamp(date).normalize())
        if key in self._underlying_cache:
            self.underlying_cache_hits += 1
            return self._underlying_cache[key]
        self.underlying_lookup_count += 1
        result = self._close_from_frames(self.option_price_frames, key[0], key[1])
        if result is not None:
            self._underlying_cache[key] = result
            return result
        result = self._close_from_frames(self.price_frames, key[0], key[1])
        if result is not None:
            self.underlying_adjusted_fallback_count += 1
        self._underlying_cache[key] = result
        return result

    @staticmethod
    def _close_from_frames(
        frames: dict[str, pd.DataFrame],
        symbol: str,
        date: pd.Timestamp,
    ) -> float | None:
        prices = frames.get(str(symbol).upper())
        if prices is None or prices.empty or "close" not in prices.columns:
            return None
        eligible = prices.loc[prices.index <= pd.Timestamp(date).normalize()]
        if eligible.empty:
            return None
        value = pd.to_numeric(eligible["close"].iloc[-1], errors="coerce")
        return None if pd.isna(value) else float(value)

    def metrics(self) -> dict[str, float]:
        return {
            "option_chain_read_count": float(self.chain_read_count),
            "option_chain_cache_hits": float(self.chain_cache_hits),
            "option_chain_cache_size": float(len(self._chain_cache)),
            "option_raw_chain_cache_hits": float(self.raw_chain_cache_hits),
            "option_raw_chain_cache_size": float(len(self._raw_chain_cache)),
            "option_chain_duplicate_rows_dropped": float(self.chain_duplicate_rows_dropped),
            "nearest_exit_cache_hits": float(self.nearest_exit_cache_hits),
            "nearest_exit_cache_size": float(len(self._nearest_exit_snapshot_cache)),
            "chain_quote_cache_hits": float(self.chain_quote_cache_hits),
            "chain_quote_cache_size": float(len(self._chain_quote_cache)),
            "option_quote_chain_read_count": float(self.quote_chain_read_count),
            "option_quote_chain_duplicate_rows_dropped": float(self.quote_chain_duplicate_rows_dropped),
            "contract_path_read_count": float(self.contract_path_read_count),
            "contract_path_cache_hits": float(self.contract_path_cache_hits),
            "contract_path_cache_size": float(len(self._contract_path_cache)),
            "underlying_lookup_count": float(self.underlying_lookup_count),
            "underlying_cache_hits": float(self.underlying_cache_hits),
            "underlying_cache_size": float(len(self._underlying_cache)),
            "underlying_raw_price_frame_count": float(len(self.option_price_frames)),
            "underlying_adjusted_fallback_count": float(self.underlying_adjusted_fallback_count),
            "affordability_checked_count": float(self.affordability_checked_count),
            "affordability_filtered_count": float(self.affordability_filtered_count),
        }


def write_oracle_option_artifacts(
    *,
    config: OracleOptionExperimentConfig,
    coverage: pd.DataFrame,
    oracle_trades: pd.DataFrame,
    option_panel: pd.DataFrame,
    train_panel: pd.DataFrame,
    eval_panel: pd.DataFrame,
    selector_summary: pd.DataFrame,
    optopsy_summary: pd.DataFrame,
    symbol_summary: pd.DataFrame,
    source_family_summary: pd.DataFrame,
    feature_coverage: pd.DataFrame,
    model: Any,
    metrics: dict[str, float],
    analysis_markdown: str,
    selected_option_trades: pd.DataFrame | None = None,
    selected_option_paths: pd.DataFrame | None = None,
    directory: Path | None = None,
) -> dict[str, Path]:
    base = directory or option_experiment_artifact_dir(config)
    base.mkdir(parents=True, exist_ok=True)
    paths = {
        "coverage": base / "coverage.parquet",
        "oracle_trades": base / "oracle_trades.parquet",
        "trade_windows": base / "trade_windows.parquet",
        "option_panel": base / "option_candidate_panel.parquet",
        "train_panel": base / "train_panel.parquet",
        "eval_panel": base / "eval_panel.parquet",
        "selector_summary": base / "selector_summary.csv",
        "optopsy_summary": base / "optopsy_summary.csv",
        "symbol_summary": base / "symbol_summary.csv",
        "source_family_summary": base / "source_family_summary.csv",
        "feature_coverage": base / "feature_coverage.csv",
        "selected_option_trades": base / "selected_option_trades.parquet",
        "selected_option_paths": base / "selected_option_paths.parquet",
        "metrics": base / "metrics.json",
        "config": base / "config.json",
        "analysis": base / "analysis.md",
    }
    coverage.to_parquet(paths["coverage"], index=False)
    oracle_trades.to_parquet(paths["oracle_trades"], index=False)
    oracle_trades.to_parquet(paths["trade_windows"], index=False)
    option_panel.to_parquet(paths["option_panel"], index=False)
    train_panel.to_parquet(paths["train_panel"], index=False)
    eval_panel.to_parquet(paths["eval_panel"], index=False)
    selector_summary.to_csv(paths["selector_summary"], index=False)
    optopsy_summary.to_csv(paths["optopsy_summary"], index=False)
    symbol_summary.to_csv(paths["symbol_summary"], index=False)
    source_family_summary.to_csv(paths["source_family_summary"], index=False)
    feature_coverage.to_csv(paths["feature_coverage"], index=False)
    (selected_option_trades if selected_option_trades is not None else pd.DataFrame()).to_parquet(
        paths["selected_option_trades"],
        index=False,
    )
    (selected_option_paths if selected_option_paths is not None else pd.DataFrame()).to_parquet(
        paths["selected_option_paths"],
        index=False,
    )
    paths["metrics"].write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    paths["config"].write_text(json.dumps(_config_dict(config), indent=2, default=str), encoding="utf-8")
    paths["analysis"].write_text(analysis_markdown, encoding="utf-8")
    if model is not None:
        model_path = base / "option_ranker.pkl"
        with model_path.open("wb") as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
        paths["model"] = model_path
    return paths


def option_experiment_artifact_dir(config: OracleOptionExperimentConfig, *, run_name: str = "latest") -> Path:
    """Standard artifact location for option experiments when no override is supplied."""

    return Path("artifacts") / "options" / str(config.experiment_name) / str(run_name)


def load_option_experiment_artifacts(base_dir: str | Path) -> OptionExperimentArtifacts:
    base = Path(base_dir)
    metrics = json.loads((base / "metrics.json").read_text(encoding="utf-8")) if (base / "metrics.json").exists() else {}
    config = json.loads((base / "config.json").read_text(encoding="utf-8")) if (base / "config.json").exists() else {}
    trade_windows_path = base / "trade_windows.parquet"
    if not trade_windows_path.exists():
        trade_windows_path = base / "oracle_trades.parquet"
    source_family_path = base / "source_family_summary.csv"
    feature_coverage_path = base / "feature_coverage.csv"
    selected_trades_path = base / "selected_option_trades.parquet"
    selected_paths_path = base / "selected_option_paths.parquet"
    return OptionExperimentArtifacts(
        base_dir=base,
        coverage=pd.read_parquet(base / "coverage.parquet"),
        trade_windows=pd.read_parquet(trade_windows_path),
        option_panel=pd.read_parquet(base / "option_candidate_panel.parquet"),
        train_panel=pd.read_parquet(base / "train_panel.parquet"),
        eval_panel=pd.read_parquet(base / "eval_panel.parquet"),
        selector_summary=pd.read_csv(base / "selector_summary.csv"),
        optopsy_summary=pd.read_csv(base / "optopsy_summary.csv"),
        symbol_summary=pd.read_csv(base / "symbol_summary.csv"),
        source_family_summary=pd.read_csv(source_family_path) if source_family_path.exists() else pd.DataFrame(),
        feature_coverage=pd.read_csv(feature_coverage_path) if feature_coverage_path.exists() else pd.DataFrame(),
        selected_option_trades=pd.read_parquet(selected_trades_path) if selected_trades_path.exists() else pd.DataFrame(),
        selected_option_paths=pd.read_parquet(selected_paths_path) if selected_paths_path.exists() else pd.DataFrame(),
        metrics=metrics,
        config=config,
        analysis_markdown=(base / "analysis.md").read_text(encoding="utf-8") if (base / "analysis.md").exists() else "",
    )


def compare_option_experiment_artifacts(named_dirs: dict[str, str | Path]) -> pd.DataFrame:
    rows = []
    for group, directory in named_dirs.items():
        artifacts = load_option_experiment_artifacts(directory)
        for row in artifacts.optopsy_summary.itertuples(index=False):
            rows.append(
                {
                    "group": group,
                    "selector": getattr(row, "selector", None),
                    "closed_trades": getattr(row, "closed_trades", np.nan),
                    "final_equity": getattr(row, "final_equity", np.nan),
                    "total_return": getattr(row, "total_return", np.nan),
                    "max_drawdown": getattr(row, "max_drawdown", np.nan),
                    "sharpe_ratio": getattr(row, "sharpe_ratio", np.nan),
                }
            )
    return pd.DataFrame(rows)


def estimate_option_runtime_scaling(
    *,
    baseline_elapsed_seconds: float,
    baseline_trade_windows: int,
    baseline_option_rows: int,
    target_trade_windows: int,
    target_option_rows: int,
    max_seconds: float = 3600.0,
) -> OptionRuntimeEstimate:
    """Estimate whether a larger option experiment is runnable from a smoke run."""

    baseline_elapsed = max(float(baseline_elapsed_seconds), 0.0)
    baseline_windows = max(int(baseline_trade_windows), 1)
    baseline_rows = max(int(baseline_option_rows), 1)
    target_windows = max(int(target_trade_windows), 0)
    target_rows = max(int(target_option_rows), 0)
    by_windows = baseline_elapsed * (target_windows / baseline_windows)
    by_rows = baseline_elapsed * (target_rows / baseline_rows)
    conservative = max(by_windows, by_rows)
    return OptionRuntimeEstimate(
        baseline_elapsed_seconds=baseline_elapsed,
        baseline_trade_windows=baseline_windows,
        baseline_option_rows=baseline_rows,
        target_trade_windows=target_windows,
        target_option_rows=target_rows,
        estimated_seconds_by_windows=float(by_windows),
        estimated_seconds_by_option_rows=float(by_rows),
        estimated_seconds_conservative=float(conservative),
        max_seconds=float(max_seconds),
        runnable_within_budget=bool(conservative <= float(max_seconds)),
    )


def _execute_rule_trade_windows(
    trade_windows: pd.DataFrame,
    retriever: _OptionRetriever,
    *,
    selector_name: str = "rule_atm_90d",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return execute_rule_trade_windows(trade_windows, retriever, selector_name=selector_name)


def _select_rule_atm_option_entry(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame()
    work = candidates.copy()
    if "option_action" in work.columns:
        work = work.loc[work["option_action"].astype(str).str.lower().isin(["buy_call", "buy_put"])].copy()
    if work.empty:
        return work
    for col in ("dte_gap", "abs_moneyness", "spread_pct"):
        if col not in work.columns:
            work[col] = np.nan
    if "option_return" not in work.columns:
        work["option_return"] = np.nan
    work["fixed_near_atm_score"] = _rule_score(
        work,
        {"dte_gap": -1.0, "abs_moneyness": -1.0, "spread_pct": -1.0},
    )
    return _choose_top_per_trade(work, "fixed_near_atm_score")


def _build_option_panel(oracle_trades: pd.DataFrame, retriever: _OptionRetriever, *, price_exit: bool = True) -> pd.DataFrame:
    frames = []
    for trade in oracle_trades.itertuples(index=False):
        frame = retriever.retrieve(pd.Series(trade._asdict()), price_exit=price_exit)
        if frame is not None and not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    panel = pd.concat(frames, ignore_index=True)
    if "option_return" in panel.columns:
        returns = pd.to_numeric(panel["option_return"], errors="coerce")
        if returns.notna().any():
            panel["rank_y"] = returns.groupby(panel["trade_id"]).rank(method="average", pct=True, ascending=True)
    return panel


def _split_trade_windows(oracle_trades: pd.DataFrame, split: SharedSplitConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    if oracle_trades.empty:
        return pd.DataFrame(), pd.DataFrame()
    dates = pd.to_datetime(oracle_trades["entry_date"], errors="coerce").dt.normalize()
    return oracle_trades.loc[dates.le(split.insample_end_ts)].copy(), oracle_trades.loc[dates.ge(split.oos_start_ts)].copy()


def _add_mean_variance_basket_labels(
    option_panel: pd.DataFrame,
    config: OptionMvBasketConfig,
    *,
    build_option_mean_variance_labels,
) -> pd.DataFrame:
    if option_panel.empty:
        return option_panel
    work = option_panel.copy()
    risk_terms = []
    for col in ("spread_pct", "abs_moneyness", "dte_gap"):
        if col in work.columns:
            values = pd.to_numeric(work[col], errors="coerce").fillna(0.0)
            scale = values.abs().median()
            if pd.notna(scale) and float(scale) > 0:
                values = values / float(scale)
            risk_terms.append(values.clip(lower=0.0))
    work["mv_risk_proxy"] = sum(risk_terms) if risk_terms else 0.0
    labels = build_option_mean_variance_labels(
        work,
        group_cols=("trade_id",),
        expected_return_col="option_return",
        risk_col="mv_risk_proxy",
        risk_aversion=float(config.risk_aversion),
        max_weight=float(config.max_weight),
        long_only=True,
    )
    keep = ["trade_id", "contract_symbol", "mv_score", "mv_rank", "mv_selected", "mv_weight", "target_value"]
    keep = [col for col in keep if col in labels.columns]
    labeled = work.merge(labels[keep], on=["trade_id", "contract_symbol"], how="left")
    for col in ("mv_score", "mv_weight", "target_value"):
        if col in labeled.columns:
            labeled[col] = pd.to_numeric(labeled[col], errors="coerce").fillna(0.0)
    if "mv_selected" in labeled.columns:
        labeled["mv_selected"] = labeled["mv_selected"].fillna(False).astype(bool)
    return labeled


def _train_option_ranker(
    train_panel: pd.DataFrame,
    eval_panel: pd.DataFrame,
    *,
    config: OracleOptionExperimentConfig,
    option_ranker_feature_columns,
) -> tuple[Any, dict[str, float], pd.DataFrame]:
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    eval_scored = eval_panel.copy()
    eval_scored["pred_return"] = np.nan
    eval_scored["pred_rank_score"] = np.nan
    if len(train_panel) < 50 or eval_panel.empty:
        return None, {}, eval_scored
    features_num = option_ranker_feature_columns(train_panel)
    features_cat = ["symbol", "equity_signal_side", "option_type", "option_action", *config.categorical_trade_context_columns]
    numeric = [col for col in features_num if col in train_panel.columns and train_panel[col].notna().any()]
    categorical = [col for col in features_cat if col in train_panel.columns and train_panel[col].notna().any()]
    def make_model(seed: int) -> Pipeline:
        return Pipeline(
            [
                (
                    "prep",
                    ColumnTransformer(
                        [
                            ("num", SimpleImputer(strategy="median"), numeric),
                            (
                                "cat",
                                Pipeline(
                                    [
                                        ("impute", SimpleImputer(strategy="most_frequent")),
                                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                                    ]
                                ),
                                categorical,
                            ),
                        ],
                        remainder="drop",
                    ),
                ),
                (
                    "rf",
                    RandomForestRegressor(
                        n_estimators=300,
                        max_depth=8,
                        min_samples_leaf=5,
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    target = "option_return"
    model = make_model(config.random_seed)
    x_train = train_panel[numeric + categorical]
    x_eval = eval_scored[numeric + categorical]
    model.fit(x_train, train_panel[target])
    train_pred = model.predict(x_train)
    eval_scored["pred_return"] = model.predict(x_eval)
    rank_model = None
    rank_train_pred = np.full(len(train_panel), np.nan)
    if "rank_y" in train_panel.columns and pd.to_numeric(train_panel["rank_y"], errors="coerce").notna().any():
        rank_model = make_model(config.random_seed + 31)
        rank_target = pd.to_numeric(train_panel["rank_y"], errors="coerce")
        rank_model.fit(x_train, rank_target)
        rank_train_pred = rank_model.predict(x_train)
        eval_scored["pred_rank_score"] = np.clip(rank_model.predict(x_eval), 0.0, 1.0)
    metrics = {
        "train_rows": float(len(train_panel)),
        "eval_rows": float(len(eval_scored)),
        "train_trades": float(train_panel["trade_id"].nunique()),
        "eval_trades": float(eval_scored["trade_id"].nunique()),
        "train_mae": float(mean_absolute_error(train_panel[target], train_pred)),
        "train_r2": float(r2_score(train_panel[target], train_pred)),
        "numeric_feature_count": float(len(numeric)),
        "categorical_feature_count": float(len(categorical)),
        "ranker_model_count": 1.0 + (1.0 if rank_model is not None else 0.0),
    }
    eval_target = pd.to_numeric(eval_scored[target], errors="coerce") if target in eval_scored.columns else pd.Series(dtype=float)
    eval_label_mask = eval_target.notna() & pd.to_numeric(eval_scored["pred_return"], errors="coerce").notna()
    metrics["eval_mae"] = (
        float(mean_absolute_error(eval_target.loc[eval_label_mask], eval_scored.loc[eval_label_mask, "pred_return"]))
        if eval_label_mask.any()
        else np.nan
    )
    metrics["eval_r2"] = (
        float(r2_score(eval_target.loc[eval_label_mask], eval_scored.loc[eval_label_mask, "pred_return"]))
        if int(eval_label_mask.sum()) > 1
        else np.nan
    )
    if rank_model is not None and "rank_y" in eval_scored.columns:
        eval_rank_target = pd.to_numeric(eval_scored["rank_y"], errors="coerce")
        eval_rank_mask = eval_rank_target.notna() & pd.to_numeric(eval_scored["pred_rank_score"], errors="coerce").notna()
        metrics.update(
            {
                "rank_train_mae": float(mean_absolute_error(train_panel["rank_y"], rank_train_pred)),
                "rank_train_r2": float(r2_score(train_panel["rank_y"], rank_train_pred)) if len(train_panel) > 1 else np.nan,
                "rank_eval_mae": (
                    float(mean_absolute_error(eval_rank_target.loc[eval_rank_mask], eval_scored.loc[eval_rank_mask, "pred_rank_score"]))
                    if eval_rank_mask.any()
                    else np.nan
                ),
                "rank_eval_r2": (
                    float(r2_score(eval_rank_target.loc[eval_rank_mask], eval_scored.loc[eval_rank_mask, "pred_rank_score"]))
                    if int(eval_rank_mask.sum()) > 1
                    else np.nan
                ),
            }
        )
    metrics["numeric_features"] = ",".join(numeric)
    metrics["categorical_features"] = ",".join(categorical)
    return {"return_ranker": model, "within_trade_ranker": rank_model}, metrics, eval_scored


def _train_mv_basket_ranker(
    train_panel: pd.DataFrame,
    eval_panel: pd.DataFrame,
    *,
    config: OracleOptionExperimentConfig,
    option_ranker_feature_columns,
) -> tuple[Any, dict[str, float], pd.DataFrame]:
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    eval_scored = eval_panel.copy()
    eval_scored["pred_mv_weight"] = np.nan
    if not config.mv_basket.enabled or "mv_weight" not in train_panel.columns or len(train_panel) < 50 or eval_panel.empty:
        return None, {}, eval_scored
    features_num = option_ranker_feature_columns(train_panel)
    features_cat = ["symbol", "equity_signal_side", "option_type", "option_action", *config.categorical_trade_context_columns]
    numeric = [col for col in features_num if col in train_panel.columns and train_panel[col].notna().any()]
    categorical = [col for col in features_cat if col in train_panel.columns and train_panel[col].notna().any()]
    if not numeric and not categorical:
        return None, {}, eval_scored
    model = Pipeline(
        [
            (
                "prep",
                ColumnTransformer(
                    [
                        ("num", SimpleImputer(strategy="median"), numeric),
                        (
                            "cat",
                            Pipeline(
                                [
                                    ("impute", SimpleImputer(strategy="most_frequent")),
                                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                                ]
                            ),
                            categorical,
                        ),
                    ],
                    remainder="drop",
                ),
            ),
            (
                "rf",
                RandomForestRegressor(
                    n_estimators=300,
                    max_depth=8,
                    min_samples_leaf=5,
                    random_state=config.random_seed + 17,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    target = "mv_weight"
    model.fit(train_panel[numeric + categorical], train_panel[target])
    train_pred = model.predict(train_panel[numeric + categorical])
    eval_scored["pred_mv_weight"] = np.clip(model.predict(eval_scored[numeric + categorical]), 0.0, None)
    eval_target = pd.to_numeric(eval_scored[target], errors="coerce") if target in eval_scored.columns else pd.Series(dtype=float)
    eval_pred = pd.to_numeric(eval_scored["pred_mv_weight"], errors="coerce")
    eval_label_mask = eval_target.notna() & eval_pred.notna()
    metrics = {
        "mv_train_mae": float(mean_absolute_error(train_panel[target], train_pred)),
        "mv_train_r2": float(r2_score(train_panel[target], train_pred)) if len(train_panel) > 1 else np.nan,
        "mv_eval_mae": (
            float(mean_absolute_error(eval_target.loc[eval_label_mask], eval_scored.loc[eval_label_mask, "pred_mv_weight"]))
            if eval_label_mask.any()
            else np.nan
        ),
        "mv_eval_r2": (
            float(r2_score(eval_target.loc[eval_label_mask], eval_scored.loc[eval_label_mask, "pred_mv_weight"]))
            if int(eval_label_mask.sum()) > 1
            else np.nan
        ),
        "mv_numeric_feature_count": float(len(numeric)),
        "mv_categorical_feature_count": float(len(categorical)),
    }
    metrics["mv_numeric_features"] = ",".join(numeric)
    metrics["mv_categorical_features"] = ",".join(categorical)
    return model, metrics, eval_scored


def _selector_summaries(
    eval_panel: pd.DataFrame,
    *,
    mv_basket_config: OptionMvBasketConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    selected_by_selector = _select_options_by_selector(eval_panel, mv_basket_config=mv_basket_config)
    selector_summary, symbol_summary = _summarize_selected_by_selector(selected_by_selector)
    return selector_summary, symbol_summary, selected_by_selector


def _select_options_by_selector(
    eval_panel: pd.DataFrame,
    *,
    mv_basket_config: OptionMvBasketConfig | None = None,
) -> dict[str, pd.DataFrame]:
    fixed_work = eval_panel.copy()
    if not fixed_work.empty:
        if "pred_return" in fixed_work.columns and "pred_rank_score" in fixed_work.columns:
            fixed_work["pred_return_pct"] = fixed_work.groupby("trade_id")["pred_return"].rank(method="average", pct=True, ascending=True)
            fixed_work["pred_blended_rank_score"] = 0.75 * pd.to_numeric(
                fixed_work["pred_rank_score"], errors="coerce"
            ) + 0.25 * pd.to_numeric(fixed_work["pred_return_pct"], errors="coerce")
        fixed_work["fixed_near_atm_score"] = _rule_score(
            fixed_work,
            {"dte_gap": -1.0, "abs_moneyness": -1.0, "spread_pct": -1.0},
        )
        fixed_work["highest_liquidity_score"] = _rule_score(
            fixed_work,
            {"liquidity_score": 1.0, "spread_pct": -1.0, "abs_moneyness": -0.1},
        )
        fixed_work["lowest_spread_score"] = _rule_score(
            fixed_work,
            {"spread_pct": -1.0, "dte_gap": -0.1, "abs_moneyness": -0.1},
        )
    selected_model = _choose_top_per_trade(fixed_work, "pred_return")
    selected_oracle = _choose_top_per_trade(fixed_work, "option_return")
    selected_rule_atm = _choose_top_per_trade(fixed_work, "fixed_near_atm_score")
    selected_mv_model = _choose_weighted_basket_per_trade(
        eval_panel,
        "pred_mv_weight",
        selector_name="model_mv_basket",
        max_legs=mv_basket_config.max_legs if mv_basket_config else None,
        min_weight=mv_basket_config.min_predicted_weight if mv_basket_config else 0.0,
    )
    selected_mv_oracle = _choose_weighted_basket_per_trade(
        eval_panel,
        "mv_weight",
        selector_name="oracle_mv_basket",
        max_legs=mv_basket_config.max_legs if mv_basket_config else None,
        min_weight=0.0,
    )
    return {
        "rule_atm_90d": selected_rule_atm,
        "model_ranker": selected_model,
        "oracle_best_possible": selected_oracle,
        "model_mv_basket": selected_mv_model,
        "oracle_mv_basket": selected_mv_oracle,
    }


def _price_selected_by_selector(
    selected_by_selector: dict[str, pd.DataFrame],
    retriever: _OptionRetriever,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    priced_by_selector: dict[str, pd.DataFrame] = {}
    path_frames = []
    for selector, selected in selected_by_selector.items():
        priced, paths = retriever.price_selected_options_with_paths(selected, selector_name=selector)
        priced_by_selector[selector] = priced
        if not paths.empty:
            path_frames.append(paths)
    selected_paths = pd.concat(path_frames, ignore_index=True) if path_frames else pd.DataFrame()
    return priced_by_selector, selected_paths


def _selected_by_selector_to_frame(selected_by_selector: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for selector, selected in selected_by_selector.items():
        if selected.empty:
            continue
        frame = selected.copy()
        frame["selector"] = selector
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _summarize_selected_by_selector(
    selected_by_selector: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_rule_atm = selected_by_selector.get("rule_atm_90d", pd.DataFrame())
    selected_model = selected_by_selector.get("model_ranker", pd.DataFrame())
    selected_oracle = selected_by_selector.get("oracle_best_possible", pd.DataFrame())
    selected_mv_model = selected_by_selector.get("model_mv_basket", pd.DataFrame())
    selected_mv_oracle = selected_by_selector.get("oracle_mv_basket", pd.DataFrame())
    selector_summary = pd.DataFrame(
        [
            _summarize_selected("rule_atm_90d", selected_rule_atm),
            _summarize_selected("model_ranker", selected_model),
            _summarize_selected("oracle_best_possible", selected_oracle),
            _summarize_basket("model_mv_basket", selected_mv_model),
            _summarize_basket("oracle_mv_basket", selected_mv_oracle),
        ]
    )
    if selected_model.empty:
        symbol_summary = pd.DataFrame()
    else:
        symbol_summary = (
            selected_model.groupby(["symbol", "side"])
            .agg(
                trades=("trade_id", "count"),
                mean_return=("option_return", "mean"),
                win_rate=("option_return", lambda x: float((x > 0).mean())),
            )
            .reset_index()
            .sort_values("mean_return", ascending=False)
        )
    return selector_summary, symbol_summary


def _source_family_diagnostics(eval_panel: pd.DataFrame, selected_by_selector: dict[str, pd.DataFrame]) -> pd.DataFrame:
    group_cols = [col for col in ("strategy_source", "source", "family", "source_family") if col in eval_panel.columns]
    if eval_panel.empty or not group_cols:
        return pd.DataFrame()
    rows = []
    work = eval_panel.copy()
    work["option_return"] = pd.to_numeric(work["option_return"], errors="coerce")
    grouped = (
        work.groupby(group_cols, dropna=False)
        .agg(
            option_rows=("trade_id", "size"),
            eval_trade_windows=("trade_id", "nunique"),
            symbols=("symbol", "nunique"),
            mean_candidate_return=("option_return", "mean"),
            median_candidate_return=("option_return", "median"),
            candidate_win_rate=("option_return", lambda values: float((pd.to_numeric(values, errors="coerce") > 0).mean())),
        )
        .reset_index()
    )
    for _, row in grouped.iterrows():
        payload = row.to_dict()
        mask = pd.Series(True, index=work.index)
        for col in group_cols:
            mask &= work[col].astype(str).eq(str(row[col]))
        payload["first_entry_date"] = work.loc[mask, "entry_date"].min() if "entry_date" in work else pd.NaT
        payload["last_entry_date"] = work.loc[mask, "entry_date"].max() if "entry_date" in work else pd.NaT
        for selector, selected in selected_by_selector.items():
            if selected.empty:
                payload[f"{selector}_selected_rows"] = 0
                payload[f"{selector}_selected_trades"] = 0
                payload[f"{selector}_mean_return"] = np.nan
                continue
            selected_work = selected.copy()
            selected_mask = pd.Series(True, index=selected_work.index)
            for col in group_cols:
                if col not in selected_work.columns:
                    selected_mask &= False
                else:
                    selected_mask &= selected_work[col].astype(str).eq(str(row[col]))
            selected_slice = selected_work.loc[selected_mask].copy()
            returns = pd.to_numeric(selected_slice["option_return"], errors="coerce") if "option_return" in selected_slice else pd.Series(dtype=float)
            payload[f"{selector}_selected_rows"] = int(len(selected_slice))
            payload[f"{selector}_selected_trades"] = int(selected_slice["trade_id"].nunique()) if "trade_id" in selected_slice else 0
            payload[f"{selector}_mean_return"] = float(returns.mean()) if len(returns) else np.nan
        rows.append(payload)
    return pd.DataFrame(rows).sort_values(["eval_trade_windows", "option_rows"], ascending=[False, False]).reset_index(drop=True)


def _option_feature_coverage(option_panel: pd.DataFrame, *, train_panel: pd.DataFrame, eval_panel: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [
        "dte",
        "dte_gap",
        "moneyness",
        "abs_moneyness",
        "spread_pct",
        "volume",
        "open_interest",
        "liquidity_score",
        "delta",
        "abs_delta",
        "gamma",
        "theta",
        "vega",
        "rho",
        "iv",
        "implied_volatility",
        "iv_expiration_z",
        "theta_to_mid",
        "vega_to_mid",
        "allocation_fraction",
        "allocation_budget",
        "affordable_contracts",
    ]
    frames = {"all": option_panel, "train": train_panel, "eval": eval_panel}
    rows: list[dict[str, Any]] = []
    for split_name, frame in frames.items():
        for col in feature_cols:
            if col not in frame.columns:
                rows.append(
                    {
                        "split": split_name,
                        "feature": col,
                        "rows": int(len(frame)),
                        "present": False,
                        "non_null_count": 0,
                        "non_null_rate": 0.0,
                    }
                )
                continue
            values = pd.to_numeric(frame[col], errors="coerce")
            non_null = int(values.notna().sum())
            rows.append(
                {
                    "split": split_name,
                    "feature": col,
                    "rows": int(len(frame)),
                    "present": True,
                    "non_null_count": non_null,
                    "non_null_rate": float(non_null / len(frame)) if len(frame) else 0.0,
                    "mean": float(values.mean()) if non_null else np.nan,
                    "median": float(values.median()) if non_null else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _option_feature_coverage_metrics(feature_coverage: pd.DataFrame) -> dict[str, float]:
    if feature_coverage.empty:
        return {}
    metrics: dict[str, float] = {}
    for feature in ("delta", "gamma", "theta", "vega", "rho", "iv", "implied_volatility"):
        row = feature_coverage.loc[
            feature_coverage["split"].astype(str).eq("all") & feature_coverage["feature"].astype(str).eq(feature)
        ]
        if row.empty:
            continue
        metrics[f"coverage_{feature}_non_null_rate"] = float(row["non_null_rate"].iloc[0])
    return metrics


def _choose_weighted_basket_per_trade(
    frame: pd.DataFrame,
    weight_col: str,
    *,
    selector_name: str,
    max_legs: int | None = None,
    min_weight: float = 0.0,
) -> pd.DataFrame:
    if frame.empty or weight_col not in frame.columns:
        return pd.DataFrame()
    max_legs = 4 if max_legs is None else int(max_legs)
    rows = []
    for trade_id, group in frame.groupby("trade_id", sort=False):
        work = group.copy()
        work["_raw_weight"] = pd.to_numeric(work[weight_col], errors="coerce").fillna(0.0).clip(lower=0.0)
        work = work.loc[work["_raw_weight"].gt(float(min_weight))].copy()
        if work.empty:
            continue
        work = work.sort_values(["_raw_weight", "option_return"], ascending=[False, False]).head(max_legs).copy()
        total = float(work["_raw_weight"].sum())
        if total <= 0:
            continue
        work["basket_weight"] = work["_raw_weight"] / total
        work["selector"] = selector_name
        rows.append(work.drop(columns=["_raw_weight"]))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _rule_score(frame: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    score = pd.Series(0.0, index=frame.index)
    for col, weight in weights.items():
        if col not in frame.columns:
            continue
        values = pd.to_numeric(frame[col], errors="coerce")
        if col == "liquidity_score":
            values = np.log1p(values.clip(lower=0))
        scale = values.abs().median()
        if pd.notna(scale) and float(scale) > 0:
            values = values / float(scale)
        score = score + float(weight) * values.fillna(0.0)
    return score


def _selected_contract_path_stats(path: pd.DataFrame) -> dict[str, float | int] | None:
    if path.empty or "path_return" not in path.columns:
        return None
    returns = pd.to_numeric(path["path_return"], errors="coerce").dropna()
    if returns.empty:
        return None
    equity = 1.0 + returns
    running_peak = equity.cummax()
    drawdown = equity / running_peak.replace(0, np.nan) - 1.0
    return {
        "path_observations": int(len(returns)),
        "path_min_return": float(returns.min()),
        "path_max_return": float(returns.max()),
        "path_max_drawdown": float(drawdown.min()) if drawdown.notna().any() else np.nan,
    }


def _run_optopsy_selector_backtests(
    selected_by_selector: dict[str, pd.DataFrame],
    config: OptopsyExecutionConfig,
) -> pd.DataFrame:
    rows = []
    for selector_name, selected in selected_by_selector.items():
        result = _run_optopsy_selected_option_backtest(selected, config=config)
        row = {"selector": selector_name, "framework": "optopsy", "selected_rows": int(len(selected))}
        if result is None:
            row.update({"closed_trades": 0, "note": "no selected trades"})
        else:
            summary = dict(result.summary)
            final_equity = (
                float(result.equity_curve.iloc[-1])
                if getattr(result, "equity_curve", pd.Series(dtype=float)) is not None and not result.equity_curve.empty
                else float(config.capital)
            )
            row.update(
                {
                    "closed_trades": int(len(result.trade_log)),
                    "equity_points": int(len(result.equity_curve)),
                    "final_equity": final_equity,
                    **{str(key): value for key, value in summary.items()},
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _run_optopsy_selected_option_backtest(
    selected: pd.DataFrame,
    *,
    config: OptopsyExecutionConfig,
):
    if selected.empty:
        return None
    if "option_action" in selected.columns:
        return _run_action_aware_selected_option_backtest(selected, config=config)
    import optopsy as op

    raw = _selected_basket_to_optopsy_raw(selected) if "basket_weight" in selected.columns else _selected_options_to_optopsy_raw(selected)
    if raw.empty:
        return None

    if str(config.sizing_mode) == "portfolio_fraction":
        return _run_optopsy_portfolio_fraction_backtest(raw, selected=selected, config=config)
    if str(config.sizing_mode) != "fixed_quantity":
        raise ValueError(f"unknown option sizing_mode={config.sizing_mode!r}")

    def _selected_strategy(data: pd.DataFrame, *, raw: bool = False, **kwargs: Any):
        return data.copy()

    _selected_strategy.__name__ = "long_calls"
    simulation = op.simulate(
        raw,
        _selected_strategy,
        capital=float(config.capital),
        quantity=int(config.quantity),
        max_positions=int(config.max_positions),
        multiplier=int(config.multiplier),
        selector=str(config.selector),
        exit_dte=0,
    )
    return simulation


def _run_action_aware_selected_option_backtest(
    selected: pd.DataFrame,
    *,
    config: OptopsyExecutionConfig,
):
    trades = _selected_actions_to_trade_rows(selected)
    if trades.empty:
        return None
    trade_log = _build_portfolio_fraction_trade_log(
        trades,
        capital=float(config.capital),
        multiplier=int(config.multiplier),
        max_positions=int(config.max_positions),
        default_fraction=_default_option_allocation_fraction(config),
        top_k_column=str(config.top_k_column),
    )
    import optopsy.simulator as optopsy_simulator

    summary = optopsy_simulator._compute_summary(trade_log, float(config.capital))
    equity_curve = trade_log["equity"].copy() if not trade_log.empty else pd.Series(dtype=float, name="equity")
    if not trade_log.empty and "exit_date" in trade_log:
        equity_curve.index = pd.to_datetime(trade_log["exit_date"])
    return SimpleNamespace(trade_log=trade_log, equity_curve=equity_curve, summary=summary)


def _selected_actions_to_trade_rows(selected: pd.DataFrame) -> pd.DataFrame:
    if selected.empty:
        return pd.DataFrame()
    required = ["symbol", "entry_date", "option_exit_date", "expiration", "return_denominator", "option_pnl", "option_return"]
    missing = [col for col in required if col not in selected.columns]
    if missing:
        raise ValueError(f"selected action-aware option trades missing columns: {missing}")
    rows = []
    work = selected.copy()
    if "basket_weight" in work.columns:
        work["basket_weight"] = pd.to_numeric(work["basket_weight"], errors="coerce").fillna(0.0).clip(lower=0.0)
        for trade_id, group in work.groupby("trade_id", sort=False):
            weights = group["basket_weight"]
            total = float(weights.sum())
            if total <= 0:
                continue
            weights = weights / total
            denominators = pd.to_numeric(group["return_denominator"], errors="coerce").fillna(0.0).clip(lower=0.0)
            pnls = pd.to_numeric(group["option_pnl"], errors="coerce").fillna(0.0)
            entry_cost = float((weights * denominators).sum())
            pnl = float((weights * pnls).sum())
            if entry_cost <= 0:
                continue
            base = group.iloc[0]
            rows.append(
                {
                    "trade_id": trade_id,
                    "underlying_symbol": str(base["symbol"]).upper(),
                    "entry_date": pd.Timestamp(base["entry_date"]).normalize(),
                    "exit_date": pd.Timestamp(base["option_exit_date"]).normalize(),
                    "expiration": pd.to_datetime(group["expiration"], errors="coerce").min(),
                    "entry_cost": entry_cost,
                    "exit_proceeds": entry_cost + pnl,
                    "pct_change": pnl / entry_cost,
                    "description": "model_mv_basket",
                    "exit_type": "basket_signal_exit",
                    "top_k": base.get("top_k", np.nan),
                }
            )
    else:
        for row in work.itertuples(index=False):
            entry_cost = _nan_float(getattr(row, "return_denominator"))
            pnl = _nan_float(getattr(row, "option_pnl"))
            if pd.isna(entry_cost) or entry_cost <= 0 or pd.isna(pnl):
                continue
            rows.append(
                {
                    "trade_id": getattr(row, "trade_id", None),
                    "underlying_symbol": str(getattr(row, "symbol")).upper(),
                    "entry_date": pd.Timestamp(getattr(row, "entry_date")).normalize(),
                    "exit_date": pd.Timestamp(getattr(row, "option_exit_date")).normalize(),
                    "expiration": pd.Timestamp(getattr(row, "expiration")).normalize(),
                    "entry_cost": entry_cost,
                    "exit_proceeds": entry_cost + pnl,
                    "pct_change": _nan_float(getattr(row, "option_return")),
                    "description": getattr(row, "option_action", "option_action"),
                    "exit_type": "signal_exit",
                    "top_k": getattr(row, "top_k", np.nan),
                }
            )
    return pd.DataFrame(rows)


def _run_optopsy_portfolio_fraction_backtest(
    raw: pd.DataFrame,
    *,
    selected: pd.DataFrame,
    config: OptopsyExecutionConfig,
):
    """Use Optopsy selection/filter semantics, then size each trade by portfolio fraction."""

    import optopsy.simulator as optopsy_simulator

    selector = str(config.selector)
    selectors = getattr(optopsy_simulator, "_BUILTIN_SELECTORS")
    if selector not in selectors:
        raise ValueError(f"portfolio_fraction sizing currently requires a built-in Optopsy selector, got {selector!r}")
    select_fn = selectors[selector]

    work = raw.copy()
    work["_entry_date"] = optopsy_simulator._resolve_entry_date(work)
    group_cols = ["underlying_symbol", "_entry_date"] if "underlying_symbol" in work.columns else ["_entry_date"]
    selected_raw = pd.DataFrame([select_fn(group) for _, group in work.groupby(group_cols)])
    trades = optopsy_simulator._normalise_trades(selected_raw, is_short_single=False, exit_dte=0)
    for col in ("trade_id", str(config.top_k_column)):
        if col in selected_raw.columns:
            trades[col] = selected_raw[col].values
    trades = trades.sort_values("entry_date").reset_index(drop=True)
    trade_log = _build_portfolio_fraction_trade_log(
        trades,
        capital=float(config.capital),
        multiplier=int(config.multiplier),
        max_positions=int(config.max_positions),
        default_fraction=_default_option_allocation_fraction(config),
        top_k_column=str(config.top_k_column),
    )
    summary = optopsy_simulator._compute_summary(trade_log, float(config.capital))
    equity_curve = trade_log["equity"].copy() if not trade_log.empty else pd.Series(dtype=float, name="equity")
    if not trade_log.empty and "exit_date" in trade_log:
        equity_curve.index = pd.to_datetime(trade_log["exit_date"])
    return SimpleNamespace(trade_log=trade_log, equity_curve=equity_curve, summary=summary)


def _default_option_allocation_fraction(config: OptopsyExecutionConfig) -> float:
    if config.allocation_fraction is not None:
        return max(0.0, min(1.0, float(config.allocation_fraction)))
    return 1.0 / max(1, int(config.max_positions))


def _build_portfolio_fraction_trade_log(
    trades: pd.DataFrame,
    *,
    capital: float,
    multiplier: int,
    max_positions: int,
    default_fraction: float,
    top_k_column: str,
) -> pd.DataFrame:
    columns = [
        "trade_id",
        "underlying_symbol",
        "entry_date",
        "exit_date",
        "expiration",
        "entry_cost",
        "exit_proceeds",
        "quantity",
        "multiplier",
        "allocation_fraction",
        "allocated_notional",
        "pct_change",
        "description",
        "exit_type",
        "dollar_cost",
        "dollar_proceeds",
        "realized_pnl",
        "days_held",
        "cumulative_pnl",
        "equity",
    ]
    if trades.empty:
        return pd.DataFrame(columns=columns)

    work = trades.copy()
    work["entry_session"] = pd.to_datetime(work["entry_date"], errors="coerce").map(_next_business_session)
    work["exit_session"] = pd.to_datetime(work["exit_date"], errors="coerce").map(_previous_business_session)
    work = work.loc[work["entry_session"].notna() & work["exit_session"].notna()].copy()
    work = work.loc[work["exit_session"].ge(work["entry_session"])].copy()
    if work.empty:
        return pd.DataFrame(columns=columns)

    cash = float(capital)
    cumulative_pnl = 0.0
    rows: list[dict[str, Any]] = []
    open_positions: list[dict[str, Any]] = []
    entry_groups = {date: group.copy() for date, group in work.sort_values("entry_session").groupby("entry_session", sort=True)}
    sessions = pd.bdate_range(work["entry_session"].min(), work["exit_session"].max())
    for session in sessions:
        session_ts = pd.Timestamp(session).normalize()
        still_open: list[dict[str, Any]] = []
        for position in open_positions:
            if pd.Timestamp(position["exit_session"]).normalize() > session_ts:
                still_open.append(position)
                continue
            cash += float(position["dollar_proceeds"])
            cumulative_pnl += float(position["realized_pnl"])
            position["cumulative_pnl"] = cumulative_pnl
            position["equity"] = cash + sum(float(open_pos["dollar_cost"]) for open_pos in still_open)
            rows.append(position)
        open_positions = still_open

        day_entries = entry_groups.get(session_ts)
        if day_entries is None or day_entries.empty:
            continue
        for _, trade in day_entries.iterrows():
            if len(open_positions) >= int(max_positions):
                break
            portfolio_value = cash + sum(float(position["dollar_cost"]) for position in open_positions)
            if portfolio_value <= 0 or cash <= 0:
                break
            position = _build_portfolio_fraction_position(
                trade,
                cash=cash,
                portfolio_value=portfolio_value,
                multiplier=multiplier,
                default_fraction=default_fraction,
                top_k_column=top_k_column,
            )
            if position is None:
                continue
            cash -= float(position["dollar_cost"])
            open_positions.append(position)
        if cash + sum(float(position["dollar_cost"]) for position in open_positions) <= 0:
            break
    if open_positions:
        for position in sorted(open_positions, key=lambda item: pd.Timestamp(item["exit_session"])):
            cash += float(position["dollar_proceeds"])
            cumulative_pnl += float(position["realized_pnl"])
            position["cumulative_pnl"] = cumulative_pnl
            position["equity"] = cash
            rows.append(position)

    return pd.DataFrame(rows, columns=columns)


def _build_portfolio_fraction_position(
    trade: pd.Series,
    *,
    cash: float,
    portfolio_value: float,
    multiplier: int,
    default_fraction: float,
    top_k_column: str,
) -> dict[str, Any] | None:
        entry_cost = abs(_nan_float(trade.get("entry_cost")))
        if pd.isna(entry_cost) or entry_cost <= 0:
            return None
        fraction = _allocation_fraction_for_trade(trade, default_fraction=default_fraction, top_k_column=top_k_column)
        allocated_notional = min(max(0.0, portfolio_value * fraction), max(0.0, cash))
        quantity = int(np.floor(allocated_notional / (entry_cost * float(multiplier))))
        if quantity <= 0:
            return None
        lot_size = quantity * int(multiplier)
        exit_proceeds = _nan_float(trade.get("exit_proceeds"))
        realized_pnl = (exit_proceeds - _nan_float(trade.get("entry_cost"))) * lot_size
        return {
            "trade_id": trade.get("trade_id"),
            "underlying_symbol": trade.get("underlying_symbol"),
            "entry_date": pd.Timestamp(trade.get("entry_session")).normalize(),
            "exit_date": pd.Timestamp(trade.get("exit_session")).normalize(),
            "exit_session": pd.Timestamp(trade.get("exit_session")).normalize(),
            "expiration": trade.get("expiration"),
            "entry_cost": trade.get("entry_cost"),
            "exit_proceeds": exit_proceeds,
            "quantity": quantity,
            "multiplier": int(multiplier),
            "allocation_fraction": fraction,
            "allocated_notional": allocated_notional,
            "pct_change": trade.get("pct_change"),
            "description": trade.get("description"),
            "exit_type": trade.get("exit_type", "expiration"),
            "dollar_cost": entry_cost * lot_size,
            "dollar_proceeds": exit_proceeds * lot_size,
            "realized_pnl": realized_pnl,
            "days_held": (pd.Timestamp(trade.get("exit_session")) - pd.Timestamp(trade.get("entry_session"))).days,
            "cumulative_pnl": np.nan,
            "equity": np.nan,
        }


def _next_business_session(value: Any) -> pd.Timestamp | pd.NaT:
    timestamp = pd.Timestamp(value).normalize()
    if pd.isna(timestamp):
        return pd.NaT
    sessions = pd.bdate_range(timestamp, timestamp + pd.Timedelta(days=7))
    return pd.Timestamp(sessions[0]).normalize() if len(sessions) else pd.NaT


def _previous_business_session(value: Any) -> pd.Timestamp | pd.NaT:
    timestamp = pd.Timestamp(value).normalize()
    if pd.isna(timestamp):
        return pd.NaT
    sessions = pd.bdate_range(timestamp - pd.Timedelta(days=7), timestamp)
    return pd.Timestamp(sessions[-1]).normalize() if len(sessions) else pd.NaT


def _allocation_fraction_for_trade(row, *, default_fraction: float, top_k_column: str) -> float:
    if row is not None and top_k_column and top_k_column in row.index:
        top_k = pd.to_numeric(row[top_k_column], errors="coerce")
        if pd.notna(top_k) and float(top_k) > 0:
            return max(0.0, min(1.0, 1.0 / float(top_k)))
    return max(0.0, min(1.0, float(default_fraction)))


def _selected_basket_to_optopsy_raw(selected: pd.DataFrame) -> pd.DataFrame:
    if selected.empty:
        return pd.DataFrame()
    required = [
        "symbol",
        "trade_id",
        "entry_date",
        "option_exit_date",
        "expiration",
        "entry_mid",
        "exit_mid",
        "option_type",
        "strike",
        "basket_weight",
    ]
    missing = [col for col in required if col not in selected.columns]
    if missing:
        raise ValueError(f"selected option basket missing Optopsy raw columns: {missing}")
    rows = []
    for trade_id, group in selected.groupby("trade_id", sort=False):
        work = group.copy()
        weights = pd.to_numeric(work["basket_weight"], errors="coerce").fillna(0.0).clip(lower=0.0)
        total = float(weights.sum())
        if total <= 0:
            continue
        weights = weights / total
        entry = pd.to_numeric(work["entry_mid"], errors="coerce")
        exit_ = pd.to_numeric(work["exit_mid"], errors="coerce")
        valid = entry.gt(0) & exit_.ge(0) & weights.gt(0)
        if not valid.any():
            continue
        work = work.loc[valid].copy()
        weights = weights.loc[valid]
        entry = entry.loc[valid]
        exit_ = exit_.loc[valid]
        total_entry = float((weights * entry).sum())
        total_exit = float((weights * exit_).sum())
        if total_entry <= 0:
            continue
        row = {
            "underlying_symbol": str(work["symbol"].iloc[0]).upper(),
            "quote_date": pd.to_datetime(work["entry_date"].iloc[0], errors="coerce").normalize(),
            "quote_date_entry": pd.to_datetime(work["entry_date"].iloc[0], errors="coerce").normalize(),
            "_early_exit_date": pd.to_datetime(work["option_exit_date"].iloc[0], errors="coerce").normalize(),
            "expiration": pd.to_datetime(work["expiration"]).min().normalize(),
            "total_entry_cost": total_entry,
            "total_exit_proceeds": total_exit,
            "pct_change": total_exit / total_entry - 1.0,
            "exit_type": "basket_signal_exit",
        }
        for optional_col in ("trade_id", "top_k"):
            if optional_col in work.columns:
                row[optional_col] = work[optional_col].iloc[0]
        for idx, leg in enumerate(work.head(4).itertuples(index=False), start=1):
            row[f"option_type_leg{idx}"] = getattr(leg, "option_type")
            row[f"strike_leg{idx}"] = getattr(leg, "strike")
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["quote_date", "underlying_symbol"]).reset_index(drop=True)


def _selected_options_to_optopsy_raw(selected: pd.DataFrame) -> pd.DataFrame:
    if selected.empty:
        return pd.DataFrame()
    out = selected.copy()
    required = ["symbol", "entry_date", "option_exit_date", "expiration", "entry_mid", "exit_mid", "option_type", "strike"]
    missing = [col for col in required if col not in out.columns]
    if missing:
        raise ValueError(f"selected option trades missing Optopsy raw columns: {missing}")
    out["quote_date"] = pd.to_datetime(out["entry_date"], errors="coerce").dt.normalize()
    out["quote_date_entry"] = out["quote_date"]
    out["_early_exit_date"] = pd.to_datetime(out["option_exit_date"], errors="coerce").dt.normalize()
    out["expiration"] = pd.to_datetime(out["expiration"], errors="coerce").dt.normalize()
    out["underlying_symbol"] = out["symbol"].astype(str).str.upper()
    out["option_type"] = out["option_type"].astype(str).str.lower()
    out["strike"] = pd.to_numeric(out["strike"], errors="coerce")
    out["entry"] = pd.to_numeric(out["entry_mid"], errors="coerce")
    out["exit"] = pd.to_numeric(out["exit_mid"], errors="coerce")
    out["pct_change"] = out["exit"] / out["entry"].replace(0, np.nan) - 1.0
    out["exit_type"] = np.where(out["_early_exit_date"].lt(out["expiration"]), "signal_exit", "expiration")
    cols = [
        "underlying_symbol",
        "quote_date",
        "quote_date_entry",
        "_early_exit_date",
        "expiration",
        "option_type",
        "strike",
        "entry",
        "exit",
        "pct_change",
        "exit_type",
    ]
    for optional_col in ("trade_id", "top_k"):
        if optional_col in out.columns:
            cols.append(optional_col)
    return out.loc[
        out["underlying_symbol"].ne("")
        & out["quote_date"].notna()
        & out["_early_exit_date"].notna()
        & out["expiration"].notna()
        & out["entry"].gt(0)
        & out["exit"].ge(0)
        & out["option_type"].isin(["call", "put"]),
        cols,
    ].sort_values(["quote_date", "underlying_symbol", "expiration", "option_type", "strike"]).reset_index(drop=True)


def _row_feature_payload(row) -> dict[str, float]:
    names = getattr(row, "_fields", ())
    payload = {}
    for name in names:
        if name.startswith("_"):
            continue
        value = getattr(row, name)
        if isinstance(value, (int, float, np.integer, np.floating)) or pd.isna(value):
            payload[name] = _nan_float(value)
    return payload


def _choose_top_per_trade(frame: pd.DataFrame, score_col: str) -> pd.DataFrame:
    if frame.empty or score_col not in frame.columns:
        return pd.DataFrame()
    frame = frame.loc[pd.to_numeric(frame[score_col], errors="coerce").notna()].copy()
    if frame.empty:
        return pd.DataFrame()
    return (
        frame.sort_values(["trade_id", score_col, "option_return"], ascending=[True, False, False])
        .groupby("trade_id", as_index=False, sort=False)
        .head(1)
        .reset_index(drop=True)
    )


def _summarize_selected(name: str, frame: pd.DataFrame) -> dict[str, float | int | str]:
    if frame.empty:
        return {"selector": name, "trades": 0}
    returns = pd.to_numeric(frame["option_return"], errors="coerce")
    return {
        "selector": name,
        "trades": int(len(frame)),
        "symbols": int(frame["symbol"].nunique()),
        "mean_return": float(returns.mean()),
        "median_return": float(returns.median()),
        "win_rate": float((returns > 0).mean()),
        "avg_spread_pct": float(pd.to_numeric(frame["spread_pct"], errors="coerce").mean()),
        "avg_abs_moneyness": float(pd.to_numeric(frame["abs_moneyness"], errors="coerce").mean()),
        "avg_dte": float(pd.to_numeric(frame["dte"], errors="coerce").mean()),
    }


def _summarize_basket(name: str, frame: pd.DataFrame) -> dict[str, float | int | str]:
    if frame.empty:
        return {"selector": name, "trades": 0}
    grouped = frame.copy()
    grouped["basket_weight"] = pd.to_numeric(grouped["basket_weight"], errors="coerce").fillna(0.0)
    grouped["weighted_return"] = grouped["basket_weight"] * pd.to_numeric(grouped["option_return"], errors="coerce").fillna(0.0)
    trade_returns = grouped.groupby("trade_id", dropna=False)["weighted_return"].sum()
    return {
        "selector": name,
        "trades": int(trade_returns.shape[0]),
        "symbols": int(grouped["symbol"].nunique()),
        "legs": int(len(grouped)),
        "avg_legs_per_trade": float(grouped.groupby("trade_id").size().mean()),
        "mean_return": float(trade_returns.mean()),
        "median_return": float(trade_returns.median()),
        "win_rate": float((trade_returns > 0).mean()),
        "avg_spread_pct": float(pd.to_numeric(grouped["spread_pct"], errors="coerce").mean()),
        "avg_abs_moneyness": float(pd.to_numeric(grouped["abs_moneyness"], errors="coerce").mean()),
        "avg_dte": float(pd.to_numeric(grouped["dte"], errors="coerce").mean()),
    }


def _split_option_panel(option_panel: pd.DataFrame, split: SharedSplitConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    if option_panel.empty:
        return pd.DataFrame(), pd.DataFrame()
    dates = pd.to_datetime(option_panel["entry_date"], errors="coerce").dt.normalize()
    return option_panel.loc[dates.le(split.insample_end_ts)].copy(), option_panel.loc[dates.ge(split.oos_start_ts)].copy()


def _normalize_oracle_trades(frame: pd.DataFrame) -> pd.DataFrame:
    return _normalize_trade_windows(frame)


def _normalize_trade_windows(frame: pd.DataFrame) -> pd.DataFrame:
    return normalize_report_trade_windows(frame)


def _load_price_frames(
    warehouse,
    symbols: tuple[str, ...],
    *,
    start: str,
    end: str | None,
    adjustment: str = "splits_and_dividends",
) -> dict[str, pd.DataFrame]:
    frames = {}
    for symbol in symbols:
        try:
            frame = warehouse.read_prices(
                symbol,
                provider="fmp",
                start=start,
                end=end,
                adjustment=adjustment,
            ).copy()
        except TypeError:
            if adjustment != "splits_and_dividends":
                continue
            frame = warehouse.read_prices(symbol, provider="fmp", start=start, end=end).copy()
        if frame.empty:
            continue
        frame.index = pd.to_datetime(frame.index, errors="coerce").normalize()
        frames[str(symbol).upper()] = frame.sort_index()
    return frames


def _build_analysis(
    config: OracleOptionExperimentConfig,
    *,
    oracle_trades: pd.DataFrame,
    option_panel: pd.DataFrame,
    train_panel: pd.DataFrame,
    eval_panel: pd.DataFrame,
    selector_summary: pd.DataFrame,
    optopsy_summary: pd.DataFrame,
    source_family_summary: pd.DataFrame,
    feature_coverage: pd.DataFrame,
    metrics: dict[str, float],
) -> str:
    lines = [
        "## Written Analysis",
        "",
        f"- Equity trade windows generated: {len(oracle_trades):,}.",
        f"- Shared split: in-sample <= {config.split.insample_end_ts.date()}, out-of-sample >= {config.split.oos_start_ts.date()}.",
        f"- Option retrieval produced {len(option_panel):,} contract rows across {option_panel['trade_id'].nunique() if not option_panel.empty else 0:,} trade windows.",
        f"- Train rows: {len(train_panel):,}; eval rows: {len(eval_panel):,}.",
    ]
    if len(config.symbols) <= 25:
        lines.append("- This universe size is best treated as a smoke/performance run, not as enough evidence for final data-science conclusions.")
    if metrics:
        lines.append(
            f"- Option ranker metrics: train_r2={metrics.get('train_r2', np.nan):.3f}, "
            f"eval_r2={metrics.get('eval_r2', np.nan):.3f}, eval_mae={metrics.get('eval_mae', np.nan):.3f}."
        )
        if "option_chain_read_count" in metrics:
            lines.append(
                f"- ThetaData retrieval cache: reads={metrics.get('option_chain_read_count', 0):.0f}, "
                f"cache_hits={metrics.get('option_chain_cache_hits', 0):.0f}, "
                f"unique_cached_chains={metrics.get('option_chain_cache_size', 0):.0f}."
            )
        if "affordability_checked_count" in metrics:
            lines.append(
                f"- Affordability filter: checked={metrics.get('affordability_checked_count', 0):.0f}, "
                f"filtered={metrics.get('affordability_filtered_count', 0):.0f}, "
                f"min_contracts={config.retrieval.min_affordable_contracts}."
            )
    option_feature_cols = [
        "liquidity_score",
        "delta",
        "abs_delta",
        "gamma",
        "theta",
        "vega",
        "rho",
        "iv",
        "iv_expiration_z",
        "theta_to_mid",
        "vega_to_mid",
    ]
    available_option_features = [
        col
        for col in option_feature_cols
        if col in train_panel.columns and pd.to_numeric(train_panel[col], errors="coerce").notna().any()
    ]
    missing_greeks = [
        col
        for col in ("delta", "gamma", "theta", "vega", "rho", "iv")
        if col not in train_panel.columns or pd.to_numeric(train_panel[col], errors="coerce").notna().sum() == 0
    ]
    lines.append(
        "- ThetaData option feature columns available in this run: "
        + (", ".join(available_option_features) if available_option_features else "none")
        + "."
    )
    if missing_greeks:
        lines.append(
            "- Greeks/IV missing from the cached chains for this run: "
            + ", ".join(missing_greeks)
            + ". The feature-engineering API will include them automatically when present in ThetaData storage."
        )
    if feature_coverage is not None and not feature_coverage.empty:
        all_coverage = feature_coverage.loc[feature_coverage["split"].astype(str).eq("all")].copy()
        greek_coverage = all_coverage.loc[all_coverage["feature"].isin(["delta", "gamma", "theta", "vega", "rho", "iv", "implied_volatility"])]
        if not greek_coverage.empty:
            summary = ", ".join(
                f"{row.feature}={float(row.non_null_rate):.1%}"
                for row in greek_coverage.itertuples(index=False)
            )
            lines.append(f"- Greeks/IV feature coverage: {summary}.")
    for row in selector_summary.itertuples(index=False):
        trades = int(getattr(row, "trades", 0))
        if trades == 0:
            lines.append(f"- {row.selector}: no eval trades.")
        else:
            lines.append(
                f"- {row.selector}: {trades:,} eval trades, "
                f"mean={row.mean_return:.2%}, median={row.median_return:.2%}, win_rate={row.win_rate:.2%}."
            )
    if not optopsy_summary.empty:
        lines.append("- Portfolio execution is simulated by the third-party Optopsy engine; Quant Orchestrator only adapts selected contracts into Optopsy's raw-trade schema.")
        for row in optopsy_summary.itertuples(index=False):
            closed = int(getattr(row, "closed_trades", 0))
            final_equity = getattr(row, "final_equity", np.nan)
            total_return = getattr(row, "total_return", np.nan)
            lines.append(
                f"- Optopsy {row.selector}: closed_trades={closed:,}, "
                f"final_equity={float(final_equity):,.2f}, total_return={float(total_return):.2%}."
            )
    if source_family_summary is not None and not source_family_summary.empty:
        top_sources = source_family_summary.head(5)
        lines.append("- Largest source-family eval coverage:")
        for row in top_sources.itertuples(index=False):
            source_name = getattr(row, "strategy_source", None) or getattr(row, "source_family", None) or "unknown"
            lines.append(
                f"  - {source_name}: option_rows={int(getattr(row, 'option_rows', 0)):,}, "
                f"eval_trade_windows={int(getattr(row, 'eval_trade_windows', 0)):,}."
            )
    lines.append("- These artifacts are experiment products, not permanent Quant Warehouse source-of-truth tables.")
    return "\n".join(lines)


def _build_execution_analysis(
    config: OracleOptionExperimentConfig,
    *,
    trade_windows: pd.DataFrame,
    selected_option_trades: pd.DataFrame,
    selector_summary: pd.DataFrame,
    optopsy_summary: pd.DataFrame,
    trade_status: pd.DataFrame,
    metrics: dict[str, float],
) -> str:
    lines = [
        "## Written Analysis",
        "",
        f"- Equity trade windows supplied: {len(trade_windows):,}.",
        f"- Fast option execution selected/priced {len(selected_option_trades):,} option trades.",
        "- Selection mode: entry-date chain only; buy calls for long equity trades and buy puts for short equity trades.",
        "- Exit mode: selected contract path is loaded forward until the equity exit date or option expiration settlement.",
    ]
    if not trade_status.empty and "status" in trade_status.columns:
        counts = trade_status["status"].value_counts()
        status_text = ", ".join(f"{status}={int(count)}" for status, count in counts.items())
        lines.append(f"- Per-trade status counts: {status_text}.")
    if metrics:
        lines.append(
            f"- Runtime: elapsed={metrics.get('elapsed_seconds', np.nan):.2f}s, "
            f"entry_chain_reads={metrics.get('option_chain_read_count', 0):.0f}, "
            f"quote_chain_reads={metrics.get('option_quote_chain_read_count', 0):.0f}, "
            f"contract_path_reads={metrics.get('contract_path_read_count', 0):.0f}."
        )
    for row in selector_summary.itertuples(index=False):
        trades = int(getattr(row, "trades", 0))
        if trades == 0:
            continue
        lines.append(
            f"- {row.selector}: {trades:,} trades, "
            f"mean={row.mean_return:.2%}, median={row.median_return:.2%}, win_rate={row.win_rate:.2%}."
        )
    if not optopsy_summary.empty:
        for row in optopsy_summary.itertuples(index=False):
            closed = int(getattr(row, "closed_trades", 0))
            final_equity = getattr(row, "final_equity", np.nan)
            total_return = getattr(row, "total_return", np.nan)
            lines.append(
                f"- Optopsy {row.selector}: closed_trades={closed:,}, "
                f"final_equity={float(final_equity):,.2f}, total_return={float(total_return):.2%}."
            )
    lines.append("- These artifacts are experiment products, not permanent Quant Warehouse source-of-truth tables.")
    return "\n".join(lines)


def _log_mlflow(config: OracleOptionExperimentConfig, artifact_paths: dict[str, Path], metrics: dict[str, float]) -> str | None:
    tracker = get_tracker(tracking_uri=config.mlflow_tracking_uri)
    tags = {
        "quant_orchestrator.kind": "options_experiment",
        "quant_orchestrator.stage": "oracle_option_retrieval_ranking",
    }
    with tracker.start_run(name=config.experiment_name, experiment=config.mlflow_experiment, tags=tags) as run:
        tracker.log_params(_config_dict(config))
        tracker.log_metrics(
            {
                key: float(value)
                for key, value in metrics.items()
                if isinstance(value, (int, float, np.integer, np.floating)) and pd.notna(value)
            }
        )
        for name, path in artifact_paths.items():
            tracker.log_artifact(str(path), artifact_path=name)
        return getattr(getattr(run, "info", None), "run_id", None)


def _config_dict(config: OracleOptionExperimentConfig) -> dict[str, Any]:
    out = asdict(config)
    out["split"] = asdict(config.split)
    out["retrieval"] = asdict(config.retrieval)
    out["execution"] = asdict(config.execution)
    out["mv_basket"] = asdict(config.mv_basket)
    return out


def _prepare_quant_warehouse_import(path: str | None) -> None:
    if not path:
        return
    import sys

    cleaned = str(Path(path).expanduser().resolve())
    sys.path[:] = [entry for entry in sys.path if str(Path(entry).expanduser().resolve()) != cleaned]
    sys.path.insert(0, cleaned)
    for module_name in list(sys.modules):
        if module_name == "quant_warehouse" or module_name.startswith("quant_warehouse."):
            del sys.modules[module_name]


def _warehouse_imports():
    from quant_warehouse import Warehouse
    from quant_warehouse.platforms.data_providers.fmp.target_engineering import LabelBuildSpec, build_trade_results
    from quant_warehouse.platforms.data_providers.thetadata.feature_engineering import (
        build_option_contract_features,
        option_ranker_feature_columns,
    )
    from quant_warehouse.platforms.data_providers.thetadata.target_engineering import build_option_mean_variance_labels
    from quant_warehouse.platforms.data_providers.thetadata.options import read_option_chain_arctic, option_chain_coverage

    return (
        Warehouse,
        LabelBuildSpec,
        build_trade_results,
        read_option_chain_arctic,
        option_chain_coverage,
        build_option_contract_features,
        option_ranker_feature_columns,
        build_option_mean_variance_labels,
    )


def _nan_float(value: Any) -> float:
    numeric = pd.to_numeric(value, errors="coerce")
    return np.nan if pd.isna(numeric) else float(numeric)
