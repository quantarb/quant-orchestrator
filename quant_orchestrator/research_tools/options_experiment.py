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
    metrics: dict[str, float]
    artifact_paths: dict[str, Path]
    analysis_markdown: str
    elapsed_seconds: float


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
    price_frames = _load_price_frames(warehouse, eligible_symbols, start=config.price_start, end=config.price_end)
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
    price_frames = _load_price_frames(warehouse, eligible_symbols, start=config.price_start, end=config.price_end)
    return _run_option_experiment_from_trades(
        config=config,
        started=started,
        coverage=coverage,
        trades=normalized_trades,
        price_frames=price_frames,
        read_option_chain_arctic=read_option_chain_arctic,
        build_option_contract_features=build_option_contract_features,
        option_ranker_feature_columns=option_ranker_feature_columns,
        build_option_mean_variance_labels=build_option_mean_variance_labels,
    )


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
        read_option_chain_arctic=read_option_chain_arctic,
        build_option_contract_features=build_option_contract_features,
    )
    option_panel = _build_option_panel(oracle_trades, retriever)
    if config.mv_basket.enabled:
        option_panel = _add_mean_variance_basket_labels(
            option_panel,
            config.mv_basket,
            build_option_mean_variance_labels=build_option_mean_variance_labels,
        )
    train_panel, eval_panel = _split_option_panel(option_panel, config.split)
    model, model_metrics, eval_scored = _train_option_ranker(
        train_panel,
        eval_panel,
        config=config,
        option_ranker_feature_columns=option_ranker_feature_columns,
    )
    mv_model, mv_metrics, eval_scored = _train_mv_basket_ranker(
        train_panel,
        eval_scored,
        config=config,
        option_ranker_feature_columns=option_ranker_feature_columns,
    )
    model_payload = {"single_leg_ranker": model, "mv_basket_ranker": mv_model}
    model_metrics.update(mv_metrics)
    model_metrics.update(retriever.metrics())
    selector_summary, symbol_summary, selected_by_selector = _selector_summaries(eval_scored, mv_basket_config=config.mv_basket)
    source_family_summary = _source_family_diagnostics(eval_scored, selected_by_selector)
    feature_coverage = _option_feature_coverage(option_panel, train_panel=train_panel, eval_panel=eval_scored)
    model_metrics.update(_option_feature_coverage_metrics(feature_coverage))
    optopsy_summary = _run_optopsy_selector_backtests(selected_by_selector, config.execution)
    elapsed_seconds = perf_counter() - started
    model_metrics["elapsed_seconds"] = float(elapsed_seconds)
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
        read_option_chain_arctic,
        build_option_contract_features,
    ):
        self.config = config
        self.execution = execution or OptopsyExecutionConfig()
        self.price_frames = price_frames
        self.read_option_chain_arctic = read_option_chain_arctic
        self.build_option_contract_features = build_option_contract_features
        self._chain_cache: dict[tuple[str, pd.Timestamp], pd.DataFrame] = {}
        self._underlying_cache: dict[tuple[str, pd.Timestamp], float | None] = {}
        self.chain_read_count = 0
        self.chain_cache_hits = 0
        self.underlying_lookup_count = 0
        self.underlying_cache_hits = 0
        self.affordability_checked_count = 0
        self.affordability_filtered_count = 0

    def retrieve(self, trade: pd.Series) -> pd.DataFrame:
        symbol = str(trade["symbol"]).upper()
        side = str(trade["side"]).lower()
        entry_date = pd.Timestamp(trade["entry_date"]).normalize()
        equity_exit_date = pd.Timestamp(trade["exit_date"]).normalize()
        realized_holding_days = max(0, int((equity_exit_date - entry_date).days))
        spot = self._underlying_close(symbol, entry_date)
        entry_chain = self._load_day_chain(symbol, entry_date)
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
            exit_row = self._price_exit(symbol, equity_exit_date, row, option_type)
            if exit_row is None:
                continue
            exit_snapshot_date, exit_mid = exit_row
            entry_mid = float(row.mid)
            entry_price = float(getattr(row, "ask", entry_mid))
            exit_price = float(exit_mid)
            row_payload = {
                    "trade_id": trade.get("trade_id", f"{symbol}|{entry_date.date()}|{side}"),
                    "symbol": symbol,
                    "side": side,
                    "equity_signal_side": side,
                    "entry_date": entry_date,
                    "equity_exit_date": equity_exit_date,
                    "option_exit_date": exit_snapshot_date,
                    "expiration": pd.Timestamp(row.expiration).normalize(),
                    "contract_symbol": row.contract_symbol,
                    "option_type": option_type,
                    "option_action": "buy_call" if option_type == "call" else "buy_put",
                    "strike": float(row.strike),
                    "entry_mid": entry_mid,
                    "exit_mid": float(exit_mid),
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "option_pnl": exit_price - entry_price,
                    "return_denominator": entry_price,
                    "option_return": (exit_price - entry_price) / entry_price if entry_price > 0 else np.nan,
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
            exit_prices = self._price_exit_prices(symbol, equity_exit_date, row, option_type)
            if exit_prices is None:
                continue
            exit_snapshot_date, exit_bid, exit_ask, exit_mid = exit_prices
            entry_bid = _nan_float(getattr(row, "bid", np.nan))
            entry_ask = _nan_float(getattr(row, "ask", np.nan))
            if option_action.startswith("buy_"):
                entry_price = entry_ask
                exit_price = exit_bid
                pnl = exit_price - entry_price
                denominator = entry_price
            else:
                entry_price = entry_bid
                exit_price = exit_ask
                pnl = entry_price - exit_price
                denominator = float(row.strike) if option_action == "sell_put" else float(spot)
            if not np.isfinite(entry_price) or not np.isfinite(exit_price) or not np.isfinite(denominator) or denominator <= 0:
                continue
            row_payload = {
                "trade_id": trade.get("trade_id", f"{symbol}|{entry_date.date()}|{side}"),
                "symbol": symbol,
                "side": side,
                "equity_signal_side": side,
                "entry_date": entry_date,
                "equity_exit_date": equity_exit_date,
                "option_exit_date": exit_snapshot_date,
                "expiration": pd.Timestamp(row.expiration).normalize(),
                "contract_symbol": row.contract_symbol,
                "option_type": option_type,
                "option_action": option_action,
                "strike": float(row.strike),
                "entry_mid": _nan_float(getattr(row, "mid", np.nan)),
                "exit_mid": float(exit_mid),
                "entry_bid": entry_bid,
                "entry_ask": entry_ask,
                "exit_bid": float(exit_bid),
                "exit_ask": float(exit_ask),
                "entry_price": float(entry_price),
                "exit_price": float(exit_price),
                "option_pnl": float(pnl),
                "return_denominator": float(denominator),
                "option_return": float(pnl) / float(denominator),
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
        prices = self._price_exit_prices(symbol, equity_exit_date, row, option_type)
        if prices is None:
            return None
        exit_snapshot_date, _bid, _ask, mid = prices
        return exit_snapshot_date, mid

    def _price_exit_prices(self, symbol: str, equity_exit_date: pd.Timestamp, row, option_type: str) -> tuple[pd.Timestamp, float, float, float] | None:
        expiration = pd.Timestamp(row.expiration).normalize()
        target_exit = min(pd.Timestamp(equity_exit_date).normalize(), expiration)
        exit_snapshot_date, exit_chain = self._load_nearest_exit_chain(symbol, target_exit)
        if not exit_chain.empty:
            match = exit_chain.loc[exit_chain["contract_symbol"].astype(str).eq(str(row.contract_symbol))]
            if not match.empty:
                bid = pd.to_numeric(match.iloc[0].get("bid"), errors="coerce")
                ask = pd.to_numeric(match.iloc[0].get("ask"), errors="coerce")
                mid = pd.to_numeric(match.iloc[0].get("mid"), errors="coerce")
                if pd.isna(mid) and pd.notna(bid) and pd.notna(ask):
                    mid = (float(bid) + float(ask)) / 2.0
                if pd.notna(mid):
                    bid = mid if pd.isna(bid) else bid
                    ask = mid if pd.isna(ask) else ask
                    return pd.Timestamp(exit_snapshot_date).normalize(), float(bid), float(ask), float(mid)
        if target_exit >= expiration:
            close = self._underlying_close(symbol, target_exit)
            if close is None:
                return None
            intrinsic = max(close - float(row.strike), 0.0) if option_type == "call" else max(float(row.strike) - close, 0.0)
            return target_exit, float(intrinsic), float(intrinsic), float(intrinsic)
        return None

    def _load_nearest_exit_chain(self, symbol: str, target_date: pd.Timestamp) -> tuple[pd.Timestamp | None, pd.DataFrame]:
        target = pd.Timestamp(target_date).normalize()
        for date in pd.bdate_range(target, target - pd.Timedelta(days=self.config.exit_lookback_days), freq="-1B"):
            chain = self._load_day_chain(symbol, pd.Timestamp(date))
            if not chain.empty:
                return pd.Timestamp(date).normalize(), chain
        return None, pd.DataFrame()

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
        featured = self.build_option_contract_features(
            chain,
            underlying_price=self._underlying_close(key[0], key[1]),
            target_dte=self.config.target_dte,
            compute_model_greeks=False,
        ).df
        self._chain_cache[key] = featured.copy()
        return featured

    def _underlying_close(self, symbol: str, date: pd.Timestamp) -> float | None:
        key = (str(symbol).upper(), pd.Timestamp(date).normalize())
        if key in self._underlying_cache:
            self.underlying_cache_hits += 1
            return self._underlying_cache[key]
        self.underlying_lookup_count += 1
        prices = self.price_frames.get(str(symbol).upper())
        if prices is None or prices.empty or "close" not in prices.columns:
            self._underlying_cache[key] = None
            return None
        eligible = prices.loc[prices.index <= pd.Timestamp(date).normalize()]
        if eligible.empty:
            self._underlying_cache[key] = None
            return None
        value = pd.to_numeric(eligible["close"].iloc[-1], errors="coerce")
        result = None if pd.isna(value) else float(value)
        self._underlying_cache[key] = result
        return result

    def metrics(self) -> dict[str, float]:
        return {
            "option_chain_read_count": float(self.chain_read_count),
            "option_chain_cache_hits": float(self.chain_cache_hits),
            "option_chain_cache_size": float(len(self._chain_cache)),
            "underlying_lookup_count": float(self.underlying_lookup_count),
            "underlying_cache_hits": float(self.underlying_cache_hits),
            "underlying_cache_size": float(len(self._underlying_cache)),
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


def _build_option_panel(oracle_trades: pd.DataFrame, retriever: _OptionRetriever) -> pd.DataFrame:
    frames = []
    for trade in oracle_trades.itertuples(index=False):
        frame = retriever.retrieve(pd.Series(trade._asdict()))
        if frame is not None and not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    panel = pd.concat(frames, ignore_index=True)
    panel["rank_y"] = panel.groupby("trade_id")["option_return"].rank(method="average", pct=True, ascending=True)
    return panel


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
        "eval_mae": float(mean_absolute_error(eval_scored[target], eval_scored["pred_return"])),
        "eval_r2": float(r2_score(eval_scored[target], eval_scored["pred_return"])) if len(eval_scored) > 1 else np.nan,
        "numeric_feature_count": float(len(numeric)),
        "categorical_feature_count": float(len(categorical)),
        "ranker_model_count": 1.0 + (1.0 if rank_model is not None else 0.0),
    }
    if rank_model is not None and "rank_y" in eval_scored.columns:
        eval_rank_target = pd.to_numeric(eval_scored["rank_y"], errors="coerce")
        metrics.update(
            {
                "rank_train_mae": float(mean_absolute_error(train_panel["rank_y"], rank_train_pred)),
                "rank_train_r2": float(r2_score(train_panel["rank_y"], rank_train_pred)) if len(train_panel) > 1 else np.nan,
                "rank_eval_mae": float(mean_absolute_error(eval_rank_target, eval_scored["pred_rank_score"])),
                "rank_eval_r2": float(r2_score(eval_rank_target, eval_scored["pred_rank_score"])) if len(eval_scored) > 1 else np.nan,
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
    metrics = {
        "mv_train_mae": float(mean_absolute_error(train_panel[target], train_pred)),
        "mv_train_r2": float(r2_score(train_panel[target], train_pred)) if len(train_panel) > 1 else np.nan,
        "mv_eval_mae": float(mean_absolute_error(eval_scored[target], eval_scored["pred_mv_weight"])) if target in eval_scored else np.nan,
        "mv_eval_r2": float(r2_score(eval_scored[target], eval_scored["pred_mv_weight"])) if target in eval_scored and len(eval_scored) > 1 else np.nan,
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
    selected_rule_atm = _choose_top_per_trade(fixed_work, "fixed_near_atm_score")
    selected_mv_model = _choose_weighted_basket_per_trade(
        eval_panel,
        "pred_mv_weight",
        selector_name="model_mv_basket",
        max_legs=mv_basket_config.max_legs if mv_basket_config else None,
        min_weight=mv_basket_config.min_predicted_weight if mv_basket_config else 0.0,
    )
    selector_summary = pd.DataFrame(
        [
            _summarize_selected("rule_atm_90d", selected_rule_atm),
            _summarize_selected("model_ranker", selected_model),
            _summarize_basket("model_mv_basket", selected_mv_model),
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
    return (
        selector_summary,
        symbol_summary,
        {
            "rule_atm_90d": selected_rule_atm,
            "model_ranker": selected_model,
            "model_mv_basket": selected_mv_model,
        },
    )


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
    if frame.empty:
        return frame
    out = frame.copy()
    required = {"symbol", "side", "entry_date", "exit_date"}
    missing = required - set(out.columns)
    if missing:
        raise KeyError(f"trade windows missing required columns: {sorted(missing)}")
    out["symbol"] = out["symbol"].astype(str).str.upper()
    out["side"] = out["side"].astype(str).str.lower().replace(
        {
            "buy": "long",
            "sell": "short",
            "long_call": "long",
            "long_put": "short",
            "call": "long",
            "put": "short",
        }
    )
    out["entry_date"] = pd.to_datetime(out["entry_date"], errors="coerce").dt.normalize()
    out["exit_date"] = pd.to_datetime(out["exit_date"], errors="coerce").dt.normalize()
    out = out.loc[
        out["symbol"].ne("")
        & out["side"].isin(["long", "short"])
        & out["entry_date"].notna()
        & out["exit_date"].notna()
        & out["exit_date"].gt(out["entry_date"])
    ].copy()
    classifier_cols = {"strategy_source", "source", "family"}
    if classifier_cols.intersection(out.columns):
        if "top_k" not in out.columns:
            raise KeyError("classifier trade windows must include top_k for option allocation sizing")
        top_k = pd.to_numeric(out["top_k"], errors="coerce")
        if top_k.isna().any() or top_k.le(0).any():
            raise ValueError("classifier trade windows must include positive top_k values for option allocation sizing")
    if "trade_id" not in out.columns:
        out["trade_id"] = out.apply(
            lambda row: f"{row['symbol']}|{pd.Timestamp(row['entry_date']).date()}|{row['side']}",
            axis=1,
        )
    return out.sort_values(["entry_date", "symbol", "side"]).reset_index(drop=True)


def _load_price_frames(warehouse, symbols: tuple[str, ...], *, start: str, end: str | None) -> dict[str, pd.DataFrame]:
    frames = {}
    for symbol in symbols:
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
