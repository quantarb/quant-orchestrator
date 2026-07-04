from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
from dagster import Definitions, DynamicOut, DynamicOutput, Field, Noneable, get_dagster_logger, job, op

from quant_orchestrator.backtests import run_framework_comparison
from quant_orchestrator.research_tools import (
    MLTradingExperimentConfig,
    OptopsyExecutionConfig,
    OptionMvBasketConfig,
    OptionRetrievalConfig,
    OptionWindowBuildConfig,
    OracleOptionExperimentConfig,
    SharedSplitConfig,
    build_option_window_dataset,
    estimate_option_runtime_scaling,
    run_ml_trading_experiment,
    run_trade_window_option_experiment,
)


BACKTEST_FRAMEWORK_COMPARISON_CONFIG_SCHEMA = {
    "name": str,
    "symbol": str,
    "providers": str,
    "frameworks": str,
    "start": str,
    "end": str,
    "fast_window": int,
    "slow_window": int,
    "capital_base": float,
}

ML_TRADING_EXPERIMENT_CONFIG_SCHEMA = {
    "experiment_name": str,
    "mode": str,
    "min_market_cap": int,
    "start_date": str,
    "end_date": str,
    "train_end": str,
    "oos_start": str,
    "score_start": str,
    "top_k_values": str,
    "feature_representations": str,
    "zipline_max_workers": int,
    "run_zipline_backtests": bool,
    "log_mlflow": bool,
    "mlflow_experiment": str,
}

OPTION_WINDOW_EXPERIMENT_CONFIG_SCHEMA = {
    "experiment_name": str,
    "score_artifact_dir": str,
    "window_group": str,
    "symbols": str,
    "price_start": str,
    "price_end": str,
    "insample_end": str,
    "variant": str,
    "top_k": int,
    "entry_threshold": float,
    "exit_threshold": float,
    "top_family_count": int,
    "ranking_framework": str,
    "min_signal_events": int,
    "max_candidates_per_trade": int,
    "log_mlflow": bool,
    "artifact_dir": str,
    "target_trade_windows": int,
    "target_option_rows": int,
    "max_runtime_seconds": float,
}

THETADATA_OPTIONS_BACKFILL_UNIVERSE_CONFIG_SCHEMA = {
    "symbols": Field(str, default_value="", description="Comma-separated symbol override."),
    "symbols_file": Field(str, default_value="", description="Newline-delimited symbol file."),
    "min_market_cap": Field(int, default_value=10_000_000_000),
    "country": Field(str, default_value="US"),
    "exchanges": Field(str, default_value="NASDAQ,NYSE,AMEX"),
    "is_etf": Field(bool, default_value=False),
    "is_fund": Field(bool, default_value=False),
    "is_active": Field(bool, default_value=True),
    "all_share_classes": Field(bool, default_value=False),
    "limit": Field(int, default_value=10_000),
}

THETADATA_OPTIONS_BACKFILL_SYMBOL_CONFIG_SCHEMA = {
    "start_date": Field(str, default_value="2018-01-02"),
    "end_date": Field(str, default_value="2026-06-30"),
    "backfill_window_days": Field(int, default_value=7),
    "fallback_window_days": Field(int, default_value=1),
    "request_sleep": Field(float, default_value=0.0),
    "overwrite": Field(bool, default_value=False),
    "include_non_us": Field(bool, default_value=False),
}


@op(config_schema=BACKTEST_FRAMEWORK_COMPARISON_CONFIG_SCHEMA)
def run_backtest_framework_comparison_scheduled(context) -> str:
    config = context.op_config
    logger = get_dagster_logger()
    result = run_framework_comparison(
        symbol=config["symbol"],
        providers=_csv_items(config["providers"]),
        frameworks=_csv_items(config["frameworks"]),
        start=config["start"],
        end=_optional(config["end"]),
        fast_window=int(config["fast_window"]),
        slow_window=int(config["slow_window"]),
        capital_base=float(config["capital_base"]),
    )
    output_dir = Path("artifacts/dagster/backtest_framework_comparison") / config["name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    result.comparison.to_csv(output_dir / "comparison.csv", index=False)
    result.factor_report.to_csv(output_dir / "factor_report.csv", index=False)
    result.framework_summary.to_csv(output_dir / "framework_summary.csv")
    result.provider_summary.to_csv(output_dir / "provider_summary.csv")
    logger.info("Wrote framework comparison outputs to %s", output_dir)
    return str(output_dir)


@job
def backtest_framework_comparison_job() -> None:
    run_backtest_framework_comparison_scheduled()


@op(config_schema=ML_TRADING_EXPERIMENT_CONFIG_SCHEMA)
def run_ml_trading_experiment_scheduled(context) -> str:
    config = context.op_config
    logger = get_dagster_logger()
    experiment_config = MLTradingExperimentConfig(
        experiment_name=config["experiment_name"],
        mode=config["mode"],
        min_market_cap=int(config["min_market_cap"]),
        start_date=config["start_date"],
        end_date=_optional(config["end_date"]),
        train_end=config["train_end"],
        oos_start=config["oos_start"],
        score_start=_optional(config["score_start"]),
        top_k_values=tuple(int(item) for item in _csv_items(config["top_k_values"])),
        feature_representations=tuple(_csv_items(config["feature_representations"])) or ("raw",),
        zipline_max_workers=int(config["zipline_max_workers"]),
        run_zipline_backtests=bool(config["run_zipline_backtests"]),
        log_mlflow=bool(config["log_mlflow"]),
        mlflow_experiment=config["mlflow_experiment"],
    )
    result = run_ml_trading_experiment(experiment_config)
    logger.info(
        "Completed ML trading experiment %s; mlflow_run_id=%s; metrics=%s",
        result.config.experiment_name,
        result.mlflow_run_id,
        result.metrics,
    )
    return result.mlflow_run_id or ""


@job
def ml_trading_experiment_job() -> None:
    run_ml_trading_experiment_scheduled()


@op(config_schema=OPTION_WINDOW_EXPERIMENT_CONFIG_SCHEMA)
def run_option_window_experiment_scheduled(context) -> str:
    config = context.op_config
    logger = get_dagster_logger()
    score_dir = Path(config["score_artifact_dir"])
    scores = pd.read_csv(score_dir / "strategy_scores.csv")
    summary_path = score_dir / "backtest_summary.csv"
    backtest_summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    window_dataset = build_option_window_dataset(
        scores,
        backtest_summary=backtest_summary,
        config=OptionWindowBuildConfig(
            variant=config["variant"],
            top_k=int(config["top_k"]),
            entry_threshold=float(config["entry_threshold"]),
            exit_threshold=float(config["exit_threshold"]),
            top_family_count=int(config["top_family_count"]),
            ranking_framework=_optional(config["ranking_framework"]),
            min_signal_events=int(config["min_signal_events"]),
        ),
    )
    window_group = config["window_group"]
    if window_group not in window_dataset.windows_by_group:
        raise ValueError(f"unknown window_group {window_group!r}; available={sorted(window_dataset.windows_by_group)}")
    artifact_dir = _optional(config["artifact_dir"])
    experiment_config = OracleOptionExperimentConfig(
        experiment_name=config["experiment_name"],
        symbols=_csv_items(config["symbols"]) or tuple(sorted(scores["symbol"].dropna().astype(str).str.upper().unique())),
        price_start=config["price_start"],
        price_end=_optional(config["price_end"]),
        split=SharedSplitConfig(insample_end=config["insample_end"]),
        retrieval=OptionRetrievalConfig(max_candidates_per_trade=int(config["max_candidates_per_trade"])),
        execution=OptopsyExecutionConfig(),
        mv_basket=OptionMvBasketConfig(),
        artifact_dir=artifact_dir,
        log_mlflow=bool(config["log_mlflow"]),
    )
    result = run_trade_window_option_experiment(experiment_config, window_dataset.windows_by_group[window_group])
    runtime_estimate = estimate_option_runtime_scaling(
        baseline_elapsed_seconds=result.elapsed_seconds,
        baseline_trade_windows=len(result.oracle_trades),
        baseline_option_rows=len(result.option_panel),
        target_trade_windows=int(config["target_trade_windows"]),
        target_option_rows=int(config["target_option_rows"]),
        max_seconds=float(config["max_runtime_seconds"]),
    )
    logger.info(
        "Completed option window experiment %s group=%s windows=%s option_rows=%s elapsed=%.2fs chain_reads=%s cache_hits=%s target_estimate=%.2fs runnable=%s",
        result.config.experiment_name,
        window_group,
        len(result.oracle_trades),
        len(result.option_panel),
        result.elapsed_seconds,
        result.metrics.get("option_chain_read_count"),
        result.metrics.get("option_chain_cache_hits"),
        runtime_estimate.estimated_seconds_conservative,
        runtime_estimate.runnable_within_budget,
    )
    return str(result.artifact_paths["analysis"].parent)


@job
def option_window_experiment_job() -> None:
    run_option_window_experiment_scheduled()


@op(config_schema=THETADATA_OPTIONS_BACKFILL_UNIVERSE_CONFIG_SCHEMA, out=DynamicOut(str))
def resolve_thetadata_options_backfill_symbols(context):
    config = context.op_config
    logger = get_dagster_logger()
    explicit_symbols = _csv_items(config["symbols"])
    if explicit_symbols:
        symbols = tuple(sorted({symbol.upper() for symbol in explicit_symbols}))
    elif _optional(config["symbols_file"]):
        symbol_path = Path(config["symbols_file"]).expanduser()
        if not symbol_path.exists():
            raise FileNotFoundError(f"symbols_file does not exist: {symbol_path}")
        symbols = tuple(
            sorted(
                {
                    line.strip().upper()
                    for line in symbol_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                }
            )
        )
        logger.info("Resolved %s ThetaData backfill symbols from %s", len(symbols), symbol_path)
    else:
        from quant_warehouse.ingest.screener_fetch import ScreenerQuery, fetch_equity_screener

        query = ScreenerQuery(
            provider="fmp",
            mktcap_min=int(config["min_market_cap"]),
            country=_optional(config["country"]),
            exchanges=_csv_items(config["exchanges"]),
            is_etf=bool(config["is_etf"]),
            is_fund=bool(config["is_fund"]),
            is_active=bool(config["is_active"]),
            all_share_classes=bool(config["all_share_classes"]),
            limit=int(config["limit"]),
        )
        frame, source = fetch_equity_screener(query)
        if frame.empty or "symbol" not in frame.columns:
            raise ValueError(f"FMP screener returned no symbols for query={query!r}")
        symbols = tuple(
            sorted(
                frame["symbol"]
                .dropna()
                .astype(str)
                .str.strip()
                .str.upper()
                .loc[lambda values: values.ne("")]
                .unique()
            )
        )
        logger.info("Resolved %s ThetaData backfill symbols from %s", len(symbols), source)

    for symbol in symbols:
        yield DynamicOutput(symbol, mapping_key=_dagster_mapping_key(symbol))


@op(config_schema=THETADATA_OPTIONS_BACKFILL_SYMBOL_CONFIG_SCHEMA)
def backfill_thetadata_options_symbol(context, symbol: str) -> dict[str, object]:
    config = context.op_config
    logger = get_dagster_logger()
    from quant_warehouse.migrate.backfill_thetadata_options import backfill_thetadata_options

    started = time.perf_counter()
    logger.info(
        "Starting full-chain ThetaData options backfill for %s start=%s end=%s window_days=%s overwrite=%s",
        symbol,
        config["start_date"],
        config["end_date"],
        config["backfill_window_days"],
        config["overwrite"],
    )
    summary = backfill_thetadata_options(
        symbols=(symbol,),
        start_date=config["start_date"],
        end_date=_optional(config["end_date"]),
        backfill_window_days=int(config["backfill_window_days"]),
        fallback_window_days=int(config["fallback_window_days"]),
        skip_existing=not bool(config["overwrite"]),
        overwrite=bool(config["overwrite"]),
        request_sleep=float(config["request_sleep"]),
        max_workers=1,
        us_only=not bool(config["include_non_us"]),
    )
    result = (summary.get("results") or [{}])[0]
    if result.get("error"):
        raise RuntimeError(f"ThetaData option backfill failed for {symbol}: {result['error']}")
    elapsed_seconds = time.perf_counter() - started
    summary["elapsed_seconds"] = elapsed_seconds
    logger.info(
        "ThetaData options %s complete: skipped=%s days=%s contracts=%s fetched_rows=%s cached_days=%s elapsed=%.2fs",
        symbol,
        result.get("skipped"),
        result.get("snapshot_days"),
        result.get("contracts_total"),
        result.get("fetched_rows"),
        result.get("cached_days"),
        elapsed_seconds,
    )
    return summary


@op
def summarize_thetadata_options_backfill(summaries: list[dict[str, object]]) -> dict[str, int]:
    totals = {
        "symbols_requested": 0,
        "symbols_completed": 0,
        "symbols_skipped": 0,
        "symbols_failed": 0,
    }
    for summary in summaries:
        for key in totals:
            totals[key] += int(summary.get(key) or 0)
    get_dagster_logger().info("ThetaData options backfill totals: %s", totals)
    return totals


@job
def thetadata_options_backfill_job() -> None:
    symbols = resolve_thetadata_options_backfill_symbols()
    summaries = symbols.map(backfill_thetadata_options_symbol)
    summarize_thetadata_options_backfill(summaries.collect())


defs = Definitions(
    jobs=[
        backtest_framework_comparison_job,
        ml_trading_experiment_job,
        option_window_experiment_job,
        thetadata_options_backfill_job,
    ]
)


def _optional(value: str) -> str | None:
    cleaned = str(value).strip()
    return cleaned or None


def _csv_items(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _dagster_mapping_key(value: str) -> str:
    return "".join(char if char.isalnum() or char == "_" else "_" for char in value.upper())
