from __future__ import annotations

from pathlib import Path

from dagster import Definitions, get_dagster_logger, job, op

from quant_orchestrator.backtests import run_framework_comparison
from quant_orchestrator.research_tools import MLTradingExperimentConfig, run_ml_trading_experiment


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
    "top_k_values": str,
    "zipline_max_workers": int,
    "log_mlflow": bool,
    "mlflow_experiment": str,
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
        top_k_values=tuple(int(item) for item in _csv_items(config["top_k_values"])),
        zipline_max_workers=int(config["zipline_max_workers"]),
        log_mlflow=bool(config["log_mlflow"]),
        mlflow_experiment=config["mlflow_experiment"],
    )
    result = run_ml_trading_experiment(experiment_config)
    logger.info(
        "Completed ML trading experiment %s; artifacts=%s; metrics=%s",
        result.config.experiment_name,
        result.artifacts.artifact_uris,
        result.metrics,
    )
    return result.artifacts.run.id


@job
def ml_trading_experiment_job() -> None:
    run_ml_trading_experiment_scheduled()


defs = Definitions(jobs=[backtest_framework_comparison_job, ml_trading_experiment_job])


def _optional(value: str) -> str | None:
    cleaned = str(value).strip()
    return cleaned or None


def _csv_items(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())
