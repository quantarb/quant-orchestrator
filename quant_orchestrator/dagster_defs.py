from __future__ import annotations

from pathlib import Path

from dagster import Definitions, get_dagster_logger, job, op

from quant_orchestrator.backtests import run_framework_comparison


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


defs = Definitions(jobs=[backtest_framework_comparison_job])


def _optional(value: str) -> str | None:
    cleaned = str(value).strip()
    return cleaned or None


def _csv_items(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())

