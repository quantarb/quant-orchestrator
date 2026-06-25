from __future__ import annotations

from pathlib import Path
from typing import Any

from dagster import Definitions, ScheduleDefinition, get_dagster_logger, job, op

from quant_orchestrator.experiments import (
    TradingAppExperimentSpec,
    UniverseSplit,
    WindowSpec,
)
from quant_orchestrator.trading_app_experiments import run_trading_app_experiment


TRADING_APP_CONFIG_SCHEMA = {
    "name": str,
    "provider": str,
    "framework": str,
    "top_k": int,
    "gross_exposure": float,
    "capital_base": float,
    "prediction_artifact": str,
    "train_model": bool,
    "train_start": str,
    "train_end": str,
    "backtest_start": str,
    "end": str,
    "max_symbols": int,
    "train_symbols": str,
    "backtest_symbols": str,
    "window_mode": str,
    "window_step": str,
    "train_lookback": str,
    "test_length": str,
    "tracking_uri": str,
    "track": bool,
}

DEFAULT_TRADING_APP_RUN_CONFIG = {
    "ops": {
        "run_trading_app_scheduled": {
            "config": {
                "name": "trading-app-equity-daily",
                "provider": "yfinance",
                "framework": "zipline",
                "top_k": 40,
                "gross_exposure": 0.75,
                "capital_base": 100_000.0,
                "prediction_artifact": "",
                "train_model": False,
                "train_start": "2020-01-01",
                "train_end": "2020-12-31",
                "backtest_start": "2021-01-01",
                "end": "",
                "max_symbols": 0,
                "train_symbols": "",
                "backtest_symbols": "",
                "window_mode": "fixed",
                "window_step": "30D",
                "train_lookback": "365D",
                "test_length": "30D",
                "tracking_uri": "",
                "track": True,
            }
        }
    }
}


@op(config_schema=TRADING_APP_CONFIG_SCHEMA)
def run_trading_app_scheduled(context) -> str:
    config = context.op_config
    logger = get_dagster_logger()
    spec = _trading_app_spec_from_config(config)
    result = run_trading_app_experiment(
        spec,
        track=bool(config["track"]),
        tracking_uri=_optional(config["tracking_uri"]),
    )
    output_dir = Path("artifacts/dagster/trading_app") / spec.name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "latest_summary.csv"
    result.to_csv(output_path, index=False)
    logger.info("Wrote trading app summary to %s", output_path)
    return str(output_path)


@job
def trading_app_experiment_job() -> None:
    run_trading_app_scheduled()


daily_trading_app_schedule = ScheduleDefinition(
    job=trading_app_experiment_job,
    cron_schedule="0 6 * * *",
    name="daily_trading_app_experiment",
    run_config=DEFAULT_TRADING_APP_RUN_CONFIG,
)

weekday_trading_app_schedule = ScheduleDefinition(
    job=trading_app_experiment_job,
    cron_schedule="30 6 * * 1-5",
    name="weekday_trading_app_experiment",
    run_config=DEFAULT_TRADING_APP_RUN_CONFIG,
)


defs = Definitions(
    jobs=[trading_app_experiment_job],
    schedules=[daily_trading_app_schedule, weekday_trading_app_schedule],
)


def _trading_app_spec_from_config(config: dict[str, Any]) -> TradingAppExperimentSpec:
    max_symbols = int(config["max_symbols"])
    prediction_artifact = _optional(config["prediction_artifact"])
    return TradingAppExperimentSpec(
        name=config["name"],
        provider=config["provider"],
        framework=config["framework"],
        top_k=int(config["top_k"]),
        gross_exposure=float(config["gross_exposure"]),
        capital_base=float(config["capital_base"]),
        prediction_artifact=Path(prediction_artifact) if prediction_artifact else None,
        train_model=bool(config["train_model"]),
        train_start=config["train_start"],
        train_end=config["train_end"],
        backtest_start=config["backtest_start"],
        end=_optional(config["end"]),
        max_symbols=max_symbols if max_symbols > 0 else None,
        universe=UniverseSplit(
            train=_symbols(config["train_symbols"]),
            backtest=_symbols(config["backtest_symbols"]),
        ),
        window=WindowSpec(
            mode=config["window_mode"],
            train_start=config["train_start"],
            train_end=config["train_end"],
            test_start=config["backtest_start"],
            test_end=_optional(config["end"]),
            step=config["window_step"],
            train_lookback=config["train_lookback"],
            test_length=config["test_length"],
        ),
    )


def _optional(value: str) -> str | None:
    cleaned = str(value).strip()
    return cleaned or None


def _symbols(value: str) -> tuple[str, ...]:
    return tuple(symbol.strip().upper() for symbol in value.split(",") if symbol.strip())
