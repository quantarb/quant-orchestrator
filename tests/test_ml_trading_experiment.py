from __future__ import annotations

from quant_orchestrator.dagster_defs import defs
from quant_orchestrator.research_tools import MLTradingExperimentConfig


def test_ml_trading_experiment_config_defaults_to_fast_1t_smoke() -> None:
    config = MLTradingExperimentConfig()

    assert config.experiment_name == "gpu_rf_shared_book_1t"
    assert config.mode == "classifier"
    assert config.min_market_cap == 1_000_000_000_000
    assert config.log_mlflow is True
    assert config.mlflow_experiment == "ml_trading"


def test_dagster_registers_ml_trading_experiment_job() -> None:
    job_def = defs.get_job_def("ml_trading_experiment_job")

    assert job_def.name == "ml_trading_experiment_job"
