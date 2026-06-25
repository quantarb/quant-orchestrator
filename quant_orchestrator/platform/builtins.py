from __future__ import annotations

from quant_orchestrator.backtest_engines import (
    nautilus_provider,
    optopsy_provider,
    pandas_provider,
    zipline_provider,
)
from quant_orchestrator.brokers import alpaca_provider, robinhood_provider
from quant_orchestrator.experiment_trackers import mlflow_provider
from quant_orchestrator.ml_frameworks import sklearn_provider, torch_provider, transformers_provider
from quant_orchestrator.platform.registry import registry


def register_builtin_providers() -> None:
    for provider in (
        sklearn_provider,
        torch_provider,
        transformers_provider,
        pandas_provider,
        zipline_provider,
        nautilus_provider,
        optopsy_provider,
        alpaca_provider,
        robinhood_provider,
        mlflow_provider,
    ):
        registry.register(provider)


register_builtin_providers()
