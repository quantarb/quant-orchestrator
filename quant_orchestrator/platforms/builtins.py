from __future__ import annotations

from quant_orchestrator.platforms.backtesting_frameworks import (
    backtesting_py_provider,
    lean_provider,
    nautilus_provider,
    optimal_trader_provider,
    panel_weight_provider,
    zipline_provider,
)
from quant_orchestrator.platforms.ml_frameworks import (
    sentence_transformers_provider,
    sklearn_provider,
    torch_provider,
    transformers_provider,
)
from quant_orchestrator.platforms.registry import registry


def register_builtin_providers() -> None:
    for provider in (
        sklearn_provider,
        sentence_transformers_provider,
        torch_provider,
        transformers_provider,
        backtesting_py_provider,
        lean_provider,
        zipline_provider,
        nautilus_provider,
        panel_weight_provider,
        optimal_trader_provider,
    ):
        registry.register(provider)


register_builtin_providers()
