from __future__ import annotations

from quant_orchestrator.platforms.backtesting_frameworks import (
    backtesting_py_provider,
    nautilus_provider,
    optopsy_provider,
    zipline_provider,
)
from quant_orchestrator.platforms.ml_frameworks import (
    sklearn_provider,
    torch_provider,
    transformers_provider,
)
from quant_orchestrator.platforms.registry import registry


def register_builtin_providers() -> None:
    for provider in (
        sklearn_provider,
        torch_provider,
        transformers_provider,
        backtesting_py_provider,
        zipline_provider,
        nautilus_provider,
        optopsy_provider,
    ):
        registry.register(provider)


register_builtin_providers()
