from __future__ import annotations

from quant_orchestrator.platforms.backtesting_frameworks import (
    nautilus_provider,
    optopsy_provider,
    pandas_provider,
    zipline_provider,
)
from quant_orchestrator.platforms.ml_frameworks import (
    sklearn_provider,
    torch_provider,
    transformers_provider,
)
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
    ):
        registry.register(provider)


register_builtin_providers()
