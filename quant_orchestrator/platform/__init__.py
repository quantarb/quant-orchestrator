"""Quant Orchestrator platform extension spine."""

from quant_orchestrator.platform.contracts import (
    BacktestingFramework,
    MLFramework,
    ProviderManifest,
)
from quant_orchestrator.platform.registry import ProviderRegistry, registry

__all__ = [
    "BacktestingFramework",
    "MLFramework",
    "ProviderManifest",
    "ProviderRegistry",
    "registry",
]
