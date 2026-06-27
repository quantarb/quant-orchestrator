"""Quant Orchestrator platform extension spine."""

from quant_orchestrator.platform.contracts import (
    BacktestEngine,
    BacktestingFramework,
    MLFramework,
    ProviderManifest,
)
from quant_orchestrator.platform.registry import ProviderRegistry, registry

__all__ = [
    "BacktestEngine",
    "BacktestingFramework",
    "MLFramework",
    "ProviderManifest",
    "ProviderRegistry",
    "registry",
]
