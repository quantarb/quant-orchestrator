"""OpenBB-style provider platforms for Quant Orchestrator."""

from quant_orchestrator.platforms import backtesting_frameworks, ml_frameworks
from quant_orchestrator.platforms.builtins import register_builtin_providers
from quant_orchestrator.platforms.contracts import BacktestingFramework, MLFramework, ProviderManifest
from quant_orchestrator.platforms.registry import ProviderRegistry, registry

__all__ = [
    "BacktestingFramework",
    "MLFramework",
    "ProviderManifest",
    "ProviderRegistry",
    "backtesting_frameworks",
    "ml_frameworks",
    "register_builtin_providers",
    "registry",
]
