"""Built-in backtest engine providers."""

from quant_orchestrator.platforms.backtesting_frameworks.nautilus.provider import nautilus_provider
from quant_orchestrator.platforms.backtesting_frameworks.backtesting_py.provider import (
    backtesting_py_provider,
)
from quant_orchestrator.platforms.backtesting_frameworks.lean.provider import lean_provider
from quant_orchestrator.platforms.backtesting_frameworks.optopsy.provider import optopsy_provider
from quant_orchestrator.platforms.backtesting_frameworks.panel_weight.provider import (
    panel_weight_provider,
)
from quant_orchestrator.platforms.backtesting_frameworks.zipline.provider import zipline_provider

__all__ = [
    "backtesting_py_provider",
    "lean_provider",
    "nautilus_provider",
    "optopsy_provider",
    "panel_weight_provider",
    "zipline_provider",
]
