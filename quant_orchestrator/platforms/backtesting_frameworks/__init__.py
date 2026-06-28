"""Built-in backtest engine providers."""

from quant_orchestrator.platforms.backtesting_frameworks.nautilus.provider import nautilus_provider
from quant_orchestrator.platforms.backtesting_frameworks.backtesting_py.provider import (
    backtesting_py_provider,
)
from quant_orchestrator.platforms.backtesting_frameworks.optopsy.provider import optopsy_provider
from quant_orchestrator.platforms.backtesting_frameworks.pandas.provider import pandas_provider
from quant_orchestrator.platforms.backtesting_frameworks.zipline.provider import zipline_provider

__all__ = [
    "backtesting_py_provider",
    "nautilus_provider",
    "optopsy_provider",
    "pandas_provider",
    "zipline_provider",
]
