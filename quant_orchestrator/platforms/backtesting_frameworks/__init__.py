"""Built-in backtest engine providers."""

from quant_orchestrator.platforms.backtesting_frameworks.nautilus.provider import nautilus_provider
from quant_orchestrator.platforms.backtesting_frameworks.optopsy.provider import optopsy_provider
from quant_orchestrator.platforms.backtesting_frameworks.pandas.provider import pandas_provider
from quant_orchestrator.platforms.backtesting_frameworks.zipline.provider import zipline_provider

__all__ = ["nautilus_provider", "optopsy_provider", "pandas_provider", "zipline_provider"]
