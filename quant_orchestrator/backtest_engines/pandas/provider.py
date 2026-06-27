"""Compatibility import for the Pandas backtesting framework provider."""

from quant_orchestrator.platforms.backtesting_frameworks.pandas.provider import (
    PandasBacktestEngine,
    pandas_provider,
)

__all__ = ["PandasBacktestEngine", "pandas_provider"]
