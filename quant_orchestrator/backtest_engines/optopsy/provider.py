"""Compatibility import for the Optopsy backtesting framework provider."""

from quant_orchestrator.platforms.backtesting_frameworks.optopsy.provider import (
    OptopsyBacktestEngine,
    optopsy_provider,
)

__all__ = ["OptopsyBacktestEngine", "optopsy_provider"]
