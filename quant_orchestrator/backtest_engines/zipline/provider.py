"""Compatibility import for the Zipline backtesting framework provider."""

from quant_orchestrator.platforms.backtesting_frameworks.zipline.provider import (
    ZiplineBacktestEngine,
    zipline_provider,
)

__all__ = ["ZiplineBacktestEngine", "zipline_provider"]
