"""NautilusTrader backtest engine provider."""

from quant_orchestrator.platforms.backtesting_frameworks.nautilus.provider import (
    NautilusBacktestEngine,
    nautilus_provider,
)
from quant_orchestrator.platforms.backtesting_frameworks.nautilus.reporting_adapter import (
    NautilusReport,
    build_nautilus_report,
)
from quant_orchestrator.platforms.backtesting_frameworks.nautilus.runner import run_nautilus_signal_strategy
from quant_orchestrator.platforms.backtesting_frameworks.nautilus.sma_crossover import run_sma_crossover_backtest

__all__ = [
    "NautilusBacktestEngine",
    "nautilus_provider",
    "NautilusReport",
    "build_nautilus_report",
    "run_nautilus_signal_strategy",
    "run_sma_crossover_backtest",
]
