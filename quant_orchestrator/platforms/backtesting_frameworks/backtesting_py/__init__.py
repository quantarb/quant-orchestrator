"""backtesting.py backtest engine provider."""

from quant_orchestrator.platforms.backtesting_frameworks.backtesting_py.provider import (
    BacktestingPyBacktestEngine,
    backtesting_py_provider,
)
from quant_orchestrator.platforms.backtesting_frameworks.backtesting_py.reporting_adapter import (
    BacktestingPyReport,
    build_backtesting_py_report,
)
from quant_orchestrator.platforms.backtesting_frameworks.backtesting_py.sma_crossover import (
    SmaCrossoverSpec,
    build_backtesting_strategy,
    build_sma_crossover_frame,
    run_sma_crossover_backtest,
)

__all__ = [
    "BacktestingPyBacktestEngine",
    "backtesting_py_provider",
    "BacktestingPyReport",
    "build_backtesting_py_report",
    "SmaCrossoverSpec",
    "build_backtesting_strategy",
    "build_sma_crossover_frame",
    "run_sma_crossover_backtest",
]
