from __future__ import annotations

import pandas as pd

from quant_orchestrator.platforms.backtesting_frameworks.shared import (
    build_sma_crossover_frame,
)
from quant_orchestrator.platforms.backtesting_frameworks.zipline.runner import (
    run_zipline_signal_strategy,
)


def run_sma_crossover_backtest(
    prices: pd.DataFrame,
    *,
    symbol: str,
    fast_window: int,
    slow_window: int,
    capital_base: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    frame = build_sma_crossover_frame(prices, fast_window=fast_window, slow_window=slow_window)
    if len(frame) <= slow_window:
        raise ValueError(f"Need more than {slow_window} rows for the Zipline SMA example.")

    perf, summary, equity = run_zipline_signal_strategy(
        frame,
        symbol=symbol,
        capital_base=capital_base,
    )
    summary["fast_window"] = fast_window
    summary["slow_window"] = slow_window
    summary["native_last_value"] = float(perf["portfolio_value"].iloc[-1])
    return perf, summary, equity
