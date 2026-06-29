from __future__ import annotations

import pandas as pd

from quant_orchestrator.platforms.backtesting_frameworks.nautilus.runner import (
    run_nautilus_signal_strategy,
)
from quant_orchestrator.platforms.backtesting_frameworks.shared import (
    build_sma_crossover_frame,
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
        raise ValueError(f"Need more than {slow_window} rows for the Nautilus SMA example.")

    fills_report, summary, equity = run_nautilus_signal_strategy(
        frame,
        symbol=symbol,
        capital_base=capital_base,
    )
    summary["fast_window"] = fast_window
    summary["slow_window"] = slow_window
    summary["native_fills"] = int(len(fills_report))
    summary["native_last_value"] = float(equity.iloc[-1])
    return fills_report, summary, equity
