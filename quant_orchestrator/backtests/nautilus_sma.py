from __future__ import annotations

import pandas as pd

from quant_orchestrator.data import load_ohlcv
from quant_orchestrator.backtests.research import run_sma_crossover_backtest


def run_nautilus_backtest(
    *,
    symbol: str,
    provider: str,
    start: str | None,
    end: str | None,
    fast_window: int,
    slow_window: int,
    capital_base: float,
) -> pd.DataFrame:
    prices = load_ohlcv(symbol, provider=provider, start=start, end=end)
    result = run_sma_crossover_backtest(
        prices,
        framework="nautilus",
        symbol=symbol,
        fast_window=fast_window,
        slow_window=slow_window,
        capital_base=capital_base,
    )
    return result.summary
