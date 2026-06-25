from __future__ import annotations

from time import perf_counter

import pandas as pd

from quant_orchestrator.data import load_ohlcv
from quant_orchestrator.strategy import fixed_size_sma_equity, fixed_trade_size, summarize_backtest


def run_pandas_backtest(
    *,
    symbols: list[str],
    provider: str,
    start: str | None,
    end: str | None,
    fast_window: int,
    slow_window: int,
    capital_base: float,
) -> pd.DataFrame:
    rows = []
    for symbol in symbols:
        started = perf_counter()
        prices = load_ohlcv(symbol, provider=provider, start=start, end=end)
        trade_size = fixed_trade_size(prices["close"], capital_base)
        equity, trades = fixed_size_sma_equity(
            prices,
            fast_window=fast_window,
            slow_window=slow_window,
            capital_base=capital_base,
            trade_size=trade_size,
        )
        report_equity = equity.iloc[slow_window - 1 :]
        report = summarize_backtest(
            framework="pandas",
            symbol=symbol,
            equity=report_equity,
            elapsed_seconds=perf_counter() - started,
            bars=len(report_equity),
            trades=trades,
        )
        rows.append(report.iloc[0].to_dict())

    return pd.DataFrame(rows)
