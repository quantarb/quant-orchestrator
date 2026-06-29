from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import pandas as pd

from quant_orchestrator.platforms.backtesting_frameworks.backtesting_py.data_adapter import build_backtesting_frame
from quant_orchestrator.platforms.backtesting_frameworks.backtesting_py.reporting_adapter import (
    build_backtesting_py_report,
)
from quant_orchestrator.platforms.backtesting_frameworks.shared import build_sma_crossover_frame as build_shared_sma_crossover_frame


@dataclass(frozen=True)
class SmaCrossoverSpec:
    fast_window: int
    slow_window: int
    capital_fraction: float = 0.25


def build_sma_crossover_frame(
    prices: pd.DataFrame,
    *,
    fast_window: int,
    slow_window: int,
) -> pd.DataFrame:
    return build_shared_sma_crossover_frame(prices, fast_window=fast_window, slow_window=slow_window)


def build_backtesting_strategy(
    *,
    fast_window: int,
    slow_window: int,
    trade_size: int,
):
    from backtesting import Strategy

    class SmaCrossoverStrategy(Strategy):
        def init(self):
            self.trade_size = trade_size

        def next(self):
            bullish = bool(self.data.fast_sma[-1] > self.data.slow_sma[-1])
            if bullish and not self.position:
                self.buy(size=self.trade_size)
            elif not bullish and self.position:
                self.position.close()

    SmaCrossoverStrategy.fast_window = fast_window
    SmaCrossoverStrategy.slow_window = slow_window
    return SmaCrossoverStrategy


def run_sma_crossover_backtest(
    prices: pd.DataFrame,
    *,
    symbol: str = "PORTFOLIO",
    fast_window: int,
    slow_window: int,
    capital_base: float,
):
    from backtesting import Backtest

    frame = build_sma_crossover_frame(prices, fast_window=fast_window, slow_window=slow_window)
    trade_size = max(1, int((capital_base * 0.25) / float(frame["close"].iloc[0])))
    strategy = build_backtesting_strategy(
        fast_window=fast_window,
        slow_window=slow_window,
        trade_size=trade_size,
    )
    bt_frame = build_backtesting_frame(frame)

    started = perf_counter()
    stats = Backtest(
        bt_frame,
        strategy,
        cash=capital_base,
        commission=0.0,
        trade_on_close=False,
        exclusive_orders=True,
    ).run()
    elapsed = perf_counter() - started
    report = build_backtesting_py_report(stats, symbol=symbol, elapsed_seconds=elapsed)
    summary = report.summary
    summary["fast_window"] = fast_window
    summary["slow_window"] = slow_window
    summary["native_return_pct"] = report.native_metrics["return_pct"]
    summary["native_sharpe"] = report.native_metrics["sharpe"]
    summary["native_max_drawdown_pct"] = report.native_metrics["max_drawdown_pct"]
    return stats, summary, report.equity_curve
