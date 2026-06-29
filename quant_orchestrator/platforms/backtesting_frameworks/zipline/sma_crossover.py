from __future__ import annotations

from time import perf_counter

import pandas as pd

from quant_orchestrator.platforms.backtesting_frameworks.zipline.data_adapter import (
    build_zipline_in_memory_data,
)
from quant_orchestrator.platforms.backtesting_frameworks.zipline.reporting_adapter import (
    build_zipline_report,
)
from quant_orchestrator.platforms.backtesting_frameworks.shared import (
    build_sma_crossover_frame,
    normalize_session_label,
)


def run_sma_crossover_backtest(
    prices: pd.DataFrame,
    *,
    symbol: str,
    fast_window: int,
    slow_window: int,
    capital_base: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    from zipline.algorithm import TradingAlgorithm
    from zipline.api import order_target, record, symbol as zipline_symbol

    started = perf_counter()
    frame = build_sma_crossover_frame(prices, fast_window=fast_window, slow_window=slow_window)
    if len(frame) <= slow_window:
        raise ValueError(f"Need more than {slow_window} rows for the Zipline SMA example.")

    adapter = build_zipline_in_memory_data(frame, symbol=symbol, capital_base=capital_base)

    def initialize(context, **kwargs):
        context.asset = zipline_symbol(symbol.upper())
        context.is_long = False

    def handle_data(context, data):
        bullish = adapter.signal_map.get(normalize_session_label(context.get_datetime()), False)
        if bullish and not context.is_long:
            order_target(context.asset, adapter.trade_size)
            context.is_long = True
        elif not bullish and context.is_long:
            order_target(context.asset, 0)
            context.is_long = False
        record(signal=float(bullish))

    algo = TradingAlgorithm(
        sim_params=adapter.sim_params,
        data_portal=adapter.data_portal,
        asset_finder=adapter.asset_finder,
        initialize=initialize,
        handle_data=handle_data,
        capital_base=capital_base,
        benchmark_returns=adapter.benchmark_returns,
    )
    perf = algo.run()

    report = build_zipline_report(perf, symbol=symbol, elapsed_seconds=perf_counter() - started)
    summary = report.summary
    summary["fast_window"] = fast_window
    summary["slow_window"] = slow_window
    summary["native_last_value"] = float(perf["portfolio_value"].iloc[-1])
    return perf, summary, report.equity_curve
