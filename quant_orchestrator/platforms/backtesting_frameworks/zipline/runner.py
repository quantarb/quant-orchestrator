from __future__ import annotations

from time import perf_counter

import pandas as pd

from quant_orchestrator.platforms.backtesting_frameworks.shared import normalize_session_label
from quant_orchestrator.platforms.backtesting_frameworks.zipline.data_adapter import (
    build_zipline_in_memory_data,
)
from quant_orchestrator.platforms.backtesting_frameworks.zipline.reporting_adapter import (
    build_zipline_report,
)


def run_zipline_signal_strategy(
    frame: pd.DataFrame,
    *,
    symbol: str,
    capital_base: float,
    signal_column: str = "signal",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Run a long/flat signal strategy in Zipline using in-memory Quant Warehouse data."""
    from zipline.algorithm import TradingAlgorithm
    from zipline.api import order_target, record, symbol as zipline_symbol

    if signal_column not in frame.columns:
        raise KeyError(f"Missing required signal column: {signal_column}")

    started = perf_counter()
    signal_frame = frame.copy()
    if signal_column != "signal":
        signal_frame["signal"] = signal_frame[signal_column]

    adapter = build_zipline_in_memory_data(signal_frame, symbol=symbol, capital_base=capital_base)

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
    return perf, report.summary, report.equity_curve
