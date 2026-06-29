from __future__ import annotations

import tempfile
from pathlib import Path
from time import perf_counter

import pandas as pd

from quant_orchestrator.data import write_zipline_csv
from quant_orchestrator.platforms.backtesting_frameworks.zipline.reporting_adapter import (
    build_zipline_report,
)
from quant_orchestrator.platforms.backtesting_frameworks.shared import (
    build_sma_crossover_frame,
    normalize_session_label,
)
from quant_orchestrator.strategy import fixed_trade_size


def run_sma_crossover_backtest(
    prices: pd.DataFrame,
    *,
    symbol: str,
    fast_window: int,
    slow_window: int,
    capital_base: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    from zipline import run_algorithm
    from zipline.api import order_target, record, symbol as zipline_symbol
    from zipline.data import bundles
    from zipline.data.bundles.csvdir import csvdir_equities
    from zipline.data.bundles.core import UnknownBundle
    from zipline.utils.calendar_utils import get_calendar

    started = perf_counter()
    frame = build_sma_crossover_frame(prices, fast_window=fast_window, slow_window=slow_window)
    if len(frame) <= slow_window:
        raise ValueError(f"Need more than {slow_window} rows for the Zipline SMA example.")

    trade_size = fixed_trade_size(frame["close"], capital_base)
    start_ts = frame.index[slow_window - 1].normalize().tz_localize(None)
    end_ts = frame.index[-1].normalize().tz_localize(None)
    bundle_name = f"quant_warehouse_{symbol.lower()}"
    signal_map = {
        normalize_session_label(date): bool(signal) for date, signal in frame["signal"].items()
    }

    with tempfile.TemporaryDirectory(prefix="quant-orchestrator-zipline-") as tmp:
        csv_root = Path(tmp) / "csv"
        write_zipline_csv(symbol, prices, csv_root)

        try:
            bundles.unregister(bundle_name)
        except (KeyError, UnknownBundle):
            pass
        bundles.register(
            bundle_name,
            csvdir_equities(["daily"], str(csv_root)),
            calendar_name="XNYS",
        )
        bundles.ingest(bundle_name)

        def initialize(context):
            context.asset = zipline_symbol(symbol.upper())
            context.is_long = False

        def handle_data(context, data):
            bullish = signal_map.get(normalize_session_label(context.get_datetime()), False)
            if bullish and not context.is_long:
                order_target(context.asset, trade_size)
                context.is_long = True
            elif not bullish and context.is_long:
                order_target(context.asset, 0)
                context.is_long = False
            record(price=data.current(context.asset, "price"), signal=float(bullish))

        perf = run_algorithm(
            start=start_ts,
            end=end_ts,
            initialize=initialize,
            handle_data=handle_data,
            capital_base=capital_base,
            data_frequency="daily",
            bundle=bundle_name,
            trading_calendar=get_calendar("XNYS"),
            default_extension=False,
        )

    report = build_zipline_report(perf, symbol=symbol, elapsed_seconds=perf_counter() - started)
    summary = report.summary
    summary["fast_window"] = fast_window
    summary["slow_window"] = slow_window
    summary["native_last_value"] = float(perf["portfolio_value"].iloc[-1])
    return perf, summary, report.equity_curve
