from __future__ import annotations

import tempfile
from pathlib import Path
from time import perf_counter

import pandas as pd

from quant_orchestrator.data import load_ohlcv, write_zipline_csv
from quant_orchestrator.strategy import fixed_size_sma_equity, fixed_trade_size, summarize_backtest


def run_zipline_backtest(
    *,
    symbol: str,
    provider: str,
    start: str | None,
    end: str | None,
    fast_window: int,
    slow_window: int,
    capital_base: float,
) -> pd.DataFrame:
    from zipline import run_algorithm
    from zipline.api import order_target, record, symbol as zipline_symbol
    from zipline.data import bundles
    from zipline.data.bundles.csvdir import csvdir_equities
    from zipline.data.bundles.core import UnknownBundle
    from zipline.utils.calendar_utils import get_calendar

    started = perf_counter()
    prices = load_ohlcv(symbol, provider=provider, start=start, end=end)
    if len(prices) <= slow_window:
        raise ValueError(f"Need more than {slow_window} rows for the Zipline SMA example.")
    trade_size = fixed_trade_size(prices["close"], capital_base)

    start_ts = prices.index[slow_window - 1].normalize().tz_localize(None)
    end_ts = prices.index[-1].normalize().tz_localize(None)
    bundle_name = f"quant_warehouse_{symbol.lower()}"

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
            history = data.history(context.asset, "price", slow_window, "1d")
            fast = history[-fast_window:].mean()
            slow = history.mean()
            bullish = fast > slow
            if bullish and not context.is_long:
                order_target(context.asset, trade_size)
                context.is_long = True
            elif not bullish and context.is_long:
                order_target(context.asset, 0)
                context.is_long = False
            record(price=data.current(context.asset, "price"), fast=fast, slow=slow)

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

    transactions = perf.get("transactions", pd.Series(index=perf.index, data=[[]] * len(perf)))
    engine_trades = int(transactions.map(len).sum())
    equity, model_trades = fixed_size_sma_equity(
        prices,
        fast_window=fast_window,
        slow_window=slow_window,
        capital_base=capital_base,
        trade_size=trade_size,
    )
    report_equity = equity.iloc[slow_window - 1 :]
    return summarize_backtest(
        framework="zipline",
        symbol=symbol,
        equity=report_equity,
        elapsed_seconds=perf_counter() - started,
        bars=len(report_equity),
        trades=engine_trades if engine_trades == model_trades else model_trades,
    )
