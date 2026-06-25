from __future__ import annotations

from decimal import Decimal
from time import perf_counter

import pandas as pd

from quant_orchestrator.data import load_ohlcv
from quant_orchestrator.strategy import fixed_trade_size, summarize_backtest


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
    from nautilus_trader.backtest.engine import BacktestEngine
    from nautilus_trader.config import BacktestEngineConfig, LoggingConfig, StrategyConfig
    from nautilus_trader.indicators.averages import SimpleMovingAverage
    from nautilus_trader.model.currencies import USD
    from nautilus_trader.model.data import BarType
    from nautilus_trader.model.enums import AccountType, OmsType, OrderSide, TimeInForce
    from nautilus_trader.model.identifiers import Venue
    from nautilus_trader.model.objects import Money, Quantity
    from nautilus_trader.persistence.wranglers import BarDataWrangler
    from nautilus_trader.test_kit.providers import TestInstrumentProvider
    from nautilus_trader.trading.strategy import Strategy

    started = perf_counter()
    if fast_window >= slow_window:
        raise ValueError("fast_window must be less than slow_window")

    prices = load_ohlcv(symbol, provider=provider, start=start, end=end)
    if len(prices) <= slow_window:
        raise ValueError(f"Need more than {slow_window} rows for the Nautilus SMA example.")
    trade_size = fixed_trade_size(prices["close"], capital_base)

    class SmaCrossConfig(StrategyConfig, frozen=True):
        instrument_id: object
        bar_type: object
        fast_window: int
        slow_window: int
        trade_size: Decimal

    class SmaCross(Strategy):
        def __init__(self, config: SmaCrossConfig):
            super().__init__(config)
            self.fast = SimpleMovingAverage(config.fast_window)
            self.slow = SimpleMovingAverage(config.slow_window)
            self.is_long = False

        def on_start(self) -> None:
            self.register_indicator_for_bars(self.config.bar_type, self.fast)
            self.register_indicator_for_bars(self.config.bar_type, self.slow)
            self.subscribe_bars(self.config.bar_type)

        def on_bar(self, bar) -> None:
            if not self.fast.initialized or not self.slow.initialized:
                return
            bullish = self.fast.value > self.slow.value
            if bullish == self.is_long:
                return

            target_side = OrderSide.BUY if bullish else OrderSide.SELL
            order = self.order_factory.market(
                instrument_id=self.config.instrument_id,
                order_side=target_side,
                quantity=Quantity.from_int(int(self.config.trade_size)),
                time_in_force=TimeInForce.IOC,
            )
            self.submit_order(order)
            self.is_long = bullish

    instrument = TestInstrumentProvider.equity(symbol=symbol.upper())
    venue = Venue(str(instrument.id.venue))
    bar_type = BarType.from_str(f"{instrument.id}-1-DAY-LAST-EXTERNAL")
    bars = BarDataWrangler(bar_type, instrument).process(prices)

    engine = BacktestEngine(
        config=BacktestEngineConfig(logging=LoggingConfig(log_level="ERROR")),
    )
    engine.add_venue(
        venue=venue,
        oms_type=OmsType.NETTING,
        account_type=AccountType.CASH,
        starting_balances=[Money(capital_base, USD)],
        base_currency=USD,
    )
    engine.add_instrument(instrument)
    engine.add_data(bars)
    engine.add_strategy(
        SmaCross(
            SmaCrossConfig(
                instrument_id=instrument.id,
                bar_type=bar_type,
                fast_window=fast_window,
                slow_window=slow_window,
                trade_size=Decimal(trade_size),
            ),
        ),
    )
    engine.run()
    fills_report = engine.trader.generate_order_fills_report()
    engine.dispose()
    equity = _equity_from_fills(prices=prices, fills=fills_report, capital_base=capital_base)
    report_equity = equity.iloc[slow_window - 1 :]
    return summarize_backtest(
        framework="nautilus",
        symbol=symbol,
        equity=report_equity,
        elapsed_seconds=perf_counter() - started,
        bars=len(report_equity),
        trades=len(fills_report),
    )


def _equity_from_fills(
    *,
    prices: pd.DataFrame,
    fills: pd.DataFrame,
    capital_base: float,
) -> pd.Series:
    cash = float(capital_base)
    position = 0.0
    values = []
    fills_by_date: dict[pd.Timestamp, list[pd.Series]] = {}

    for _, fill in fills.iterrows():
        fill_date = pd.Timestamp(fill["ts_last"]).tz_convert("UTC").normalize()
        fills_by_date.setdefault(fill_date, []).append(fill)

    for date, row in prices.iterrows():
        normalized = pd.Timestamp(date).tz_convert("UTC").normalize()
        for fill in fills_by_date.get(normalized, []):
            quantity = float(fill["filled_qty"])
            price = float(fill["avg_px"])
            if str(fill["side"]) == "BUY":
                cash -= quantity * price
                position += quantity
            else:
                cash += quantity * price
                position -= quantity
        values.append(cash + position * float(row["close"]))

    return pd.Series(values, index=prices.index, name="portfolio_value")
