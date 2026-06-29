from __future__ import annotations

from decimal import Decimal
from time import perf_counter

import pandas as pd

from quant_orchestrator.platforms.backtesting_frameworks.nautilus.reporting_adapter import (
    build_nautilus_report,
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
    from nautilus_trader.backtest.engine import BacktestEngine
    from nautilus_trader.config import BacktestEngineConfig, LoggingConfig, StrategyConfig
    from nautilus_trader.model.currencies import USD
    from nautilus_trader.model.data import BarType
    from nautilus_trader.model.enums import AccountType, OmsType, OrderSide, TimeInForce
    from nautilus_trader.model.identifiers import Venue
    from nautilus_trader.model.objects import Money, Quantity
    from nautilus_trader.persistence.wranglers import BarDataWrangler
    from nautilus_trader.test_kit.providers import TestInstrumentProvider
    from nautilus_trader.trading.strategy import Strategy

    started = perf_counter()
    frame = build_sma_crossover_frame(prices, fast_window=fast_window, slow_window=slow_window)
    if len(frame) <= slow_window:
        raise ValueError(f"Need more than {slow_window} rows for the Nautilus SMA example.")

    trade_size = max(1, int((capital_base * 0.25) / float(frame["close"].iloc[0])))

    class SmaCrossConfig(StrategyConfig, frozen=True):
        instrument_id: object
        bar_type: object
        trade_size: Decimal
        signal_map: dict

    class SmaCross(Strategy):
        def __init__(self, config: SmaCrossConfig):
            super().__init__(config)
            self.is_long = False

        def on_start(self) -> None:
            self.subscribe_bars(self.config.bar_type)

        def on_bar(self, bar) -> None:
            bullish = self.config.signal_map.get(normalize_session_label(bar.ts_event), False)
            if bullish == self.is_long:
                return

            side = OrderSide.BUY if bullish else OrderSide.SELL
            order = self.order_factory.market(
                instrument_id=self.config.instrument_id,
                order_side=side,
                quantity=Quantity.from_int(int(self.config.trade_size)),
                time_in_force=TimeInForce.IOC,
            )
            self.submit_order(order)
            self.is_long = bullish

    instrument = TestInstrumentProvider.equity(symbol=symbol.upper())
    venue = Venue(str(instrument.id.venue))
    bar_type = BarType.from_str(f"{instrument.id}-1-DAY-LAST-EXTERNAL")
    bars = BarDataWrangler(bar_type, instrument).process(
        prices.loc[:, ["open", "high", "low", "close", "volume"]].copy(),
    )
    signal_map = {normalize_session_label(date): bool(signal) for date, signal in frame["signal"].items()}

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
                trade_size=Decimal(trade_size),
                signal_map=signal_map,
            ),
        ),
    )

    engine.run()
    fills_report = engine.trader.generate_order_fills_report()
    engine.dispose()

    equity = _equity_from_fills(prices=frame, fills=fills_report, capital_base=capital_base)
    report = build_nautilus_report(
        fills_report,
        equity,
        symbol=symbol,
        elapsed_seconds=perf_counter() - started,
    )
    summary = report.summary
    summary["fast_window"] = fast_window
    summary["slow_window"] = slow_window
    summary["native_fills"] = int(len(fills_report))
    summary["native_last_value"] = float(equity.iloc[-1])
    return fills_report, summary, report.equity_curve


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
        fill_date = normalize_session_label(fill["ts_last"])
        fills_by_date.setdefault(fill_date, []).append(fill)

    for date, row in prices.iterrows():
        normalized = normalize_session_label(date)
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
