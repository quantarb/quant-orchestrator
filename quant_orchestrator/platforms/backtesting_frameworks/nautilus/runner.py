from __future__ import annotations

from decimal import Decimal
from time import perf_counter

import pandas as pd

from quant_orchestrator.platforms.backtesting_frameworks.nautilus.data_adapter import (
    build_nautilus_in_memory_data,
)
from quant_orchestrator.platforms.backtesting_frameworks.nautilus.reporting_adapter import (
    build_nautilus_report,
)
from quant_orchestrator.platforms.backtesting_frameworks.shared import (
    normalize_session_label,
)


def run_nautilus_signal_strategy(
    frame: pd.DataFrame,
    *,
    symbol: str,
    capital_base: float,
    signal_column: str = "signal",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Run a long/flat signal strategy in Nautilus using in-memory Quant Warehouse data."""
    from nautilus_trader.backtest.engine import BacktestEngine
    from nautilus_trader.config import BacktestEngineConfig, LoggingConfig, StrategyConfig
    from nautilus_trader.model.currencies import USD
    from nautilus_trader.model.enums import AccountType, OmsType, OrderSide, TimeInForce
    from nautilus_trader.model.objects import Money, Quantity
    from nautilus_trader.trading.strategy import Strategy

    if signal_column not in frame.columns:
        raise KeyError(f"Missing required signal column: {signal_column}")

    started = perf_counter()
    signal_frame = frame.copy()
    if signal_column != "signal":
        signal_frame["signal"] = signal_frame[signal_column]

    adapter = build_nautilus_in_memory_data(signal_frame, symbol=symbol, capital_base=capital_base)

    class SignalStrategyConfig(StrategyConfig, frozen=True):
        instrument_id: object
        bar_type: object
        trade_size: Decimal
        signal_map: dict

    class SignalStrategy(Strategy):
        def __init__(self, config: SignalStrategyConfig):
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

    engine = BacktestEngine(
        config=BacktestEngineConfig(
            logging=LoggingConfig(log_level="OFF", bypass_logging=True),
        ),
    )
    engine.add_venue(
        venue=adapter.venue,
        oms_type=OmsType.NETTING,
        account_type=AccountType.CASH,
        starting_balances=[Money(capital_base, USD)],
        base_currency=USD,
    )
    engine.add_instrument(adapter.instrument)
    engine.add_data(adapter.bars)
    engine.add_strategy(
        SignalStrategy(
            SignalStrategyConfig(
                instrument_id=adapter.instrument.id,
                bar_type=adapter.bar_type,
                trade_size=Decimal(adapter.trade_size),
                signal_map=adapter.signal_map,
            ),
        ),
    )

    engine.run()
    fills_report = engine.trader.generate_order_fills_report()
    engine.dispose()

    equity = build_nautilus_equity_curve(prices=signal_frame, fills=fills_report, capital_base=capital_base)
    report = build_nautilus_report(
        fills_report,
        equity,
        symbol=symbol,
        elapsed_seconds=perf_counter() - started,
    )
    return fills_report, report.summary, report.equity_curve


def build_nautilus_equity_curve(
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
