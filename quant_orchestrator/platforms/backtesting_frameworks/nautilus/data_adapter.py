from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from quant_orchestrator.platforms.backtesting_frameworks.shared import (
    OHLCV_COLUMNS,
    normalize_session_label,
)


@dataclass(frozen=True)
class NautilusInMemoryData:
    instrument: Any
    venue: Any
    bar_type: Any
    bars: list[Any]
    trade_size: int
    signal_map: dict[pd.Timestamp, bool]


def build_nautilus_in_memory_data(
    frame: pd.DataFrame,
    *,
    symbol: str,
    capital_base: float,
) -> NautilusInMemoryData:
    from nautilus_trader.model.data import BarType
    from nautilus_trader.model.identifiers import Venue
    from nautilus_trader.persistence.wranglers import BarDataWrangler
    from nautilus_trader.test_kit.providers import TestInstrumentProvider

    instrument = TestInstrumentProvider.equity(symbol=symbol.upper())
    venue = Venue(str(instrument.id.venue))
    bar_type = BarType.from_str(f"{instrument.id}-1-DAY-LAST-EXTERNAL")
    bars = BarDataWrangler(bar_type, instrument).process(frame.loc[:, list(OHLCV_COLUMNS)].copy())
    signal_map = {
        normalize_session_label(date): bool(signal)
        for date, signal in frame["signal"].items()
    }
    trade_size = max(1, int((capital_base * 0.25) / float(frame["close"].iloc[0])))
    return NautilusInMemoryData(
        instrument=instrument,
        venue=venue,
        bar_type=bar_type,
        bars=bars,
        trade_size=trade_size,
        signal_map=signal_map,
    )
