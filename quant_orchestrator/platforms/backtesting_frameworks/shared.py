from __future__ import annotations

from typing import Any

import pandas as pd

from quant_warehouse import Warehouse
from quant_warehouse.feature_engineering import compute_features_worldclass

MAG7_SYMBOLS: tuple[str, ...] = ("AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA", "TSLA")
OHLCV_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")


def normalize_session_label(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    return ts.normalize()


def load_signal_frame(
    symbol: str,
    *,
    provider: str = "yfinance",
    start: str | None = "2020-01-01",
    end: str | None = None,
) -> pd.DataFrame:
    warehouse = Warehouse()
    prices = warehouse.read_prices(symbol, provider=provider, start=start, end=end)
    if prices.empty:
        raise ValueError(f"No prices found for {symbol.upper()} from Quant Warehouse")

    frame = prices.rename(columns=str.lower).copy()
    frame = frame.loc[:, list(OHLCV_COLUMNS)]
    frame.index = pd.DatetimeIndex(frame.index)
    if frame.index.tz is not None:
        frame.index = frame.index.tz_convert(None)
    frame = frame.sort_index()

    features = compute_features_worldclass(frame.copy())
    required = ["SMA50", "SMA200"]
    missing = [column for column in required if column not in features.columns]
    if missing:
        raise ValueError(
            f"Quant Warehouse feature output for {symbol.upper()} is missing required columns: {missing}"
        )

    frame = features.loc[:, list(OHLCV_COLUMNS) + required].copy()
    frame["signal"] = (frame["SMA50"] > frame["SMA200"]).astype(float)
    frame["sma_50"] = frame["SMA50"]
    frame["sma_200"] = frame["SMA200"]
    frame = frame.dropna(subset=["sma_50", "sma_200"]).copy()
    frame["signal"] = frame["signal"].fillna(0.0).astype(int)
    if frame.empty:
        raise ValueError(
            f"{symbol.upper()} does not have enough rows for the 50/200 SMA crossover warmup window"
        )
    return frame


def load_price_frame(
    symbol: str,
    *,
    provider: str = "yfinance",
    start: str | None = "2020-01-01",
    end: str | None = None,
) -> pd.DataFrame:
    warehouse = Warehouse()
    prices = warehouse.read_prices(symbol, provider=provider, start=start, end=end)
    if prices.empty:
        raise ValueError(f"No prices found for {symbol.upper()} from Quant Warehouse")

    frame = prices.rename(columns=str.lower).copy()
    frame = frame.loc[:, list(OHLCV_COLUMNS)]
    frame.index = pd.DatetimeIndex(frame.index)
    if frame.index.tz is not None:
        frame.index = frame.index.tz_convert(None)
    return frame.sort_index()
