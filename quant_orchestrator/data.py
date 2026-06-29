from __future__ import annotations

import pandas as pd
from quant_warehouse import Warehouse

REQUIRED_OHLCV = ("open", "high", "low", "close", "volume")


def load_ohlcv(
    symbol: str,
    *,
    provider: str = "yfinance",
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    warehouse = Warehouse()
    prices = warehouse.read_prices(symbol, provider=provider, start=start, end=end)
    if prices.empty:
        raise ValueError(
            f"No {provider} prices found in Quant Warehouse for {symbol}. "
            f"Refresh first with: quant-warehouse refresh {symbol.upper()} "
            f"--sections prices --providers {provider}"
        )

    df = prices.rename(columns=str.lower).copy()
    missing = [column for column in REQUIRED_OHLCV if column not in df.columns]
    if missing:
        raise ValueError(f"{symbol.upper()} prices are missing required columns: {missing}")

    df = df.loc[:, list(REQUIRED_OHLCV)]
    df.index = pd.DatetimeIndex(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df = df.sort_index().dropna(subset=["open", "high", "low", "close"])
    ohlc = df.loc[:, ["open", "high", "low", "close"]]
    df["high"] = ohlc.max(axis=1)
    df["low"] = ohlc.min(axis=1)
    return df
