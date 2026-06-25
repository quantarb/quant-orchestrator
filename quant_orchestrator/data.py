from __future__ import annotations

from pathlib import Path

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


def write_zipline_csv(symbol: str, prices: pd.DataFrame, root: Path, *, calendar_name: str = "XNYS") -> Path:
    from zipline.utils.calendar_utils import get_calendar

    daily_dir = root / "daily"
    daily_dir.mkdir(parents=True, exist_ok=True)
    out = prices.copy()
    out.index = out.index.tz_convert(None).normalize()
    calendar = get_calendar(calendar_name)
    sessions = calendar.sessions_in_range(out.index.min(), out.index.max()).tz_localize(None)
    out = out.reindex(sessions)
    out.loc[:, ["open", "high", "low", "close"]] = out.loc[:, ["open", "high", "low", "close"]].ffill()
    out["volume"] = out["volume"].fillna(0.0)
    out.index.name = "date"
    out["dividend"] = 0.0
    out["split"] = 1.0
    path = daily_dir / f"{symbol.upper()}.csv"
    out.to_csv(path)
    return path
