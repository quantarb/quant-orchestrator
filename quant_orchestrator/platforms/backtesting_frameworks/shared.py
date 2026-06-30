from __future__ import annotations

from typing import Any

import pandas as pd

from quant_orchestrator.platforms.backtesting_frameworks.reporting import normalize_equity_curve
from quant_warehouse import Warehouse
from quant_warehouse.platforms.data_providers.fmp.feature_engineering import compute_features_worldclass

MAG7_SYMBOLS: tuple[str, ...] = ("AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA", "TSLA")
OHLCV_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")


def normalize_session_label(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    return ts.normalize()


def equal_notional_capital(capital_base: float, item_count: int) -> float:
    if item_count <= 0:
        raise ValueError("item_count must be positive")
    return float(capital_base) / item_count


def combine_equity_curves(curves: list[pd.Series]) -> pd.Series:
    if not curves:
        raise ValueError("At least one equity curve is required")

    normalized_curves = [normalize_equity_curve(curve) for curve in curves]
    combined_index = pd.DatetimeIndex([])
    for curve in normalized_curves:
        combined_index = combined_index.union(curve.index)

    combined_index = pd.DatetimeIndex(combined_index).sort_values()
    combined = pd.Series(0.0, index=combined_index, name="portfolio_value")
    for curve in normalized_curves:
        aligned = curve.reindex(combined_index).ffill().fillna(curve.iloc[0])
        combined = combined.add(aligned, fill_value=0.0)
    return combined.sort_index().rename("portfolio_value")


def build_sma_crossover_frame(
    prices: pd.DataFrame,
    *,
    fast_window: int,
    slow_window: int,
) -> pd.DataFrame:
    if fast_window >= slow_window:
        raise ValueError("fast_window must be less than slow_window")

    frame = compute_features_worldclass(prices.copy())
    fast_col = f"SMA{fast_window}"
    slow_col = f"SMA{slow_window}"
    missing = [column for column in (fast_col, slow_col) if column not in frame.columns]
    if missing:
        raise ValueError(
            "Quant Warehouse feature output is missing required SMA columns: "
            f"{missing}. Update quant-warehouse feature engineering first.",
        )

    frame["fast_sma"] = frame[fast_col]
    frame["slow_sma"] = frame[slow_col]
    frame["signal"] = (frame["fast_sma"] > frame["slow_sma"]).astype(int).fillna(0)
    return frame.dropna(subset=list(OHLCV_COLUMNS)).copy()


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
