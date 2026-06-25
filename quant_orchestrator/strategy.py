from __future__ import annotations

import pandas as pd


def sma_positions(close: pd.Series, *, fast_window: int, slow_window: int) -> pd.Series:
    if fast_window >= slow_window:
        raise ValueError("fast_window must be less than slow_window")

    fast = close.rolling(fast_window).mean()
    slow = close.rolling(slow_window).mean()
    positions = (fast > slow).astype(float)
    return positions.shift(1).fillna(0.0)


def fixed_trade_size(close: pd.Series, capital_base: float, *, exposure: float = 0.9) -> int:
    return max(1, int((capital_base * exposure) / close.max()))


def fixed_size_sma_equity(
    prices: pd.DataFrame,
    *,
    fast_window: int,
    slow_window: int,
    capital_base: float,
    trade_size: int,
) -> tuple[pd.Series, int]:
    if fast_window >= slow_window:
        raise ValueError("fast_window must be less than slow_window")

    close = prices["close"]
    fast = close.rolling(fast_window).mean()
    slow = close.rolling(slow_window).mean()
    cash = float(capital_base)
    position = 0
    trades = 0
    values = []

    for i, (date, price) in enumerate(close.items()):
        if i >= slow_window - 1:
            bullish = fast.loc[date] > slow.loc[date]
            if bullish and position == 0:
                cash -= trade_size * float(price)
                position = trade_size
                trades += 1
            elif not bullish and position > 0:
                cash += position * float(price)
                position = 0
                trades += 1
        values.append(cash + position * float(price))

    return pd.Series(values, index=prices.index, name="portfolio_value"), trades


def summarize_equity(equity: pd.Series) -> dict[str, float | str]:
    returns = equity.pct_change().fillna(0.0)
    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    drawdown = equity / equity.cummax() - 1
    return {
        "start": str(equity.index[0].date()),
        "end": str(equity.index[-1].date()),
        "final_equity": round(float(equity.iloc[-1]), 2),
        "total_return": round(float(total_return), 4),
        "max_drawdown": round(float(drawdown.min()), 4),
        "daily_vol": round(float(returns.std()), 4),
    }


def summarize_backtest(
    *,
    framework: str,
    symbol: str,
    equity: pd.Series,
    elapsed_seconds: float,
    bars: int,
    trades: int | None = None,
) -> pd.DataFrame:
    summary = summarize_equity(equity)
    summary.update(
        {
            "framework": framework,
            "symbol": symbol.upper(),
            "bars": bars,
            "trades": trades,
            "elapsed_seconds": round(elapsed_seconds, 4),
            "bars_per_second": round(bars / elapsed_seconds, 2) if elapsed_seconds else None,
        }
    )
    columns = [
        "framework",
        "symbol",
        "start",
        "end",
        "bars",
        "trades",
        "final_equity",
        "total_return",
        "max_drawdown",
        "daily_vol",
        "elapsed_seconds",
        "bars_per_second",
    ]
    return pd.DataFrame([summary], columns=columns)
