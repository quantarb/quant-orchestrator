from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


SUMMARY_COLUMNS: tuple[str, ...] = (
    "framework",
    "symbol",
    "start",
    "end",
    "bars",
    "trades",
    "initial_equity",
    "final_equity",
    "total_return",
    "annualized_return",
    "max_drawdown",
    "annualized_vol",
    "sharpe",
    "calmar",
    "elapsed_seconds",
    "bars_per_second",
)

TRADE_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "symbol",
    "side",
    "quantity",
    "price",
    "notional",
    "fees",
    "order_id",
    "native_id",
)


@dataclass(frozen=True)
class NormalizedBacktestReport:
    summary: pd.DataFrame
    equity_curve: pd.Series
    returns: pd.Series
    trade_log: pd.DataFrame
    native_report: Any
    native_metrics: dict[str, Any]


def normalize_datetime_index(index: Any) -> pd.DatetimeIndex:
    normalized = pd.DatetimeIndex(pd.to_datetime(index))
    if normalized.tz is not None:
        normalized = normalized.tz_convert(None)
    return normalized.normalize()


def normalize_equity_curve(equity: pd.Series) -> pd.Series:
    if equity.empty:
        raise ValueError("equity curve must not be empty")

    normalized = pd.to_numeric(equity.copy(), errors="coerce")
    normalized.index = normalize_datetime_index(normalized.index)
    normalized = normalized.sort_index()
    normalized = normalized[~normalized.index.duplicated(keep="last")]
    normalized = normalized.dropna()
    if normalized.empty:
        raise ValueError("equity curve has no numeric values after normalization")
    return normalized.rename("portfolio_value")


def normalize_returns(equity: pd.Series) -> pd.Series:
    normalized = normalize_equity_curve(equity)
    return normalized.pct_change().replace([np.inf, -np.inf], np.nan).dropna().rename("returns")


def build_common_summary(
    *,
    framework: str,
    symbol: str,
    equity: pd.Series,
    elapsed_seconds: float,
    bars: int | None = None,
    trades: int | None = None,
) -> pd.DataFrame:
    normalized = normalize_equity_curve(equity)
    returns = normalized.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    total_return = float(normalized.iloc[-1] / normalized.iloc[0] - 1)
    drawdown = normalized / normalized.cummax() - 1
    periods_per_year = 252.0
    years = max(len(normalized) / periods_per_year, 1.0 / periods_per_year)
    annualized_return = float((normalized.iloc[-1] / normalized.iloc[0]) ** (1 / years) - 1)
    annualized_vol = float(returns.std() * np.sqrt(periods_per_year)) if not returns.empty else 0.0
    sharpe = float((returns.mean() / returns.std()) * np.sqrt(periods_per_year)) if not returns.empty and float(returns.std()) != 0.0 else 0.0
    max_drawdown = float(drawdown.min())
    calmar = float(annualized_return / abs(max_drawdown)) if max_drawdown < 0 else np.nan
    elapsed = float(elapsed_seconds)
    bar_count = int(bars if bars is not None else len(normalized))

    return pd.DataFrame(
        [
            {
                "framework": framework,
                "symbol": symbol.upper(),
                "start": str(normalized.index[0].date()),
                "end": str(normalized.index[-1].date()),
                "bars": bar_count,
                "trades": int(trades if trades is not None else 0),
                "initial_equity": round(float(normalized.iloc[0]), 2),
                "final_equity": round(float(normalized.iloc[-1]), 2),
                "total_return": round(total_return, 4),
                "annualized_return": round(annualized_return, 4),
                "max_drawdown": round(max_drawdown, 4),
                "annualized_vol": round(annualized_vol, 4),
                "sharpe": round(sharpe, 4),
                "calmar": round(calmar, 4) if pd.notna(calmar) else None,
                "elapsed_seconds": round(elapsed, 4),
                "bars_per_second": round(bar_count / elapsed, 2) if elapsed else None,
            }
        ],
        columns=SUMMARY_COLUMNS,
    )


def normalize_trade_log(trades: pd.DataFrame | None) -> pd.DataFrame:
    if trades is None or trades.empty:
        return pd.DataFrame(columns=TRADE_COLUMNS)

    normalized = trades.copy()
    for column in TRADE_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = None

    if normalized["timestamp"].notna().any():
        timestamps = pd.to_datetime(normalized["timestamp"], errors="coerce")
        if getattr(timestamps.dt, "tz", None) is not None:
            timestamps = timestamps.dt.tz_convert(None)
        normalized["timestamp"] = timestamps.dt.normalize()

    normalized["quantity"] = pd.to_numeric(normalized["quantity"], errors="coerce")
    normalized["price"] = pd.to_numeric(normalized["price"], errors="coerce")
    missing_notional = normalized["notional"].isna()
    normalized.loc[missing_notional, "notional"] = (
        normalized.loc[missing_notional, "quantity"].abs()
        * normalized.loc[missing_notional, "price"]
    )
    return normalized.loc[:, TRADE_COLUMNS]


def build_normalized_report(
    *,
    framework: str,
    symbol: str,
    equity: pd.Series,
    elapsed_seconds: float,
    bars: int | None,
    trades: pd.DataFrame | None,
    trade_count: int | None,
    native_report: Any,
    native_metrics: dict[str, Any] | None = None,
) -> NormalizedBacktestReport:
    equity_curve = normalize_equity_curve(equity)
    trade_log = normalize_trade_log(trades)
    summary = build_common_summary(
        framework=framework,
        symbol=symbol,
        equity=equity_curve,
        elapsed_seconds=elapsed_seconds,
        bars=bars,
        trades=int(trade_count if trade_count is not None else len(trade_log)),
    )
    return NormalizedBacktestReport(
        summary=summary,
        equity_curve=equity_curve,
        returns=normalize_returns(equity_curve),
        trade_log=trade_log,
        native_report=native_report,
        native_metrics=native_metrics or {},
    )
