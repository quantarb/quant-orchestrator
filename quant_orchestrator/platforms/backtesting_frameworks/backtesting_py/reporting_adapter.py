from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from quant_orchestrator.platforms.backtesting_frameworks.reporting import (
    build_normalized_report,
)


@dataclass(frozen=True)
class BacktestingPyReport:
    summary: pd.DataFrame
    equity_curve: pd.Series
    trade_log: pd.DataFrame
    native_report: pd.Series
    native_metrics: dict[str, Any]


def _normalize_trade_log(trades: pd.DataFrame) -> pd.DataFrame:
    if trades is None or trades.empty:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "symbol",
                "side",
                "quantity",
                "price",
                "notional",
                "fees",
                "order_id",
                "native_id",
                "pnl",
                "return_pct",
                "duration",
            ],
        )

    renamed = trades.rename(
        columns={
            "EntryTime": "entry_time",
            "ExitTime": "exit_time",
            "Size": "size",
            "EntryPrice": "entry_price",
            "ExitPrice": "exit_price",
            "PnL": "pnl",
            "ReturnPct": "return_pct",
            "Duration": "duration",
        },
    ).copy()
    normalized = pd.DataFrame(
        {
            "timestamp": renamed.get("entry_time"),
            "symbol": None,
            "side": renamed.get("size", pd.Series(dtype=float)).map(lambda value: "BUY" if float(value) > 0 else "SELL"),
            "quantity": renamed.get("size", pd.Series(dtype=float)).abs(),
            "price": renamed.get("entry_price"),
            "notional": renamed.get("size", pd.Series(dtype=float)).abs() * renamed.get("entry_price", pd.Series(dtype=float)),
            "fees": 0.0,
            "order_id": None,
            "native_id": renamed.index,
            "pnl": renamed.get("pnl"),
            "return_pct": renamed.get("return_pct"),
            "duration": renamed.get("duration"),
            "exit_time": renamed.get("exit_time"),
            "exit_price": renamed.get("exit_price"),
        }
    )
    return normalized


def build_backtesting_py_report(
    stats: pd.Series,
    *,
    symbol: str,
    elapsed_seconds: float,
    trade_count: int | None = None,
) -> BacktestingPyReport:
    trade_log = _normalize_trade_log(stats.get("_trades", pd.DataFrame()))
    native_metrics = {
        "return_pct": float(stats["Return [%]"]),
        "sharpe": float(stats["Sharpe Ratio"]) if pd.notna(stats["Sharpe Ratio"]) else None,
        "sortino": float(stats["Sortino Ratio"]) if pd.notna(stats["Sortino Ratio"]) else None,
        "calmar": float(stats["Calmar Ratio"]) if pd.notna(stats["Calmar Ratio"]) else None,
        "max_drawdown_pct": float(stats["Max. Drawdown [%]"]),
        "win_rate_pct": float(stats["Win Rate [%]"]) if pd.notna(stats["Win Rate [%]"]) else None,
        "profit_factor": float(stats["Profit Factor"]) if pd.notna(stats["Profit Factor"]) else None,
        "expectancy_pct": float(stats["Expectancy [%]"]) if pd.notna(stats["Expectancy [%]"]) else None,
        "sqn": float(stats["SQN"]) if pd.notna(stats["SQN"]) else None,
    }
    report = build_normalized_report(
        framework="backtesting.py",
        symbol=symbol,
        equity=stats["_equity_curve"]["Equity"],
        elapsed_seconds=elapsed_seconds,
        bars=len(stats["_equity_curve"]),
        trades=trade_log,
        trade_count=trade_count,
        native_report=stats,
        native_metrics=native_metrics,
    )
    return BacktestingPyReport(
        summary=report.summary,
        equity_curve=report.equity_curve,
        trade_log=report.trade_log,
        native_report=stats,
        native_metrics=native_metrics,
    )
