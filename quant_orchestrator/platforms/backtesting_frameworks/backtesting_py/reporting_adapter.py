from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from quant_orchestrator.strategy import summarize_backtest


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
                "entry_time",
                "exit_time",
                "size",
                "entry_price",
                "exit_price",
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
    keep = [column for column in renamed.columns if column in {"entry_time", "exit_time", "size", "entry_price", "exit_price", "pnl", "return_pct", "duration"}]
    return renamed.loc[:, keep]


def build_backtesting_py_report(
    stats: pd.Series,
    *,
    symbol: str,
    elapsed_seconds: float,
    trade_count: int | None = None,
) -> BacktestingPyReport:
    equity_curve = stats["_equity_curve"]["Equity"].rename("portfolio_value")
    trade_log = _normalize_trade_log(stats.get("_trades", pd.DataFrame()))
    summary = summarize_backtest(
        framework="backtesting.py",
        symbol=symbol,
        equity=equity_curve,
        elapsed_seconds=elapsed_seconds,
        bars=len(equity_curve),
        trades=int(trade_count if trade_count is not None else len(trade_log)),
    )
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
    return BacktestingPyReport(
        summary=summary,
        equity_curve=equity_curve,
        trade_log=trade_log,
        native_report=stats,
        native_metrics=native_metrics,
    )
