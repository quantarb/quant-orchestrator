from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from quant_orchestrator.platforms.backtesting_frameworks.reporting import (
    build_normalized_report,
)
from quant_orchestrator.platforms.backtesting_frameworks.shared import normalize_session_label


@dataclass(frozen=True)
class ZiplineReport:
    summary: pd.DataFrame
    equity_curve: pd.Series
    trade_log: pd.DataFrame
    native_report: pd.DataFrame
    native_metrics: dict[str, Any]


def _normalize_trade_log(perf: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if "transactions" not in perf.columns:
        return pd.DataFrame(columns=["timestamp", "symbol", "side", "quantity", "price", "order_id", "commission"])

    for date, tx_list in perf["transactions"].items():
        if not tx_list:
            continue
        for tx in tx_list:
            amount = float(tx.get("amount", 0.0))
            asset = tx.get("sid")
            symbol = getattr(asset, "symbol", None) or str(asset)
            rows.append(
                {
                    "timestamp": normalize_session_label(tx.get("dt", date)),
                    "symbol": symbol,
                    "side": "BUY" if amount > 0 else "SELL",
                    "quantity": abs(amount),
                    "price": float(tx.get("price", 0.0)),
                    "order_id": tx.get("order_id"),
                    "fees": tx.get("commission"),
                    "notional": abs(amount) * float(tx.get("price", 0.0)),
                    "native_id": tx.get("id"),
                },
            )
    return pd.DataFrame(rows)


def build_zipline_report(
    perf: pd.DataFrame,
    *,
    symbol: str,
    elapsed_seconds: float,
    trade_count: int | None = None,
) -> ZiplineReport:
    trade_log = _normalize_trade_log(perf)
    last = perf.iloc[-1]
    native_metrics = {
        "algorithm_period_return": float(last["algorithm_period_return"]) if pd.notna(last.get("algorithm_period_return")) else None,
        "benchmark_period_return": float(last["benchmark_period_return"]) if pd.notna(last.get("benchmark_period_return")) else None,
        "benchmark_volatility": float(last["benchmark_volatility"]) if pd.notna(last.get("benchmark_volatility")) else None,
        "algo_volatility": float(last["algo_volatility"]) if pd.notna(last.get("algo_volatility")) else None,
        "max_drawdown": float(last["max_drawdown"]) if pd.notna(last.get("max_drawdown")) else None,
        "sharpe": float(last["sharpe"]) if pd.notna(last.get("sharpe")) else None,
        "sortino": float(last["sortino"]) if pd.notna(last.get("sortino")) else None,
        "alpha": float(last["alpha"]) if pd.notna(last.get("alpha")) else None,
        "beta": float(last["beta"]) if pd.notna(last.get("beta")) else None,
        "gross_leverage": float(last["gross_leverage"]) if pd.notna(last.get("gross_leverage")) else None,
        "net_leverage": float(last["net_leverage"]) if pd.notna(last.get("net_leverage")) else None,
    }
    report = build_normalized_report(
        framework="zipline",
        symbol=symbol,
        equity=perf["portfolio_value"],
        elapsed_seconds=elapsed_seconds,
        bars=len(perf),
        trades=trade_log,
        trade_count=trade_count,
        native_report=perf,
        native_metrics=native_metrics,
    )
    return ZiplineReport(
        summary=report.summary,
        equity_curve=report.equity_curve,
        trade_log=report.trade_log,
        native_report=perf,
        native_metrics=native_metrics,
    )
