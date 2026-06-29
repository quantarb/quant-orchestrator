from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from quant_orchestrator.platforms.backtesting_frameworks.reporting import (
    build_normalized_report,
)
from quant_orchestrator.platforms.backtesting_frameworks.shared import normalize_session_label


@dataclass(frozen=True)
class NautilusReport:
    summary: pd.DataFrame
    equity_curve: pd.Series
    trade_log: pd.DataFrame
    native_report: pd.DataFrame
    native_metrics: dict[str, Any]


def _normalize_trade_log(fills: pd.DataFrame) -> pd.DataFrame:
    if fills is None or fills.empty:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "symbol",
                "side",
                "quantity",
                "price",
                "slippage",
                "commissions",
                "status",
                "client_order_id",
            ],
        )

    rows = []
    for client_order_id, fill in fills.iterrows():
        timestamp = fill.get("ts_last", fill.get("ts_init"))
        rows.append(
            {
                "timestamp": normalize_session_label(timestamp),
                "symbol": str(fill.get("instrument_id", "")),
                "side": str(fill.get("side", "")),
                "quantity": float(fill.get("filled_qty", 0.0)),
                "price": float(fill.get("avg_px", 0.0)),
                "notional": float(fill.get("filled_qty", 0.0)) * float(fill.get("avg_px", 0.0)),
                "fees": fill.get("commissions"),
                "status": fill.get("status"),
                "order_id": client_order_id,
                "native_id": client_order_id,
                "slippage": fill.get("slippage"),
            },
        )
    return pd.DataFrame(rows)


def build_nautilus_report(
    fills: pd.DataFrame,
    equity_curve: pd.Series,
    *,
    symbol: str,
    elapsed_seconds: float,
    trade_count: int | None = None,
) -> NautilusReport:
    trade_log = _normalize_trade_log(fills)
    native_metrics = {
        "fill_count": int(len(fills)),
        "buy_fills": int((fills["side"] == "BUY").sum()) if not fills.empty else 0,
        "sell_fills": int((fills["side"] == "SELL").sum()) if not fills.empty else 0,
        "avg_fill_price": float(pd.to_numeric(fills["avg_px"], errors="coerce").mean()) if not fills.empty else None,
        "avg_slippage": float(pd.to_numeric(fills["slippage"], errors="coerce").mean()) if not fills.empty else None,
        "total_commissions": str(fills["commissions"].iloc[0]) if not fills.empty and "commissions" in fills.columns else None,
    }
    report = build_normalized_report(
        framework="nautilus",
        symbol=symbol,
        equity=equity_curve,
        elapsed_seconds=elapsed_seconds,
        bars=len(equity_curve),
        trades=trade_log,
        trade_count=trade_count,
        native_report=fills,
        native_metrics=native_metrics,
    )
    return NautilusReport(
        summary=report.summary,
        equity_curve=report.equity_curve,
        trade_log=report.trade_log,
        native_report=fills,
        native_metrics=native_metrics,
    )
