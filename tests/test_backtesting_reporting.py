from __future__ import annotations

import pandas as pd

from quant_orchestrator.platforms.backtesting_frameworks.reporting import (
    build_common_summary,
    normalize_equity_curve,
    normalize_trade_log,
)
from quant_orchestrator.platforms.backtesting_frameworks.shared import combine_equity_curves


def test_normalize_equity_curve_removes_timezone_and_duplicates() -> None:
    equity = pd.Series(
        [100.0, 101.0, 102.0],
        index=pd.to_datetime(
            [
                "2026-01-02 16:00:00+00:00",
                "2026-01-02 20:00:00+00:00",
                "2026-01-05 16:00:00+00:00",
            ],
        ),
    )

    normalized = normalize_equity_curve(equity)

    assert normalized.index.tz is None
    assert normalized.index.tolist() == [pd.Timestamp("2026-01-02"), pd.Timestamp("2026-01-05")]
    assert normalized.iloc[0] == 101.0
    assert normalized.name == "portfolio_value"


def test_build_common_summary_adds_cross_framework_metrics() -> None:
    equity = pd.Series(
        [100.0, 110.0, 105.0],
        index=pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"]),
    )

    summary = build_common_summary(
        framework="zipline",
        symbol="AAPL",
        equity=equity,
        elapsed_seconds=2.0,
        bars=3,
        trades=4,
    )
    row = summary.iloc[0]

    assert row["framework"] == "zipline"
    assert row["symbol"] == "AAPL"
    assert row["initial_equity"] == 100.0
    assert row["final_equity"] == 105.0
    assert row["total_return"] == 0.05
    assert "sharpe" in summary.columns
    assert "calmar" in summary.columns


def test_normalize_trade_log_keeps_common_columns_and_notional() -> None:
    trades = pd.DataFrame(
        [
            {
                "timestamp": "2026-01-02 09:30:00-05:00",
                "symbol": "AAPL",
                "side": "BUY",
                "quantity": 10,
                "price": 150.0,
            }
        ]
    )

    normalized = normalize_trade_log(trades)

    assert list(normalized.columns) == [
        "timestamp",
        "symbol",
        "side",
        "quantity",
        "price",
        "notional",
        "fees",
        "order_id",
        "native_id",
    ]
    assert normalized.loc[0, "timestamp"] == pd.Timestamp("2026-01-02")
    assert normalized.loc[0, "notional"] == 1500.0


def test_combine_equity_curves_handles_mixed_timezone_indexes() -> None:
    naive = pd.Series([100.0, 101.0], index=pd.to_datetime(["2026-01-02", "2026-01-05"]))
    aware = pd.Series(
        [50.0, 51.0],
        index=pd.to_datetime(["2026-01-02 00:00:00+00:00", "2026-01-05 00:00:00+00:00"]),
    )

    combined = combine_equity_curves([naive, aware])

    assert combined.index.tz is None
    assert combined.index.tolist() == [pd.Timestamp("2026-01-02"), pd.Timestamp("2026-01-05")]
    assert combined.tolist() == [150.0, 152.0]
