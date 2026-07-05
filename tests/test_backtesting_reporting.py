from __future__ import annotations

import pandas as pd

from quant_orchestrator.platforms.backtesting_frameworks.reporting import (
    build_common_summary,
    normalize_equity_curve,
    normalize_trade_log,
    normalize_trade_list,
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
        "exit_time",
        "exit_price",
        "pnl",
        "return_pct",
        "duration",
    ]
    assert normalized.loc[0, "timestamp"] == pd.Timestamp("2026-01-02")
    assert normalized.loc[0, "notional"] == 1500.0


def test_normalize_trade_list_uses_closed_trade_fields() -> None:
    trades = pd.DataFrame(
        [
            {
                "timestamp": "2026-01-02",
                "exit_time": "2026-01-09",
                "symbol": "aapl",
                "side": "BUY",
                "quantity": 10,
                "price": 100.0,
                "exit_price": 110.0,
                "return_pct": 0.10,
            }
        ]
    )

    trade_list = normalize_trade_list(trades)

    assert trade_list.loc[0, "trade_id"] == "AAPL|2026-01-02|long"
    assert trade_list.loc[0, "symbol"] == "AAPL"
    assert trade_list.loc[0, "side"] == "long"
    assert trade_list.loc[0, "entry_date"] == pd.Timestamp("2026-01-02")
    assert trade_list.loc[0, "exit_date"] == pd.Timestamp("2026-01-09")
    assert trade_list.loc[0, "entry_price"] == 100.0
    assert trade_list.loc[0, "exit_price"] == 110.0
    assert trade_list.loc[0, "ret_dec"] == 0.10


def test_normalize_trade_list_keeps_classifier_top_k_guard() -> None:
    trades = pd.DataFrame(
        [
            {
                "symbol": "AAPL",
                "side": "long",
                "entry_date": "2026-01-02",
                "exit_date": "2026-01-09",
                "strategy_source": "fmp.a",
            }
        ]
    )

    try:
        normalize_trade_list(trades)
    except KeyError as exc:
        assert "top_k" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected top_k validation error")


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
