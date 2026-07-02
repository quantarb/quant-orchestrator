from __future__ import annotations

import pandas as pd

from quant_orchestrator.platforms.backtesting_frameworks.shared_book import (
    build_shared_book_weights,
    run_shared_book_backtest,
)
from quant_orchestrator.platforms.backtesting_frameworks.zipline.shared_book import (
    ZiplineSharedBookSummaryJob,
    run_zipline_shared_book,
    run_zipline_shared_book_summary_jobs,
)


def test_shared_book_uses_one_capacity_for_long_short() -> None:
    scores = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02"] * 4),
            "symbol": ["A", "B", "C", "D"],
            "long_score": [0.70, 0.20, 0.65, 0.10],
            "short_score": [0.10, 0.80, 0.20, 0.75],
        }
    )

    weights, trades = build_shared_book_weights(
        scores,
        symbols=["A", "B", "C", "D"],
        dates=pd.DatetimeIndex(["2024-01-02"]),
        top_k=2,
        variant="long_short",
        entry_threshold=0.5,
    )

    assert weights.loc[pd.Timestamp("2024-01-02"), "B"] == -0.5
    assert weights.loc[pd.Timestamp("2024-01-02"), "D"] == -0.5
    assert weights.abs().sum(axis=1).iloc[0] == 1.0
    assert len(trades) == 2


def test_shared_book_exits_then_refills_open_slots() -> None:
    scores = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"]),
            "symbol": ["A", "B", "A", "B"],
            "long_score": [0.70, 0.65, 0.40, 0.80],
            "short_score": [0.10, 0.10, 0.75, 0.10],
        }
    )

    weights, trades = build_shared_book_weights(
        scores,
        symbols=["A", "B"],
        dates=pd.DatetimeIndex(["2024-01-02", "2024-01-03"]),
        top_k=1,
        variant="long_only",
        entry_threshold=0.5,
        exit_threshold=0.5,
    )

    assert weights.loc[pd.Timestamp("2024-01-02"), "A"] == 1.0
    assert weights.loc[pd.Timestamp("2024-01-03"), "A"] == 0.0
    assert weights.loc[pd.Timestamp("2024-01-03"), "B"] == 1.0
    assert trades["action"].tolist() == ["enter_long", "exit_long", "enter_long"]


def test_shared_book_costs_apply_to_turnover() -> None:
    weights = pd.DataFrame(
        {"A": [1.0, 0.0]},
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    )
    returns = pd.DataFrame(
        {"A": [0.10, 0.00]},
        index=weights.index,
    )

    net_returns, equity, turnover = run_shared_book_backtest(weights, returns, cost_bps=100.0, capital_base=100.0)

    assert turnover.tolist() == [1.0, 1.0]
    assert round(float(net_returns.iloc[0]), 4) == 0.09
    assert round(float(net_returns.iloc[1]), 4) == -0.01
    assert round(float(equity.iloc[-1]), 2) == 107.91


def test_zipline_shared_book_runs_native_multi_asset_orders() -> None:
    dates = pd.bdate_range("2024-01-02", periods=20)

    def prices(base: float) -> pd.DataFrame:
        close = pd.Series([base + offset for offset in range(len(dates))], index=dates, dtype=float)
        return pd.DataFrame(
            {
                "open": close,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": 1_000_000,
            },
            index=dates,
        )

    target_weights = pd.DataFrame(
        {
            "AAA": [0.5] * 10 + [0.0] * 10,
            "BBB": [0.0] * 10 + [0.5] * 10,
        },
        index=dates,
    )

    result = run_zipline_shared_book(
        {"AAA": prices(100.0), "BBB": prices(50.0)},
        target_weights,
        capital_base=100_000.0,
        commission_per_share=0.005,
        slippage_bps=5.0,
    )

    assert result.summary.loc[0, "framework"] == "zipline_shared_book_native"
    assert result.summary.loc[0, "trades"] > 0
    assert result.equity_curve.iloc[-1] > result.equity_curve.iloc[0]
    assert not result.orders.empty


def test_zipline_shared_book_summary_jobs_return_metadata() -> None:
    dates = pd.bdate_range("2024-01-02", periods=20)
    close = pd.Series(range(100, 120), index=dates, dtype=float)
    prices = pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 1_000_000,
        },
        index=dates,
    )
    target_weights = pd.DataFrame({"AAA": [1.0] * len(dates)}, index=dates)

    rows = run_zipline_shared_book_summary_jobs(
        [
            ZiplineSharedBookSummaryJob(
                price_frames={"AAA": prices},
                target_weights=target_weights,
                metadata={"strategy_source": "test_model", "variant": "long_only", "top_k": 1},
                capital_base=100_000.0,
            )
        ],
        max_workers=1,
    )

    assert rows.loc[0, "framework"] == "zipline_shared_book_native"
    assert rows.loc[0, "strategy_source"] == "test_model"
    assert rows.loc[0, "variant"] == "long_only"
