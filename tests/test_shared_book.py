from __future__ import annotations

import pandas as pd

from quant_orchestrator.platforms.backtesting_frameworks.shared_book import (
    build_shared_book_weights,
    run_shared_book_backtest,
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
