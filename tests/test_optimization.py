from __future__ import annotations

import pandas as pd

from quant_orchestrator.optimization import (
    MetricFilter,
    build_parameter_grid,
    filter_results,
    optimize_long_only_sharpe_weights,
    optimize_random_mean_variance_weights,
    rank_results,
    returns_matrix_from_equity_curves,
)


def test_build_parameter_grid_applies_constraints() -> None:
    grid = build_parameter_grid(
        {"fast_window": [5, 10], "slow_window": [8, 20]},
        constraints=(lambda row: row["fast_window"] < row["slow_window"],),
    )

    assert grid.to_dict("records") == [
        {"fast_window": 5, "slow_window": 8},
        {"fast_window": 5, "slow_window": 20},
        {"fast_window": 10, "slow_window": 20},
    ]


def test_filter_and_rank_results_are_metric_driven() -> None:
    table = pd.DataFrame(
        [
            {"variant": "a", "total_return": 0.2, "trades": 10},
            {"variant": "b", "total_return": -0.1, "trades": 40},
            {"variant": "c", "total_return": 0.3, "trades": 5},
        ]
    )

    filtered = filter_results(
        table,
        [
            MetricFilter("total_return", ">", 0.0),
            MetricFilter("trades", ">=", 10),
        ],
    )
    ranked = rank_results(filtered, by="total_return", ascending=False, rank_column="train_rank")

    assert ranked["variant"].tolist() == ["a"]
    assert ranked["train_rank"].tolist() == [1]


def test_returns_matrix_from_equity_curves_aligns_curves() -> None:
    curves = {
        "a": pd.Series([100.0, 101.0, 102.0], index=pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"])),
        "b": pd.Series([50.0, 55.0, 54.0], index=pd.to_datetime(["2026-01-02", "2026-01-03", "2026-01-04"])),
    }

    matrix = returns_matrix_from_equity_curves(curves)

    assert list(matrix.columns) == ["a", "b"]
    assert not matrix.empty


def test_long_only_sharpe_weights_respect_max_weight() -> None:
    returns = pd.DataFrame(
        {
            "a": [0.01, 0.02, -0.01, 0.03],
            "b": [0.00, 0.01, 0.00, 0.01],
            "c": [-0.01, 0.00, 0.01, 0.00],
        }
    )

    weights = optimize_long_only_sharpe_weights(returns, max_weight=0.6)

    assert abs(float(weights.sum()) - 1.0) < 1e-9
    assert float(weights.max()) <= 0.600001


def test_random_mean_variance_weights_support_group_constraints() -> None:
    returns = pd.DataFrame(
        {
            "fmp|zipline": [0.01, 0.02, 0.00, 0.01],
            "fmp|nautilus": [0.01, -0.01, 0.02, 0.01],
            "yfinance|zipline": [0.00, 0.01, 0.01, 0.00],
            "yfinance|nautilus": [0.02, 0.00, 0.01, -0.01],
        }
    )
    providers = pd.Series(["fmp", "fmp", "yfinance", "yfinance"], index=returns.columns)
    frameworks = pd.Series(["zipline", "nautilus", "zipline", "nautilus"], index=returns.columns)

    result = optimize_random_mean_variance_weights(
        returns,
        iterations=500,
        max_weight=0.7,
        min_active_weight=0.05,
        risk_aversion=1.0,
        seed=1,
        group_constraints={"provider": providers, "framework": frameworks},
    )

    assert abs(float(result.weights.sum()) - 1.0) < 1e-9
    assert providers.loc[result.weights.index].nunique() >= 2
    assert frameworks.loc[result.weights.index].nunique() >= 2

