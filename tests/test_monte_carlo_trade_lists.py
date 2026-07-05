from __future__ import annotations

import pandas as pd
import pytest

from quant_orchestrator.monte_carlo import (
    simulate_trade_list_paths,
    trade_list_return_series,
)
from quant_orchestrator.platforms.backtesting_frameworks.strategy_artifacts import (
    write_trade_list_artifact,
)


def _trades(returns: list[float], *, prefix: str = "t") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_id": [f"{prefix}{idx}" for idx in range(len(returns))],
            "symbol": ["AAPL"] * len(returns),
            "side": ["long"] * len(returns),
            "entry_date": pd.date_range("2024-01-02", periods=len(returns), freq="D"),
            "exit_date": pd.date_range("2024-01-03", periods=len(returns), freq="D"),
            "ret_dec": returns,
        }
    )


def test_trade_list_return_series_prefers_standard_realized_return() -> None:
    returns = trade_list_return_series(_trades([0.10, -0.05, 0.02]))

    assert returns.tolist() == pytest.approx([0.10, -0.05, 0.02])


def test_trade_list_return_series_can_use_pnl_and_notional() -> None:
    trades = _trades([0.0, 0.0]).drop(columns=["ret_dec"])
    trades["pnl"] = [10.0, -5.0]
    trades["equity_entry_notional"] = [100.0, 100.0]

    returns = trade_list_return_series(trades)

    assert returns.tolist() == pytest.approx([0.10, -0.05])


def test_simulate_trade_list_paths_accepts_manifest_and_mixed_sources(tmp_path) -> None:
    manifest_dir = tmp_path / "strategy_a"
    write_trade_list_artifact(_trades([0.10, -0.05], prefix="a"), manifest_dir)
    direct = tmp_path / "strategy_b.parquet"
    _trades([0.02, 0.03], prefix="b").to_parquet(direct, index=False)

    result = simulate_trade_list_paths(
        {
            "strategy_a": manifest_dir,
            "strategy_b": direct,
        },
        iterations=25,
        horizon=4,
        seed=7,
    )

    assert result.paths.shape == (4, 25)
    assert result.summary.loc[0, "iterations"] == 25
    assert result.summary.loc[0, "horizon"] == 4
