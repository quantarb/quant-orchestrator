from __future__ import annotations

import pandas as pd
import pytest

from quant_orchestrator.platforms.backtesting_frameworks.optimal_trader import (
    OptimalTraderBacktestConfig,
    run_optimal_trader_equity_backtest,
)
from quant_orchestrator.platforms.registry import registry


def test_optimal_trader_provider_is_registered() -> None:
    provider = registry.get("backtesting_framework", "optimal_trader")
    assert provider.name == "optimal_trader"
    assert provider.capabilities == ("run", "equity", "strategy_dataset")


def test_optimal_trader_equity_backtest_matches_hand_computed_vectorized_accounting() -> None:
    frame = _strategy_frame()

    result = run_optimal_trader_equity_backtest(
        frame,
        config=OptimalTraderBacktestConfig(fee_bps=0.0, slippage_bps=0.0),
    )

    daily = pd.DataFrame(result.daily_rows).set_index("date")
    assert daily["positions"].tolist() == [0, 2, 1, 1]
    assert daily["turnover"].tolist() == pytest.approx([0.0, 0.5, 0.5, 1.0])
    assert daily["daily_return"].tolist() == pytest.approx([0.0, 0.15, -0.10, 0.05])
    assert daily["net_daily_return"].tolist() == pytest.approx([0.0, 0.15, -0.10, 0.05])
    assert result.final_equity == pytest.approx(1.08675)
    assert result.cumulative_return == pytest.approx(0.08675)
    assert result.max_drawdown == pytest.approx(-0.10)

    effective = result.trade_frame.pivot(index="date", columns="symbol", values="effective_weight").fillna(0.0)
    assert effective.loc["2024-01-02", "AAA"] == pytest.approx(0.5)
    assert effective.loc["2024-01-02", "BBB"] == pytest.approx(-0.5)
    assert effective.loc["2024-01-03", "AAA"] == pytest.approx(1.0)
    assert effective.loc["2024-01-04", "BBB"] == pytest.approx(-1.0)


def test_optimal_trader_provider_runs_strategy_dataset_frame() -> None:
    engine_cls = registry.adapter("backtesting_framework", "optimal_trader")
    engine = engine_cls()

    result = engine.run(None, _strategy_frame(), config={"fee_bps": 0.0, "slippage_bps": 0.0})

    assert result.final_equity == pytest.approx(1.08675)
    assert len(result.daily_rows) == 4


def test_optimal_trader_equity_backtest_uses_transaction_cost_fallback() -> None:
    result = run_optimal_trader_equity_backtest(
        _strategy_frame(),
        config={
            "transaction_cost_bps": 10.0,
            "fee_bps": 0.0,
            "slippage_bps": 0.0,
        },
    )

    daily = pd.DataFrame(result.daily_rows).set_index("date")
    assert daily["turnover_cost"].tolist() == pytest.approx([0.0, 0.0005, 0.0005, 0.001])
    assert daily["net_daily_return"].tolist() == pytest.approx([0.0, 0.1495, -0.1005, 0.049])


def _strategy_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    specs = [
        ("2024-01-01", "AAA", 1.0, 0.00),
        ("2024-01-01", "BBB", -1.0, 0.00),
        ("2024-01-02", "AAA", 1.0, 0.10),
        ("2024-01-02", "BBB", 0.0, -0.20),
        ("2024-01-03", "AAA", 0.0, -0.10),
        ("2024-01-03", "BBB", -1.0, 0.10),
        ("2024-01-04", "AAA", 0.0, 0.02),
        ("2024-01-04", "BBB", 0.0, -0.05),
    ]
    for date, symbol, target_weight, ret_1 in specs:
        rows.append(
            {
                "date": date,
                "symbol": symbol,
                "target_weight": target_weight,
                "ret_1": ret_1,
                "strategy_signal": int(target_weight > 0) - int(target_weight < 0),
                "strategy_score": abs(target_weight),
                "close": 100.0,
                "volume": 1_000_000,
            }
        )
    return pd.DataFrame(rows)
