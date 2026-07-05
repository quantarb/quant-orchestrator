from __future__ import annotations

import pandas as pd
import pytest
import numpy as np

from quant_orchestrator.platforms.backtesting_frameworks.optimal_trader import (
    build_moe_ranked_scores,
    OptimalTraderBacktestConfig,
    action_tape_to_trade_windows,
    replay_moe_paper_top_k_rule,
    replay_option_portfolio_from_selected_paths,
    replay_trading_app_top_k_rule,
    load_strategy_dataset_artifact,
    run_optimal_trader_equity_backtest,
    score_moe_family_panel,
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


def test_optimal_trader_provider_runs_saved_strategy_dataset_path(tmp_path) -> None:
    path = tmp_path / "strategy_dataset.csv"
    _strategy_frame().to_csv(path, index=False)
    engine_cls = registry.adapter("backtesting_framework", "optimal_trader")
    engine = engine_cls()

    result = engine.run(None, path, config={"fee_bps": 0.0, "slippage_bps": 0.0})

    assert result.final_equity == pytest.approx(1.08675)


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


def test_optimal_trader_data_adapter_loads_saved_strategy_dataset(tmp_path) -> None:
    path = tmp_path / "strategy_dataset.pkl"
    pd.DataFrame(
        {
            "Date": ["2024-01-02", "2024-01-02"],
            "Symbol": ["aaa", "bbb"],
            "Target Weight": [0.5, -0.5],
            "Asset Return": [0.10, -0.20],
        }
    ).to_pickle(path)

    dataset = load_strategy_dataset_artifact(path)

    assert dataset.to_dict("records") == [
        {"date": "2024-01-02", "symbol": "AAA", "target_weight": 0.5, "asset_return": 0.10},
        {"date": "2024-01-02", "symbol": "BBB", "target_weight": -0.5, "asset_return": -0.20},
    ]


def test_trading_app_rule_replay_builds_shifted_action_tape_and_trade_windows() -> None:
    panel = _scored_panel()

    replay = replay_trading_app_top_k_rule(
        panel,
        score_col="buy_score_mean_raw_pct6",
        top_k=1,
        component_threshold=0.50,
        initial_balance=100_000.0,
        fee_bps=0.0,
        slippage_bps=0.0,
    )

    assert replay.action_tape[["date", "symbol", "action", "reason"]].to_dict("records") == [
        {
            "date": pd.Timestamp("2024-01-02"),
            "symbol": "AAA",
            "action": "buy",
            "reason": "entry_top_k",
        },
        {
            "date": pd.Timestamp("2024-01-03"),
            "symbol": "AAA",
            "action": "sell",
            "reason": "exit_classifier_or_invalid",
        },
    ]
    assert replay.trade_windows[["symbol", "entry_date", "exit_date", "ret_dec", "exit_reason"]].to_dict("records") == [
        {
            "symbol": "AAA",
            "entry_date": pd.Timestamp("2024-01-02"),
            "exit_date": pd.Timestamp("2024-01-03"),
            "ret_dec": pytest.approx((120.0 / 110.0) - 1.0),
            "exit_reason": "exit_classifier_or_invalid",
        }
    ]
    assert replay.equity.loc[pd.Timestamp("2024-01-03")] == pytest.approx(100_000.0 * (120.0 / 110.0))
    assert replay.trade_windows["equity_entry_notional"].iloc[0] == pytest.approx(100_000.0)


def test_moe_ranked_scores_select_top_k_by_prob_buy() -> None:
    latest = pd.DataFrame(
        {"close": [100.0, 200.0, 300.0], "prob_buy": [0.7, 0.9, 0.4]},
        index=pd.Index(["AAA", "BBB", "CCC"], name="symbol"),
    )

    ranked = build_moe_ranked_scores(latest, top_k=1, threshold=0.5)

    assert ranked.index.tolist() == ["BBB", "AAA", "CCC"]
    assert ranked["selected"].tolist() == [True, False, False]


def test_moe_family_panel_scores_average_available_family_models() -> None:
    panel = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
            "symbol": ["AAA", "BBB", "AAA", "BBB"],
            "close": [100.0, 200.0, 110.0, 190.0],
            "family_a__x": [1.0, -1.0, 2.0, -2.0],
            "family_b__x": [0.5, 0.5, None, None],
        }
    )
    models = {
        "family_a": _ConstantFeatureModel("family_a__x"),
        "family_b": _ConstantFeatureModel("family_b__x"),
    }
    metadata = {
        "trained_families": ["family_a", "family_b"],
        "feature_family_weights": {"family_a": 1.0, "family_b": 1.0},
        "feature_list_by_family": {
            "family_a": ["family_a__x"],
            "family_b": ["family_b__x"],
        },
    }

    scored = score_moe_family_panel(panel, models=models, metadata=metadata)

    assert scored.loc[(pd.Timestamp("2024-01-01"), "AAA"), "prob_buy"] == pytest.approx((0.75 + 0.625) / 2)
    assert scored.loc[(pd.Timestamp("2024-01-02"), "AAA"), "prob_buy"] == pytest.approx(1.0)
    assert scored.loc[(pd.Timestamp("2024-01-02"), "AAA"), "classifier_available_families"] == 1


def test_moe_paper_rule_replay_uses_shifted_top_k_scores() -> None:
    panel = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"],
            "symbol": ["AAA", "BBB", "AAA", "BBB", "AAA", "BBB"],
            "close": [100.0, 100.0, 110.0, 100.0, 120.0, 90.0],
            "prob_buy": [0.9, 0.4, 0.3, 0.8, 0.3, 0.8],
        }
    )

    replay = replay_moe_paper_top_k_rule(
        panel,
        top_k=1,
        threshold=0.5,
        initial_balance=100_000.0,
        fee_bps=0.0,
        slippage_bps=0.0,
    )

    assert replay.action_tape[["date", "symbol", "action", "reason"]].to_dict("records") == [
        {
            "date": pd.Timestamp("2024-01-02"),
            "symbol": "AAA",
            "action": "buy",
            "reason": "entry_moe_top_k",
        },
        {
            "date": pd.Timestamp("2024-01-03"),
            "symbol": "AAA",
            "action": "sell",
            "reason": "exit_moe_top_k_or_invalid",
        },
        {
            "date": pd.Timestamp("2024-01-03"),
            "symbol": "BBB",
            "action": "buy",
            "reason": "entry_moe_top_k",
        },
    ]
    assert replay.trade_windows.loc[0, "symbol"] == "AAA"
    assert replay.trade_windows.loc[0, "ret_dec"] == pytest.approx((120.0 / 110.0) - 1.0)


def test_action_tape_to_trade_windows_ignores_unfunded_buys() -> None:
    actions = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2024-01-02"),
                "symbol": "AAA",
                "action": "buy",
                "price": 100.0,
                "gross_notional": 0.0,
            },
            {
                "date": pd.Timestamp("2024-01-03"),
                "symbol": "AAA",
                "action": "sell",
                "price": 110.0,
                "gross_notional": 0.0,
            },
        ]
    )

    windows = action_tape_to_trade_windows(actions)

    assert windows.empty


def test_action_tape_to_trade_windows_closes_open_positions_at_backtest_end() -> None:
    actions = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2024-01-02"),
                "symbol": "AAA",
                "action": "buy",
                "price": 100.0,
                "score": 0.9,
                "top_k": 1,
            }
        ]
    )
    prices = pd.DataFrame({"AAA": [100.0, 110.0]}, index=pd.to_datetime(["2024-01-02", "2024-01-03"]))

    windows = action_tape_to_trade_windows(actions, prices=prices)

    assert windows[["symbol", "entry_date", "exit_date", "ret_dec", "exit_reason"]].to_dict("records") == [
        {
            "symbol": "AAA",
            "entry_date": pd.Timestamp("2024-01-02"),
            "exit_date": pd.Timestamp("2024-01-03"),
            "ret_dec": pytest.approx(0.10),
            "exit_reason": "end_of_backtest",
        }
    ]


def test_option_portfolio_replay_uses_equity_entry_notional_as_option_budget() -> None:
    selected = pd.DataFrame(
        [
            {
                "trade_id": "t1",
                "symbol": "AAA",
                "entry_date": pd.Timestamp("2024-01-02"),
                "option_exit_date": pd.Timestamp("2024-01-04"),
                "entry_price": 10.0,
                "exit_price": 20.0,
                "equity_entry_notional": 100.0,
                "expired_before_equity_exit": False,
            }
        ]
    )
    paths = pd.DataFrame(
        [
            {"trade_id": "t1", "snapshot_date": pd.Timestamp("2024-01-02"), "mark_price": 10.0},
            {"trade_id": "t1", "snapshot_date": pd.Timestamp("2024-01-03"), "mark_price": 15.0},
            {"trade_id": "t1", "snapshot_date": pd.Timestamp("2024-01-04"), "mark_price": 20.0},
        ]
    )

    replay = replay_option_portfolio_from_selected_paths(
        selected,
        paths,
        date_index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
        initial_balance=1_000.0,
    )

    assert replay.equity.tolist() == pytest.approx([1_000.0, 1_050.0, 1_100.0])
    assert replay.cash.tolist() == pytest.approx([900.0, 900.0, 1_100.0])
    assert replay.trade_ledger["option_pnl_dollars"].iloc[0] == pytest.approx(100.0)
    assert replay.summary["total_return_pct"] == pytest.approx(10.0)


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


def _scored_panel() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    specs = [
        ("2024-01-01", "AAA", 100.0, 0.90, 0.80, 0.20, 0.80),
        ("2024-01-01", "BBB", 100.0, 0.40, 0.70, 0.30, 0.40),
        ("2024-01-02", "AAA", 110.0, 0.95, 0.30, 0.70, 0.95),
        ("2024-01-02", "BBB", 100.0, 0.30, 0.70, 0.30, 0.30),
        ("2024-01-03", "AAA", 120.0, 0.95, 0.30, 0.70, 0.95),
        ("2024-01-03", "BBB", 100.0, 0.20, 0.70, 0.30, 0.20),
    ]
    for date, symbol, close, score, prob_buy, prob_short, pct in specs:
        rows.append(
            {
                "date": pd.Timestamp(date),
                "symbol": symbol,
                "close": close,
                "buy_score_mean_raw_pct6": score,
                "prob_buy": prob_buy,
                "prob_short": prob_short,
                "pred_rf_reg": pct,
                "ae_familiarity": pct,
                "prob_buy_pct": pct,
                "pred_rf_reg_pct": pct,
                "ae_familiarity_pct": pct,
            }
        )
    return pd.DataFrame(rows).set_index(["date", "symbol"]).sort_index()


class _ConstantFeatureModel:
    def __init__(self, feature: str) -> None:
        self._used_features = [feature]

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        values = pd.to_numeric(frame[self._used_features[0]], errors="coerce").fillna(0.0).clip(-2.0, 2.0)
        prob = ((values + 2.0) / 4.0).to_numpy(dtype=float)
        return np.column_stack([1.0 - prob, prob])
