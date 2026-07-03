from __future__ import annotations

import numpy as np
import pandas as pd

from quant_orchestrator.platforms.registry import registry
from quant_orchestrator.platforms.backtesting_frameworks.panel_weight import (
    ExecutionConfig,
    SyntheticOptionsBacktestConfig,
    backtest_panel,
    build_constant_maturity_call_price_panel,
    build_realized_vol_panel,
    build_synthetic_option_return_panels,
    run_synthetic_options_backtest,
    run_top_k_long_short_score_rule,
)
from quant_orchestrator.research_tools.synthetic_options_backtest import (
    SyntheticOptionsBacktestRunConfig,
    run_synthetic_options_backtest_experiment,
)
from quant_orchestrator.platforms.backtesting_frameworks.panel_weight.synthetic_options_experiment import (
    _build_maturing_option_return_panel,
    _build_real_quote_option_return_panel,
)


class BuyFirstSymbolStrategy:
    name = "buy_first"

    def compute_weights(self, panel: pd.DataFrame) -> pd.DataFrame:
        dates = pd.DatetimeIndex(sorted(panel.index.get_level_values("date").unique()))
        symbols = sorted(panel.index.get_level_values("symbol").unique())
        weights = pd.DataFrame(0.0, index=dates, columns=symbols)
        weights.iloc[:, 0] = 1.0
        return weights


def test_panel_weight_provider_is_registered() -> None:
    provider = registry.get("backtesting_framework", "panel_weight")
    assert provider.name == "panel_weight"
    assert "synthetic_options" in provider.capabilities


def test_backtest_panel_runs_weight_engine() -> None:
    panel = _sample_panel()
    result = backtest_panel(
        panel,
        strategy=BuyFirstSymbolStrategy(),
        cfg=ExecutionConfig(fee_bps=0.0, slippage_bps=0.0),
    )
    assert result.strategy_name == "buy_first"
    assert len(result.returns) == 5
    assert result.equity_curve.iloc[-1] > 1.0
    assert "total_return_pct" in result.stats


def test_synthetic_options_build_constant_maturity_returns() -> None:
    close = pd.DataFrame(
        {
            "AAA": [100.0, 101.0, 102.0, 101.0, 103.0],
            "BBB": [50.0, 49.0, 50.0, 51.0, 52.0],
        },
        index=pd.date_range("2024-01-02", periods=5, freq="B"),
    )
    realized_vol = build_realized_vol_panel(close, window=2, vol_floor=0.15, vol_cap=0.8)
    call_prices = build_constant_maturity_call_price_panel(
        close,
        realized_vol,
        strike_multiplier=1.0,
        tenor_days=30,
        premium_floor=0.01,
    )
    assert call_prices.shape == close.shape
    assert np.isfinite(call_prices.to_numpy()).all()
    return_panels, price_panels = build_synthetic_option_return_panels(
        close,
        option_buckets={"atm_option": {"long_strike_multiplier": 1.0, "short_strike_multiplier": 1.0}},
        realized_vol=realized_vol,
        premium_floor=0.01,
    )
    assert set(return_panels) == {"equity", "atm_option"}
    assert set(price_panels["atm_option"]) >= {"call", "put"}


def test_top_k_long_short_rule_produces_positions() -> None:
    panel = _sample_panel()
    run = run_top_k_long_short_score_rule(
        panel=panel,
        long_score_col="prob_buy",
        short_score_col="prob_short",
        long_component_cols=["prob_buy"],
        short_component_cols=["prob_short"],
        component_threshold=0.5,
        price_col="close",
        top_k=1,
    )
    positions = run["positions"]
    assert positions.shape == (5, 2)
    assert positions.abs().sum(axis=1).max() <= 1


def test_synthetic_options_experiment_matches_notebook_summary_shape() -> None:
    panel = _sample_panel()
    result = run_synthetic_options_backtest(
        panel,
        config=SyntheticOptionsBacktestConfig(
            top_k_values=(1,),
            strategy_variants=("classifier_prob", "momentum_21d"),
            option_buckets={
                "atm_option": {
                    "long_strike_multiplier": 1.0,
                    "short_strike_multiplier": 1.0,
                },
            },
            premium_floor=0.01,
            fee_bps=0.0,
            slippage_bps=0.0,
        ),
    )
    assert set(result.summary["strategy"]) == {
        "classifier_prob_long_only",
        "classifier_prob_long_short",
        "momentum_21d",
    }
    assert set(result.summary["instrument"]) == {"equity", "atm_option"}
    assert len(result.summary) == 6
    assert {"buy_count", "sell_count", "short_count", "cover_count"}.issubset(
        result.summary.columns,
    )


def test_maturing_option_returns_expire_to_intrinsic_value() -> None:
    dates = pd.date_range("2024-01-02", periods=4, freq="B")
    positions = pd.DataFrame({"AAA": [1, 1, 1, 1]}, index=dates)
    close = pd.DataFrame({"AAA": [100.0, 100.0, 100.0, 100.0]}, index=dates)
    realized_vol = pd.DataFrame({"AAA": [0.2, 0.2, 0.2, 0.2]}, index=dates)

    returns = _build_maturing_option_return_panel(
        positions,
        close,
        realized_vol,
        strike_multiplier=1.05,
        option_type="call",
        side=1,
        config=SyntheticOptionsBacktestConfig(
            tenor_days=2,
            premium_floor=0.01,
            fee_bps=0.0,
            slippage_bps=0.0,
        ),
    )

    assert returns.iloc[0, 0] == 0.0
    assert returns.iloc[2, 0] == -1.0


def test_real_quote_option_returns_use_cached_contract_quotes() -> None:
    dates = pd.date_range("2024-01-02", periods=4, freq="B")
    positions = pd.DataFrame({"AAA": [1, 1, 1, 1]}, index=dates)
    close = pd.DataFrame({"AAA": [100.0, 101.0, 102.0, 103.0]}, index=dates)
    realized_vol = pd.DataFrame({"AAA": [0.2, 0.2, 0.2, 0.2]}, index=dates)
    chain = pd.DataFrame(
        {
            "snapshot_date": dates,
            "contract_symbol": ["AAA_call_20240105_100"] * 4,
            "expiration": [pd.Timestamp("2024-01-05")] * 4,
            "strike": [100.0] * 4,
            "option_type": ["call"] * 4,
            "bid": [1.0, 2.0, 4.0, 8.0],
            "ask": [1.25, 2.5, 5.0, 10.0],
        },
    )

    returns = _build_real_quote_option_return_panel(
        positions,
        close,
        realized_vol,
        strike_multiplier=1.0,
        option_type="call",
        side=1,
        config=SyntheticOptionsBacktestConfig(
            tenor_days=3,
            option_pricing_mode="real_quotes",
            option_chain_loader=lambda _symbol, _start, _end: chain,
            premium_floor=0.01,
            fee_bps=0.0,
            slippage_bps=0.0,
        ),
        option_chain_cache={},
        coverage_rows=[],
        coverage_context={"strategy": "unit", "instrument": "atm_option", "top_k": 1, "side": "long"},
    )

    np.testing.assert_allclose(returns["AAA"].to_numpy(), np.array([0.0, 0.6, 1.0, 1.0]))


def test_real_quote_option_capacity_scales_unfilled_returns() -> None:
    dates = pd.date_range("2024-01-02", periods=2, freq="B")
    positions = pd.DataFrame({"AAA": [1, 1]}, index=dates)
    close = pd.DataFrame({"AAA": [100.0, 101.0]}, index=dates)
    realized_vol = pd.DataFrame({"AAA": [0.2, 0.2]}, index=dates)
    chain = pd.DataFrame(
        {
            "snapshot_date": dates,
            "contract_symbol": ["AAA_call_20240119_100"] * 2,
            "expiration": [pd.Timestamp("2024-01-19")] * 2,
            "strike": [100.0] * 2,
            "option_type": ["call"] * 2,
            "bid": [9.5, 20.0],
            "ask": [10.0, 21.0],
            "volume": [10.0, 10.0],
            "open_interest": [1000.0, 1000.0],
        },
    )

    returns = _build_real_quote_option_return_panel(
        positions,
        close,
        realized_vol,
        strike_multiplier=1.0,
        option_type="call",
        side=1,
        config=SyntheticOptionsBacktestConfig(
            tenor_days=12,
            option_pricing_mode="real_quotes",
            option_chain_loader=lambda _symbol, _start, _end: chain,
            enforce_option_capacity=True,
            initial_balance=100000.0,
            max_volume_participation=0.10,
            max_open_interest_participation=1.0,
            fee_bps=0.0,
            slippage_bps=0.0,
        ),
        option_chain_cache={},
        coverage_rows=[],
        coverage_context={"strategy": "unit", "instrument": "atm_option", "top_k": 1, "side": "long"},
    )

    np.testing.assert_allclose(returns["AAA"].to_numpy(), np.array([0.0, 0.01]))


def test_synthetic_options_research_runner_writes_artifacts(tmp_path) -> None:
    result = run_synthetic_options_backtest_experiment(
        SyntheticOptionsBacktestRunConfig(
            experiment_name="unit",
            artifact_dir=str(tmp_path),
            backtest=SyntheticOptionsBacktestConfig(
                top_k_values=(1,),
                strategy_variants=("classifier_prob",),
                option_buckets={
                    "atm_option": {
                        "long_strike_multiplier": 1.0,
                        "short_strike_multiplier": 1.0,
                    },
                },
                premium_floor=0.01,
            ),
        ),
        bt_panel=_sample_panel(),
    )
    assert result.artifact_paths["summary"].exists()
    assert result.artifact_paths["yearly_summary"].exists()
    assert result.artifact_paths["positions"].is_dir()
    saved = pd.read_csv(result.artifact_paths["summary"])
    assert len(saved) == len(result.result.summary)


def _sample_panel() -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=5, freq="B")
    rows = []
    for date in dates:
        rows.append(
            {
                "date": date,
                "symbol": "AAA",
                "close": 100.0 + len(rows),
                "prob_buy": 0.8,
                "prob_short": 0.2,
            },
        )
        rows.append(
            {
                "date": date,
                "symbol": "BBB",
                "close": 50.0 + len(rows),
                "prob_buy": 0.3,
                "prob_short": 0.7,
            },
        )
    return pd.DataFrame(rows).set_index(["date", "symbol"]).sort_index()
