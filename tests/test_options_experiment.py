from __future__ import annotations

import pandas as pd
import pytest

from quant_orchestrator.research_tools.options_experiment import (
    OracleOptionExperimentConfig,
    OptionRetrievalConfig,
    OptionMvBasketConfig,
    OptopsyExecutionConfig,
    OptionWindowBuildConfig,
    build_classifier_signal_trade_windows,
    build_option_window_dataset,
    estimate_option_runtime_scaling,
    load_option_experiment_artifacts,
    option_experiment_artifact_dir,
    rank_option_window_strategy_sources,
    write_oracle_option_artifacts,
    _choose_weighted_basket_per_trade,
    _OptionRetriever,
    _normalize_trade_windows,
    _option_feature_coverage,
    _build_portfolio_fraction_trade_log,
    _selector_summaries,
    _run_optopsy_selector_backtests,
    _selected_basket_to_optopsy_raw,
    _selected_options_to_optopsy_raw,
    _selected_actions_to_trade_rows,
    _source_family_diagnostics,
)


def test_selected_options_are_mapped_to_optopsy_raw_schema() -> None:
    selected = pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "entry_date": ["2025-01-02"],
            "option_exit_date": ["2025-01-20"],
            "expiration": ["2025-02-21"],
            "entry_mid": [2.0],
            "exit_mid": [3.0],
            "option_type": ["call"],
            "strike": [100.0],
        }
    )

    raw = _selected_options_to_optopsy_raw(selected)

    assert raw.loc[0, "underlying_symbol"] == "AAPL"
    assert raw.loc[0, "quote_date_entry"] == pd.Timestamp("2025-01-02")
    assert raw.loc[0, "_early_exit_date"] == pd.Timestamp("2025-01-20")
    assert raw.loc[0, "pct_change"] == 0.5


def test_selector_backtest_uses_optopsy_execution_summary() -> None:
    selected = pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "entry_date": ["2025-01-02"],
            "option_exit_date": ["2025-01-20"],
            "expiration": ["2025-02-21"],
            "entry_mid": [2.0],
            "exit_mid": [3.0],
            "option_type": ["call"],
            "strike": [100.0],
        }
    )

    summary = _run_optopsy_selector_backtests(
        {"model_ranker": selected},
        OptopsyExecutionConfig(capital=100_000.0, quantity=1, max_positions=5, multiplier=100, sizing_mode="fixed_quantity"),
    )

    assert summary.loc[0, "framework"] == "optopsy"
    assert summary.loc[0, "closed_trades"] == 1
    assert summary.loc[0, "total_pnl"] == 100.0
    assert summary.loc[0, "final_equity"] == 100_100.0


def test_selector_backtest_defaults_to_portfolio_fraction_sizing() -> None:
    selected = pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "entry_date": ["2025-01-02"],
            "option_exit_date": ["2025-01-20"],
            "expiration": ["2025-02-21"],
            "entry_mid": [2.0],
            "exit_mid": [3.0],
            "option_type": ["call"],
            "strike": [100.0],
        }
    )

    summary = _run_optopsy_selector_backtests(
        {"model_ranker": selected},
        OptopsyExecutionConfig(capital=100_000.0, max_positions=5, multiplier=100),
    )

    assert summary.loc[0, "framework"] == "optopsy"
    assert summary.loc[0, "closed_trades"] == 1
    assert summary.loc[0, "total_pnl"] == 10_000.0
    assert summary.loc[0, "final_equity"] == 110_000.0


def test_selector_backtest_uses_top_k_for_portfolio_fraction_sizing() -> None:
    selected = pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "entry_date": ["2025-01-02"],
            "option_exit_date": ["2025-01-20"],
            "expiration": ["2025-02-21"],
            "entry_mid": [2.0],
            "exit_mid": [3.0],
            "option_type": ["call"],
            "strike": [100.0],
            "top_k": [10],
        }
    )

    summary = _run_optopsy_selector_backtests(
        {"model_ranker": selected},
        OptopsyExecutionConfig(capital=100_000.0, max_positions=5, multiplier=100),
    )

    assert summary.loc[0, "total_pnl"] == 5_000.0
    assert summary.loc[0, "final_equity"] == 105_000.0


def test_selector_summaries_include_simplified_model_selectors() -> None:
    frame = pd.DataFrame(
        {
            "trade_id": ["t1", "t1", "t2", "t2"],
            "symbol": ["AAPL", "AAPL", "MSFT", "MSFT"],
            "side": ["long", "long", "short", "short"],
            "pred_return": [0.1, 0.2, 0.4, 0.3],
            "pred_rank_score": [0.7, 0.6, 0.2, 0.9],
            "option_return": [0.5, 0.1, 0.2, 0.8],
            "dte_gap": [1.0, 5.0, 2.0, 3.0],
            "abs_moneyness": [0.01, 0.10, 0.05, 0.04],
            "spread_pct": [0.02, 0.03, 0.01, 0.04],
            "liquidity_score": [10.0, 100.0, 50.0, 5.0],
            "dte": [40, 45, 42, 48],
        }
    )

    summary, _, selected = _selector_summaries(frame)

    assert set(summary["selector"]) == {
        "rule_atm_90d",
        "model_ranker",
        "oracle_best_possible",
        "model_mv_basket",
        "oracle_mv_basket",
    }
    assert selected["rule_atm_90d"].set_index("trade_id").loc["t1", "dte_gap"] == 1.0
    assert selected["model_ranker"].set_index("trade_id").loc["t1", "pred_return"] == 0.2
    assert selected["oracle_best_possible"].set_index("trade_id").loc["t1", "option_return"] == 0.5
    assert selected["oracle_best_possible"].set_index("trade_id").loc["t2", "option_return"] == 0.8


def test_selector_summaries_do_not_use_oracle_when_model_predictions_missing() -> None:
    frame = pd.DataFrame(
        {
            "trade_id": ["t1", "t1"],
            "symbol": ["AAPL", "AAPL"],
            "side": ["long", "long"],
            "pred_return": [float("nan"), float("nan")],
            "pred_rank_score": [float("nan"), float("nan")],
            "option_return": [0.5, 1.0],
            "dte_gap": [1.0, 2.0],
            "abs_moneyness": [0.01, 0.02],
            "spread_pct": [0.01, 0.02],
            "liquidity_score": [10.0, 20.0],
            "dte": [40, 41],
        }
    )

    summary, _, selected = _selector_summaries(frame)

    assert selected["model_ranker"].empty
    assert summary.set_index("selector").loc["model_ranker", "trades"] == 0
    assert summary.set_index("selector").loc["oracle_best_possible", "trades"] == 1


def test_selector_summaries_respect_mv_basket_limits() -> None:
    frame = pd.DataFrame(
        {
            "trade_id": ["t1", "t1", "t1", "t1"],
            "symbol": ["AAPL"] * 4,
            "side": ["long"] * 4,
            "pred_return": [0.1, 0.2, 0.3, 0.4],
            "option_return": [0.1, 0.2, 0.3, 0.4],
            "pred_mv_weight": [0.7, 0.2, 0.05, 0.01],
            "mv_weight": [0.25, 0.25, 0.25, 0.25],
            "dte_gap": [1.0, 2.0, 3.0, 4.0],
            "abs_moneyness": [0.01, 0.02, 0.03, 0.04],
            "spread_pct": [0.01, 0.02, 0.03, 0.04],
            "liquidity_score": [10.0, 20.0, 30.0, 40.0],
            "dte": [40, 41, 42, 43],
        }
    )

    _, _, selected = _selector_summaries(
        frame,
        mv_basket_config=OptionMvBasketConfig(max_legs=2, min_predicted_weight=0.1),
    )

    assert len(selected["model_mv_basket"]) == 2
    assert set(selected["model_mv_basket"]["pred_mv_weight"]) == {0.7, 0.2}


def test_classifier_signal_trade_windows_are_event_only() -> None:
    scores = pd.DataFrame(
        {
            "strategy_source": ["ensemble_mean"] * 4,
            "source": ["ensemble"] * 4,
            "family": ["mean"] * 4,
            "symbol": ["AAPL", "MSFT", "AAPL", "MSFT"],
            "date": ["2025-01-02", "2025-01-02", "2025-01-03", "2025-01-03"],
            "long_score": [0.8, 0.4, 0.2, 0.3],
            "short_score": [0.1, 0.2, 0.9, 0.2],
            "long_exit_score": [0.1, 0.2, 0.2, 0.2],
            "short_exit_score": [0.1, 0.2, 0.95, 0.2],
            "ae_familiarity": [0.9, 0.9, 0.9, 0.9],
        }
    )

    trades = build_classifier_signal_trade_windows(
        scores,
        strategy_sources=("ensemble_mean",),
        variant="long_short",
        top_k=1,
        entry_threshold=0.5,
        exit_threshold=0.5,
    )

    assert len(trades) == 1
    assert trades.loc[0, "symbol"] == "AAPL"
    assert trades.loc[0, "side"] == "long"
    assert trades.loc[0, "entry_date"] == pd.Timestamp("2025-01-02")
    assert trades.loc[0, "exit_date"] == pd.Timestamp("2025-01-03")
    assert "MSFT" not in set(trades["symbol"])


def test_rank_option_window_strategy_sources_excludes_ensemble_and_sorts_quality() -> None:
    summary = pd.DataFrame(
        {
            "framework": ["zipline", "zipline", "zipline", "zipline"],
            "strategy_source": ["ensemble_mean", "fmp.a", "fmp.b", "fmp.c"],
            "source": ["ensemble", "fmp", "fmp", "fmp"],
            "family": ["mean", "a", "b", "c"],
            "variant": ["long_short", "long_short", "long_short", "long_only"],
            "top_k": [5, 5, 5, 5],
            "signal_events": [10, 0, 3, 9],
            "sharpe": [5.0, 2.0, 1.5, 4.0],
            "total_return": [10.0, 3.0, 2.0, 8.0],
            "max_drawdown": [-0.1, -0.2, -0.3, -0.4],
        }
    )

    ranked = rank_option_window_strategy_sources(summary, framework="zipline", min_signal_events=1)

    assert ranked["strategy_source"].tolist() == ["fmp.b"]


def test_build_option_window_dataset_creates_standard_groups() -> None:
    scores = pd.DataFrame(
        {
            "strategy_source": ["ensemble_mean", "fmp.a", "fmp.b", "ensemble_mean", "fmp.a", "fmp.b"],
            "source": ["ensemble", "fmp", "fmp", "ensemble", "fmp", "fmp"],
            "family": ["mean", "a", "b", "mean", "a", "b"],
            "symbol": ["AAPL", "AAPL", "MSFT", "AAPL", "AAPL", "MSFT"],
            "date": ["2025-01-02", "2025-01-02", "2025-01-02", "2025-01-03", "2025-01-03", "2025-01-03"],
            "long_score": [0.8, 0.7, 0.1, 0.2, 0.1, 0.2],
            "short_score": [0.1, 0.2, 0.7, 0.9, 0.8, 0.1],
            "long_exit_score": [0.8, 0.7, 0.1, 0.2, 0.1, 0.2],
            "short_exit_score": [0.1, 0.2, 0.7, 0.9, 0.8, 0.1],
            "ae_familiarity": [1.0] * 6,
        }
    )
    summary = pd.DataFrame(
        {
            "framework": ["zipline", "zipline"],
            "strategy_source": ["fmp.a", "fmp.b"],
            "source": ["fmp", "fmp"],
            "family": ["a", "b"],
            "variant": ["long_short", "long_short"],
            "top_k": [1, 1],
            "signal_events": [2, 2],
            "sharpe": [1.0, 0.5],
            "total_return": [0.2, 0.1],
            "max_drawdown": [-0.1, -0.2],
        }
    )

    dataset = build_option_window_dataset(
        scores,
        backtest_summary=summary,
        config=OptionWindowBuildConfig(top_k=1, top_family_count=1, ranking_framework="zipline"),
    )

    assert set(dataset.windows_by_group) == {"individual__fmp.a", "individual__fmp.b", "all_feature_families"}
    assert dataset.source_groups["individual__fmp.a"] == ("fmp.a",)
    assert dataset.window_summary.set_index("group").loc["all_feature_families", "sources"] == 2


def test_option_retriever_full_chain_actions_include_long_calls_and_short_puts_with_collateral_returns() -> None:
    def fake_read(symbol, *, start_date, end_date, columns):
        if pd.Timestamp(start_date) == pd.Timestamp("2025-01-02"):
            return pd.DataFrame(
                {
                    "snapshot_date": [start_date, start_date],
                    "contract_symbol": ["AAPL_C_100", "AAPL_P_100"],
                    "expiration": [pd.Timestamp("2025-04-18"), pd.Timestamp("2025-04-18")],
                    "strike": [100.0, 100.0],
                    "option_type": ["call", "put"],
                    "bid": [5.0, 4.0],
                    "ask": [5.5, 4.5],
                    "mid": [5.25, 4.25],
                    "dte": [106, 106],
                    "spread_pct": [0.1, 0.12],
                }
            )
        return pd.DataFrame(
            {
                "snapshot_date": [start_date, start_date],
                "contract_symbol": ["AAPL_C_100", "AAPL_P_100"],
                "expiration": [pd.Timestamp("2025-04-18"), pd.Timestamp("2025-04-18")],
                "strike": [100.0, 100.0],
                "option_type": ["call", "put"],
                "bid": [7.0, 2.0],
                "ask": [7.5, 2.5],
                "mid": [7.25, 2.25],
                "dte": [88, 88],
                "spread_pct": [0.07, 0.22],
            }
        )

    class _Features:
        def __init__(self, frame):
            self.df = frame

    def fake_features(frame, *, underlying_price=None, target_dte=None, compute_model_greeks=True):
        out = frame.copy()
        out["dte_gap"] = (out["dte"] - int(target_dte or 90)).abs()
        out["abs_moneyness"] = (out["strike"] / float(underlying_price) - 1.0).abs()
        return _Features(out)

    prices = {
        "AAPL": pd.DataFrame(
            {"close": [100.0, 110.0]},
            index=[pd.Timestamp("2025-01-02"), pd.Timestamp("2025-01-20")],
        )
    }
    retriever = _OptionRetriever(
        OptionRetrievalConfig(option_universe="full_chain_actions", target_dte=90, exit_lookback_days=0),
        price_frames=prices,
        read_option_chain_arctic=fake_read,
        build_option_contract_features=fake_features,
    )

    rows = retriever.retrieve(
        pd.Series(
            {
                "symbol": "AAPL",
                "side": "long",
                "entry_date": pd.Timestamp("2025-01-02"),
                "exit_date": pd.Timestamp("2025-01-20"),
            }
        )
    )

    by_action = rows.set_index("option_action")
    assert set(by_action.index) == {"buy_call", "sell_put"}
    assert by_action.loc["buy_call", "option_return"] == pytest.approx((7.0 - 5.5) / 5.5)
    assert by_action.loc["sell_put", "option_return"] == pytest.approx((4.0 - 2.5) / 100.0)
    assert by_action.loc["sell_put", "return_denominator"] == pytest.approx(100.0)


def test_option_retriever_uses_last_contract_quote_before_expiration_intrinsic() -> None:
    def fake_read(symbol, *, start_date, end_date, columns):
        date = pd.Timestamp(start_date).normalize()
        if date == pd.Timestamp("2021-02-03"):
            return pd.DataFrame(
                {
                    "snapshot_date": [date],
                    "contract_symbol": ["GOOG_put_20210205_2015"],
                    "expiration": [pd.Timestamp("2021-02-05")],
                    "strike": [2015.0],
                    "option_type": ["put"],
                    "bid": [1.8],
                    "ask": [2.5],
                    "mid": [2.15],
                    "dte": [2],
                    "spread_pct": [0.25],
                }
            )
        if date == pd.Timestamp("2021-02-04"):
            return pd.DataFrame(
                {
                    "snapshot_date": [date],
                    "contract_symbol": ["GOOG_put_20210205_2015"],
                    "expiration": [pd.Timestamp("2021-02-05")],
                    "strike": [2015.0],
                    "option_type": ["put"],
                    "bid": [0.10],
                    "ask": [1.20],
                    "mid": [0.65],
                    "dte": [1],
                    "spread_pct": [1.69],
                }
            )
        return pd.DataFrame()

    class _Features:
        def __init__(self, frame):
            self.df = frame

    def fake_features(frame, *, underlying_price=None, target_dte=None, compute_model_greeks=True):
        out = frame.copy()
        out["dte_gap"] = (pd.to_numeric(out["dte"], errors="coerce") - int(target_dte or 2)).abs()
        out["abs_moneyness"] = (pd.to_numeric(out["strike"], errors="coerce") / float(underlying_price) - 1.0).abs()
        return _Features(out)

    prices = {
        "GOOG": pd.DataFrame(
            {"close": [102.66, 102.29, 104.05]},
            index=[pd.Timestamp("2021-02-03"), pd.Timestamp("2021-02-04"), pd.Timestamp("2021-02-05")],
        )
    }
    retriever = _OptionRetriever(
        OptionRetrievalConfig(
            option_universe="filtered",
            min_dte=1,
            max_dte=5,
            target_dte=2,
            max_abs_moneyness=100.0,
            min_entry_mid=0.0,
            max_entry_spread_pct=10.0,
            exit_lookback_days=3,
        ),
        price_frames=prices,
        read_option_chain_arctic=fake_read,
        build_option_contract_features=fake_features,
    )

    rows = retriever.retrieve(
        pd.Series(
            {
                "symbol": "GOOG",
                "side": "short",
                "entry_date": pd.Timestamp("2021-02-03"),
                "exit_date": pd.Timestamp("2021-02-16"),
            }
        )
    )

    assert len(rows) == 1
    row = rows.iloc[0]
    assert row["option_exit_date"] == pd.Timestamp("2021-02-04")
    assert row["exit_price_source"] == "last_contract_quote"
    assert row["exit_price"] == pytest.approx(0.10)
    assert row["option_return"] == pytest.approx((0.10 - 2.5) / 2.5)


def test_option_retriever_blocks_scale_mismatched_intrinsic_fallback() -> None:
    def fake_read(symbol, *, start_date, end_date, columns):
        date = pd.Timestamp(start_date).normalize()
        if date == pd.Timestamp("2021-02-03"):
            return pd.DataFrame(
                {
                    "snapshot_date": [date],
                    "contract_symbol": ["GOOG_put_20210205_2015"],
                    "expiration": [pd.Timestamp("2021-02-05")],
                    "strike": [2015.0],
                    "option_type": ["put"],
                    "bid": [1.8],
                    "ask": [2.5],
                    "mid": [2.15],
                    "dte": [2],
                    "spread_pct": [0.25],
                }
            )
        return pd.DataFrame()

    class _Features:
        def __init__(self, frame):
            self.df = frame

    def fake_features(frame, *, underlying_price=None, target_dte=None, compute_model_greeks=True):
        out = frame.copy()
        out["dte_gap"] = 0
        out["abs_moneyness"] = (
            pd.to_numeric(out["strike"], errors="coerce")
            / float(underlying_price)
            - 1.0
        ).abs()
        return _Features(out)

    prices = {
        "GOOG": pd.DataFrame(
            {"close": [102.66, 104.05]},
            index=[pd.Timestamp("2021-02-03"), pd.Timestamp("2021-02-05")],
        )
    }
    retriever = _OptionRetriever(
        OptionRetrievalConfig(
            option_universe="filtered",
            min_dte=1,
            max_dte=5,
            target_dte=2,
            max_abs_moneyness=100.0,
            min_entry_mid=0.0,
            max_entry_spread_pct=10.0,
            exit_lookback_days=3,
        ),
        price_frames=prices,
        read_option_chain_arctic=fake_read,
        build_option_contract_features=fake_features,
    )

    rows = retriever.retrieve(
        pd.Series(
            {
                "symbol": "GOOG",
                "side": "short",
                "entry_date": pd.Timestamp("2021-02-03"),
                "exit_date": pd.Timestamp("2021-02-16"),
            }
        )
    )

    assert rows.empty


def test_option_retriever_uses_raw_underlying_prices_for_intrinsic_fallback() -> None:
    def fake_read(symbol, *, start_date, end_date, columns):
        date = pd.Timestamp(start_date).normalize()
        if date == pd.Timestamp("2021-02-03"):
            return pd.DataFrame(
                {
                    "snapshot_date": [date],
                    "contract_symbol": ["GOOG_put_20210205_2015"],
                    "expiration": [pd.Timestamp("2021-02-05")],
                    "strike": [2015.0],
                    "option_type": ["put"],
                    "bid": [1.8],
                    "ask": [2.5],
                    "mid": [2.15],
                    "dte": [2],
                    "spread_pct": [0.25],
                }
            )
        return pd.DataFrame()

    class _Features:
        def __init__(self, frame):
            self.df = frame

    def fake_features(frame, *, underlying_price=None, target_dte=None, compute_model_greeks=True):
        out = frame.copy()
        out["dte_gap"] = 0
        out["abs_moneyness"] = (
            pd.to_numeric(out["strike"], errors="coerce")
            / float(underlying_price)
            - 1.0
        ).abs()
        return _Features(out)

    adjusted_prices = {
        "GOOG": pd.DataFrame(
            {"close": [102.66, 104.05]},
            index=[pd.Timestamp("2021-02-03"), pd.Timestamp("2021-02-05")],
        )
    }
    raw_prices = {
        "GOOG": pd.DataFrame(
            {"close": [2010.0, 2000.0]},
            index=[pd.Timestamp("2021-02-03"), pd.Timestamp("2021-02-05")],
        )
    }
    retriever = _OptionRetriever(
        OptionRetrievalConfig(
            option_universe="filtered",
            min_dte=1,
            max_dte=5,
            target_dte=2,
            max_abs_moneyness=0.1,
            min_entry_mid=0.0,
            max_entry_spread_pct=10.0,
            exit_lookback_days=3,
        ),
        price_frames=adjusted_prices,
        option_price_frames=raw_prices,
        read_option_chain_arctic=fake_read,
        build_option_contract_features=fake_features,
    )

    rows = retriever.retrieve(
        pd.Series(
            {
                "symbol": "GOOG",
                "side": "short",
                "entry_date": pd.Timestamp("2021-02-03"),
                "exit_date": pd.Timestamp("2021-02-16"),
            }
        )
    )

    assert len(rows) == 1
    row = rows.iloc[0]
    assert row["exit_price_source"] == "expiration_intrinsic"
    assert row["exit_price"] == pytest.approx(15.0)
    assert retriever.metrics()["underlying_raw_price_frame_count"] == 1.0
    assert retriever.metrics()["underlying_adjusted_fallback_count"] == 0.0


def test_option_retriever_uses_chain_underlying_price_for_entry_features() -> None:
    feature_spots = []

    def fake_read(symbol, *, start_date, end_date, columns):
        assert "underlying_price" in columns
        date = pd.Timestamp(start_date).normalize()
        if date != pd.Timestamp("2021-02-03"):
            return pd.DataFrame()
        return pd.DataFrame(
            {
                "snapshot_date": [date],
                "contract_symbol": ["GOOG_call_20210205_2075"],
                "expiration": [pd.Timestamp("2021-02-05")],
                "strike": [2075.0],
                "option_type": ["call"],
                "bid": [10.0],
                "ask": [11.0],
                "mid": [10.5],
                "dte": [2],
                "spread_pct": [0.05],
                "underlying_price": [2070.07],
            }
        )

    class _Features:
        def __init__(self, frame):
            self.df = frame

    def fake_features(frame, *, underlying_price=None, target_dte=None, compute_model_greeks=True):
        feature_spots.append(underlying_price)
        out = frame.copy()
        out["dte_gap"] = 0
        out["abs_moneyness"] = (
            pd.to_numeric(out["strike"], errors="coerce")
            / float(underlying_price)
            - 1.0
        ).abs()
        return _Features(out)

    retriever = _OptionRetriever(
        OptionRetrievalConfig(
            option_universe="filtered",
            min_dte=1,
            max_dte=5,
            target_dte=2,
            max_abs_moneyness=0.1,
            min_entry_mid=0.0,
            max_entry_spread_pct=10.0,
            exit_lookback_days=3,
        ),
        price_frames={},
        read_option_chain_arctic=fake_read,
        build_option_contract_features=fake_features,
    )

    rows = retriever.retrieve(
        pd.Series(
            {
                "symbol": "GOOG",
                "side": "long",
                "entry_date": pd.Timestamp("2021-02-03"),
                "exit_date": pd.Timestamp("2021-02-16"),
            }
        ),
        price_exit=False,
    )

    assert len(rows) == 1
    assert len(feature_spots) == 2
    assert feature_spots[0] == pytest.approx(2070.07)
    assert feature_spots[1] == pytest.approx(2070.07)
    assert retriever.metrics()["underlying_lookup_count"] == 0.0
    assert retriever.metrics()["underlying_adjusted_fallback_count"] == 0.0


def test_selected_actions_to_trade_rows_uses_action_aware_denominator() -> None:
    selected = pd.DataFrame(
        {
            "trade_id": ["t1"],
            "symbol": ["AAPL"],
            "entry_date": [pd.Timestamp("2025-01-02")],
            "option_exit_date": [pd.Timestamp("2025-01-20")],
            "expiration": [pd.Timestamp("2025-04-18")],
            "return_denominator": [100.0],
            "option_pnl": [1.5],
            "option_return": [0.015],
            "option_action": ["sell_put"],
            "top_k": [5],
        }
    )

    rows = _selected_actions_to_trade_rows(selected)

    assert rows.loc[0, "entry_cost"] == pytest.approx(100.0)
    assert rows.loc[0, "exit_proceeds"] == pytest.approx(101.5)
    assert rows.loc[0, "pct_change"] == pytest.approx(0.015)


def test_multiple_classifier_sources_trade_as_one_mean_planner_stream() -> None:
    scores = pd.DataFrame(
        {
            "strategy_source": ["fmp.a", "fmp.b", "fmp.a", "fmp.b"] * 3,
            "source": ["fmp"] * 12,
            "family": ["a", "b", "a", "b"] * 3,
            "symbol": ["AAPL", "AAPL", "MSFT", "MSFT"] * 3,
            "date": ["2025-01-02"] * 4 + ["2025-01-03"] * 4 + ["2025-01-04"] * 4,
            "long_score": [0.9, 0.8, 0.7, 0.6, 0.2, 0.3, 0.8, 0.7, 0.2, 0.3, 0.2, 0.3],
            "short_score": [0.1, 0.2, 0.2, 0.1, 0.9, 0.8, 0.1, 0.2, 0.8, 0.7, 0.9, 0.8],
            "long_exit_score": [0.9, 0.8, 0.7, 0.6, 0.2, 0.3, 0.8, 0.7, 0.2, 0.3, 0.2, 0.3],
            "short_exit_score": [0.1, 0.2, 0.2, 0.1, 0.9, 0.8, 0.1, 0.2, 0.8, 0.7, 0.9, 0.8],
            "ae_familiarity": [1.0] * 12,
        }
    )

    trades = build_classifier_signal_trade_windows(
        scores,
        strategy_sources=("fmp.a", "fmp.b"),
        variant="long_short",
        top_k=1,
        entry_threshold=0.5,
        exit_threshold=0.5,
    )

    assert len(trades) == 2
    assert trades["strategy_source"].nunique() == 1
    assert trades["strategy_source"].iloc[0] == "ensemble_2_sources"
    assert trades["entry_date"].tolist() == [pd.Timestamp("2025-01-02"), pd.Timestamp("2025-01-03")]
    assert trades["side"].tolist() == ["long", "short"]


def test_classifier_trade_windows_require_unanimous_selected_classifier_entry_and_exit_on_any_disagreement() -> None:
    scores = pd.DataFrame(
        {
            "strategy_source": ["fmp.a", "fmp.b", "fmp.a", "fmp.b", "fmp.a", "fmp.b"],
            "source": ["fmp"] * 6,
            "family": ["a", "b"] * 3,
            "symbol": ["AAPL", "AAPL"] * 3,
            "date": ["2025-01-02", "2025-01-02", "2025-01-03", "2025-01-03", "2025-01-04", "2025-01-04"],
            "long_score": [0.9, 0.9, 0.9, 0.8, 0.9, 0.9],
            "short_score": [0.1, 0.9, 0.1, 0.2, 0.1, 0.9],
            "long_exit_score": [0.9, 0.1, 0.9, 0.8, 0.9, 0.1],
            "short_exit_score": [0.1, 0.9, 0.1, 0.2, 0.1, 0.9],
            "long_agree_count": [1, 0, 1, 1, 1, 0],
            "short_agree_count": [0, 1, 0, 0, 0, 1],
            "model_count": [1] * 6,
            "ae_familiarity": [1.0] * 6,
        }
    )

    trades = build_classifier_signal_trade_windows(
        scores,
        strategy_sources=("fmp.a", "fmp.b"),
        variant="long_short",
        top_k=1,
        entry_threshold=0.5,
        exit_threshold=0.5,
    )

    assert len(trades) == 1
    assert trades.loc[0, "entry_date"] == pd.Timestamp("2025-01-03")
    assert trades.loc[0, "exit_date"] == pd.Timestamp("2025-01-04")
    assert trades.loc[0, "side"] == "long"


def test_option_retriever_caches_repeated_symbol_date_chain_reads() -> None:
    calls = []

    def fake_read(symbol, *, start_date, end_date, columns):
        calls.append((symbol, pd.Timestamp(start_date)))
        return pd.DataFrame(
            {
                "snapshot_date": [start_date],
                "contract_symbol": ["AAPL_C_100"],
                "expiration": [pd.Timestamp("2025-03-21")],
                "strike": [100.0],
                "option_type": ["call"],
                "bid": [1.0],
                "ask": [1.2],
                "mid": [1.1],
                "dte": [45],
                "spread_pct": [0.1],
            }
        )

    class _Features:
        def __init__(self, frame):
            self.df = frame

    def fake_features(frame, *, underlying_price=None, target_dte=None, compute_model_greeks=True):
        out = frame.copy()
        out["dte_gap"] = 0.0
        return _Features(out)

    prices = {"AAPL": pd.DataFrame({"close": [100.0]}, index=[pd.Timestamp("2025-01-02")])}
    retriever = _OptionRetriever(
        OptionRetrievalConfig(max_candidates_per_trade=1, exit_lookback_days=0),
        price_frames=prices,
        read_option_chain_arctic=fake_read,
        build_option_contract_features=fake_features,
    )

    retriever._load_day_chain("AAPL", pd.Timestamp("2025-01-02"))
    retriever._load_day_chain("AAPL", pd.Timestamp("2025-01-02"))

    assert len(calls) == 1
    assert retriever.metrics()["option_chain_cache_hits"] == 1.0


def test_option_retriever_exit_quote_lookup_skips_feature_engineering() -> None:
    calls = []
    feature_calls = []

    def fake_read(symbol, *, start_date, end_date, columns):
        calls.append(
            {
                "symbol": symbol,
                "start_date": pd.Timestamp(start_date),
                "end_date": pd.Timestamp(end_date),
                "columns": tuple(columns),
            }
        )
        return pd.DataFrame(
            {
                "snapshot_date": [start_date, start_date],
                "contract_symbol": ["AAPL_C_100", "AAPL_C_100"],
                "bid": [1.0, 1.2],
                "ask": [1.4, 1.6],
                "mid": [None, None],
            }
        )

    class _Features:
        def __init__(self, frame):
            self.df = frame

    def fake_features(frame, *, underlying_price=None, target_dte=None, compute_model_greeks=True):
        feature_calls.append(len(frame))
        return _Features(frame.copy())

    retriever = _OptionRetriever(
        OptionRetrievalConfig(max_candidates_per_trade=1, exit_lookback_days=0),
        price_frames={},
        read_option_chain_arctic=fake_read,
        build_option_contract_features=fake_features,
    )

    quote = retriever._chain_quote("AAPL", pd.Timestamp("2025-01-03"), "AAPL_C_100")
    cached_quote = retriever._chain_quote("AAPL", pd.Timestamp("2025-01-03"), "AAPL_C_100")

    assert quote == pytest.approx((1.2, 1.6, 1.4))
    assert cached_quote == pytest.approx((1.2, 1.6, 1.4))
    assert feature_calls == []
    assert calls == [
        {
            "symbol": "AAPL",
            "start_date": pd.Timestamp("2025-01-03"),
            "end_date": pd.Timestamp("2025-01-03"),
            "columns": ("snapshot_date", "contract_symbol", "bid", "ask", "mid", "underlying_price"),
        }
    ]
    metrics = retriever.metrics()
    assert metrics["option_quote_chain_read_count"] == 1.0
    assert metrics["chain_quote_cache_hits"] == 1.0
    assert metrics["option_quote_chain_duplicate_rows_dropped"] == 1.0
    assert metrics["option_chain_read_count"] == 0.0


def test_option_retriever_quote_lookup_caches_chain_underlying_price() -> None:
    def fake_read(symbol, *, start_date, end_date, columns):
        assert tuple(columns) == ("snapshot_date", "contract_symbol", "bid", "ask", "mid", "underlying_price")
        return pd.DataFrame(
            {
                "snapshot_date": [start_date],
                "contract_symbol": ["AAPL_C_100"],
                "bid": [1.0],
                "ask": [1.4],
                "mid": [1.2],
                "underlying_price": [101.5],
            }
        )

    class _Features:
        def __init__(self, frame):
            self.df = frame

    def fake_features(frame, *, underlying_price=None, target_dte=None, compute_model_greeks=True):
        return _Features(frame.copy())

    retriever = _OptionRetriever(
        OptionRetrievalConfig(max_candidates_per_trade=1, exit_lookback_days=0),
        price_frames={},
        read_option_chain_arctic=fake_read,
        build_option_contract_features=fake_features,
    )

    assert retriever._chain_quote("AAPL", pd.Timestamp("2025-01-03"), "AAPL_C_100") == pytest.approx((1.0, 1.4, 1.2))
    assert retriever._underlying_close("AAPL", pd.Timestamp("2025-01-03")) == pytest.approx(101.5)
    assert retriever.metrics()["underlying_cache_hits"] == 1.0


def test_option_retriever_filters_contracts_that_allocation_cannot_afford() -> None:
    def fake_read(symbol, *, start_date, end_date, columns):
        return pd.DataFrame(
            {
                "snapshot_date": [start_date, start_date],
                "contract_symbol": ["AAPL_C_100_CHEAP", "AAPL_C_100_EXPENSIVE"],
                "expiration": [pd.Timestamp("2025-03-21"), pd.Timestamp("2025-03-21")],
                "strike": [100.0, 100.0],
                "option_type": ["call", "call"],
                "bid": [1.9, 49.5],
                "ask": [2.1, 50.5],
                "mid": [2.0, 50.0],
                "dte": [45, 45],
                "spread_pct": [0.05, 0.02],
            }
        )

    class _Features:
        def __init__(self, frame):
            self.df = frame

    def fake_features(frame, *, underlying_price=None, target_dte=None, compute_model_greeks=True):
        out = frame.copy()
        out["dte_gap"] = 0.0
        return _Features(out)

    prices = {
        "AAPL": pd.DataFrame(
            {"close": [100.0, 110.0]},
            index=[pd.Timestamp("2025-01-02"), pd.Timestamp("2025-01-20")],
        )
    }
    retriever = _OptionRetriever(
        OptionRetrievalConfig(max_candidates_per_trade=10, exit_lookback_days=0),
        execution=OptopsyExecutionConfig(capital=100_000.0, max_positions=5, multiplier=100),
        price_frames=prices,
        read_option_chain_arctic=fake_read,
        build_option_contract_features=fake_features,
    )
    trade = pd.Series(
        {
            "symbol": "AAPL",
            "side": "long",
            "entry_date": pd.Timestamp("2025-01-02"),
            "exit_date": pd.Timestamp("2025-01-20"),
            "top_k": 100,
        }
    )

    rows = retriever.retrieve(trade)

    assert rows["contract_symbol"].tolist() == ["AAPL_C_100_CHEAP"]
    assert rows.loc[0, "allocation_fraction"] == pytest.approx(0.01)
    assert rows.loc[0, "allocation_budget"] == pytest.approx(1_000.0)
    assert rows.loc[0, "affordable_contracts"] == 5
    assert retriever.metrics()["affordability_checked_count"] == 2.0
    assert retriever.metrics()["affordability_filtered_count"] == 1.0


def test_option_retriever_defers_model_greeks_until_after_candidate_filtering() -> None:
    def fake_read(symbol, *, start_date, end_date, columns):
        return pd.DataFrame(
            {
                "snapshot_date": [start_date, start_date, start_date],
                "contract_symbol": ["AAPL_C_100", "AAPL_C_130", "AAPL_C_101"],
                "expiration": [pd.Timestamp("2025-03-21")] * 3,
                "strike": [100.0, 130.0, 101.0],
                "option_type": ["call", "call", "call"],
                "bid": [1.9, 0.1, 2.0],
                "ask": [2.1, 0.2, 2.4],
                "mid": [2.0, 0.15, 2.2],
                "dte": [45, 45, 45],
                "spread_pct": [0.05, 0.5, 0.1],
            }
        )

    class _Features:
        def __init__(self, frame):
            self.df = frame

    calls = []

    def fake_features(frame, *, underlying_price=None, target_dte=None, compute_model_greeks=True):
        calls.append({"rows": len(frame), "compute_model_greeks": compute_model_greeks})
        out = frame.copy()
        out["dte_gap"] = (out["dte"] - int(target_dte or 45)).abs()
        if compute_model_greeks:
            out["delta"] = 0.5
        return _Features(out)

    prices = {
        "AAPL": pd.DataFrame(
            {"close": [100.0, 110.0]},
            index=[pd.Timestamp("2025-01-02"), pd.Timestamp("2025-01-20")],
        )
    }
    retriever = _OptionRetriever(
        OptionRetrievalConfig(max_candidates_per_trade=1, exit_lookback_days=0),
        price_frames=prices,
        read_option_chain_arctic=fake_read,
        build_option_contract_features=fake_features,
    )

    rows = retriever.retrieve(
        pd.Series(
            {
                "symbol": "AAPL",
                "side": "long",
                "entry_date": pd.Timestamp("2025-01-02"),
                "exit_date": pd.Timestamp("2025-01-20"),
            }
        )
    )

    assert calls[0] == {"rows": 3, "compute_model_greeks": False}
    assert {"rows": 1, "compute_model_greeks": True} in calls
    assert rows.loc[0, "delta"] == 0.5


def test_source_family_diagnostics_counts_selected_rows() -> None:
    frame = pd.DataFrame(
        {
            "strategy_source": ["fmp.a", "fmp.a", "fmp.b"],
            "source": ["fmp", "fmp", "fmp"],
            "family": ["a", "a", "b"],
            "source_family": ["fmp.a", "fmp.a", "fmp.b"],
            "trade_id": ["t1", "t1", "t2"],
            "symbol": ["AAPL", "AAPL", "MSFT"],
            "entry_date": ["2025-01-02", "2025-01-02", "2025-01-03"],
            "option_return": [0.1, -0.2, 0.3],
        }
    )
    selected = {"model_ranker": frame.iloc[[0, 2]].copy()}

    diagnostics = _source_family_diagnostics(frame, selected)

    by_source = diagnostics.set_index("strategy_source")
    assert by_source.loc["fmp.a", "option_rows"] == 2
    assert by_source.loc["fmp.a", "model_ranker_selected_rows"] == 1
    assert by_source.loc["fmp.b", "model_ranker_selected_rows"] == 1


def test_classifier_trade_windows_require_top_k_for_option_sizing() -> None:
    frame = pd.DataFrame(
        {
            "strategy_source": ["fmp.a"],
            "source": ["fmp"],
            "family": ["a"],
            "symbol": ["AAPL"],
            "side": ["long"],
            "entry_date": [pd.Timestamp("2025-01-02")],
            "exit_date": [pd.Timestamp("2025-01-10")],
        }
    )

    with pytest.raises(KeyError, match="top_k"):
        _normalize_trade_windows(frame)


def test_portfolio_fraction_option_log_reserves_cash_across_trading_days() -> None:
    trades = pd.DataFrame(
        {
            "trade_id": ["t1", "t2"],
            "underlying_symbol": ["AAPL", "MSFT"],
            "entry_date": [pd.Timestamp("2025-01-02"), pd.Timestamp("2025-01-03")],
            "exit_date": [pd.Timestamp("2025-01-10"), pd.Timestamp("2025-01-13")],
            "expiration": [pd.Timestamp("2025-02-21"), pd.Timestamp("2025-02-21")],
            "entry_cost": [10.0, 10.0],
            "exit_proceeds": [20.0, 20.0],
            "pct_change": [1.0, 1.0],
            "description": ["call", "call"],
            "exit_type": ["signal_exit", "signal_exit"],
            "top_k": [1, 1],
        }
    )

    log = _build_portfolio_fraction_trade_log(
        trades,
        capital=1_000.0,
        multiplier=100,
        max_positions=2,
        default_fraction=1.0,
        top_k_column="top_k",
    )

    assert len(log) == 1
    assert log.loc[0, "trade_id"] == "t1"
    assert log.loc[0, "dollar_cost"] == 1_000.0
    assert log.loc[0, "equity"] == 2_000.0


def test_portfolio_fraction_option_log_maps_weekend_exit_to_previous_business_day() -> None:
    trades = pd.DataFrame(
        {
            "trade_id": ["t1"],
            "underlying_symbol": ["AAPL"],
            "entry_date": [pd.Timestamp("2025-01-02")],
            "exit_date": [pd.Timestamp("2025-01-11")],
            "expiration": [pd.Timestamp("2025-01-11")],
            "entry_cost": [10.0],
            "exit_proceeds": [11.0],
            "pct_change": [0.1],
            "description": ["call"],
            "exit_type": ["expiration"],
            "top_k": [1],
        }
    )

    log = _build_portfolio_fraction_trade_log(
        trades,
        capital=1_000.0,
        multiplier=100,
        max_positions=1,
        default_fraction=1.0,
        top_k_column="top_k",
    )

    assert log.loc[0, "exit_date"] == pd.Timestamp("2025-01-10")


def test_option_feature_coverage_reports_greeks_and_iv_rates() -> None:
    option_panel = pd.DataFrame(
        {
            "entry_date": [pd.Timestamp("2025-01-02"), pd.Timestamp("2025-01-03")],
            "delta": [0.5, None],
            "gamma": [None, None],
            "iv": [0.2, 0.3],
            "liquidity_score": [10.0, 20.0],
        }
    )

    coverage = _option_feature_coverage(option_panel, train_panel=option_panel.iloc[:1], eval_panel=option_panel.iloc[1:])

    all_rows = coverage.loc[coverage["split"].eq("all")].set_index("feature")
    assert all_rows.loc["delta", "non_null_rate"] == pytest.approx(0.5)
    assert all_rows.loc["gamma", "non_null_rate"] == pytest.approx(0.0)
    assert all_rows.loc["iv", "non_null_rate"] == pytest.approx(1.0)


def test_standard_artifacts_round_trip_trade_windows_and_source_summary(tmp_path) -> None:
    config = OracleOptionExperimentConfig(experiment_name="unit_option_artifacts", artifact_dir=None)
    expected_default = option_experiment_artifact_dir(config)
    assert str(expected_default) == "artifacts/options/unit_option_artifacts/latest"
    frame = pd.DataFrame({"symbol": ["AAPL"], "side": ["long"], "entry_date": [pd.Timestamp("2025-01-02")], "exit_date": [pd.Timestamp("2025-01-10")], "trade_id": ["t1"]})
    option_panel = pd.DataFrame({"trade_id": ["t1"], "symbol": ["AAPL"], "option_return": [0.1]})
    source_summary = pd.DataFrame({"strategy_source": ["fmp.a"], "option_rows": [1]})

    paths = write_oracle_option_artifacts(
        config=config,
        coverage=pd.DataFrame({"symbol": ["AAPL"]}),
        oracle_trades=frame,
        option_panel=option_panel,
        train_panel=option_panel,
        eval_panel=option_panel,
        selector_summary=pd.DataFrame({"selector": ["model_ranker"], "trades": [1]}),
        optopsy_summary=pd.DataFrame({"selector": ["model_ranker"], "closed_trades": [1]}),
        symbol_summary=pd.DataFrame({"symbol": ["AAPL"], "trades": [1]}),
        source_family_summary=source_summary,
        feature_coverage=pd.DataFrame({"split": ["all"], "feature": ["delta"], "non_null_rate": [1.0]}),
        model=None,
        metrics={"option_chain_read_count": 1.0},
        analysis_markdown="analysis",
        directory=tmp_path,
    )
    loaded = load_option_experiment_artifacts(tmp_path)

    assert paths["trade_windows"].exists()
    assert paths["feature_coverage"].exists()
    assert loaded.trade_windows.loc[0, "trade_id"] == "t1"
    assert loaded.source_family_summary.loc[0, "strategy_source"] == "fmp.a"
    assert loaded.feature_coverage.loc[0, "feature"] == "delta"


def test_estimate_option_runtime_scaling_uses_conservative_budget() -> None:
    estimate = estimate_option_runtime_scaling(
        baseline_elapsed_seconds=60.0,
        baseline_trade_windows=10,
        baseline_option_rows=100,
        target_trade_windows=100,
        target_option_rows=500,
        max_seconds=3600.0,
    )

    assert estimate.estimated_seconds_by_windows == 600.0
    assert estimate.estimated_seconds_by_option_rows == 300.0
    assert estimate.estimated_seconds_conservative == 600.0
    assert estimate.runnable_within_budget


def test_weighted_basket_maps_to_optopsy_multi_leg_raw() -> None:
    frame = pd.DataFrame(
        {
            "trade_id": ["t1", "t1", "t1"],
            "symbol": ["AAPL", "AAPL", "AAPL"],
            "entry_date": ["2025-01-02", "2025-01-02", "2025-01-02"],
            "option_exit_date": ["2025-01-20", "2025-01-20", "2025-01-20"],
            "expiration": ["2025-02-21", "2025-02-21", "2025-03-21"],
            "entry_mid": [2.0, 4.0, 10.0],
            "exit_mid": [3.0, 2.0, 9.0],
            "option_type": ["call", "put", "call"],
            "strike": [100.0, 95.0, 110.0],
            "option_return": [0.5, -0.5, -0.1],
            "pred_mv_weight": [0.75, 0.25, 0.0],
        }
    )

    basket = _choose_weighted_basket_per_trade(frame, "pred_mv_weight", selector_name="model_mv_basket")
    raw = _selected_basket_to_optopsy_raw(basket)

    assert len(basket) == 2
    assert raw.loc[0, "underlying_symbol"] == "AAPL"
    assert raw.loc[0, "total_entry_cost"] == 2.5
    assert raw.loc[0, "total_exit_proceeds"] == 2.75
    assert raw.loc[0, "pct_change"] == pytest.approx(0.1)
    assert raw.loc[0, "option_type_leg1"] == "call"
