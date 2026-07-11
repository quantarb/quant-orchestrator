from __future__ import annotations

import pandas as pd
import pytest

from quant_orchestrator.research_tools.option_family_ranker import (
    OptionFamilyRankerConfig,
    _filter_entry_date_tradable_options,
    _filter_oracle_entry_options,
    _join_family_features,
    _load_option_panel,
    _training_and_scoring_frames,
)


def test_option_panel_load_projects_filters_and_caps_candidates(tmp_path) -> None:
    rows = []
    for symbol in ("AAPL", "MSFT"):
        for index in range(5):
            rows.append(
                {
                    "trade_id": f"{symbol}-trade",
                    "symbol": symbol,
                    "entry_date": "2026-01-02",
                    "expiration": "2026-03-20",
                    "snapshot_date": "2026-01-02",
                    "option_return": float(index),
                    "fixed_near_atm_score": float(index),
                    "side": "buy",
                    "option_type": "call",
                    "unused_large_column": "x" * 100,
                }
            )
    path = tmp_path / "options.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)

    loaded = _load_option_panel(
        path,
        max_trades=0,
        symbols=("AAPL",),
        max_candidates_per_trade=2,
    )

    assert set(loaded["symbol"]) == {"AAPL"}
    assert len(loaded) == 2
    assert "unused_large_column" not in loaded.columns
    assert loaded["option_return"].tolist() == [4.0, 3.0]


def test_filter_oracle_entry_options_keeps_buy_calls_for_longs_and_buy_puts_for_shorts() -> None:
    frame = pd.DataFrame(
        {
            "trade_id": ["t1", "t1", "t1", "t2", "t2", "t2"],
            "symbol": ["AAPL", "AAPL", "AAPL", "MSFT", "MSFT", "MSFT"],
            "equity_signal_side": ["long", "long", "long", "short", "short", "short"],
            "option_type": ["call", "put", "call", "put", "call", "put"],
            "option_action": ["buy_call", "sell_put", "buy_call", "buy_put", "sell_call", "buy_put"],
            "option_return": [0.10, 0.90, 0.30, -0.20, 0.80, 0.40],
            "rank_y": [0.1, 1.0, 0.2, 0.1, 1.0, 0.2],
        }
    )

    out = _filter_oracle_entry_options(frame, target_col="rank_y")

    assert list(out["option_action"]) == ["buy_call", "buy_call", "buy_put", "buy_put"]
    assert set(out.loc[out["equity_signal_side"].eq("long"), "option_type"]) == {"call"}
    assert set(out.loc[out["equity_signal_side"].eq("short"), "option_type"]) == {"put"}
    assert out.set_index(["trade_id", "option_return"]).loc[("t1", 0.10), "rank_y"] == pytest.approx(0.5)
    assert out.set_index(["trade_id", "option_return"]).loc[("t1", 0.30), "rank_y"] == pytest.approx(1.0)
    assert out.set_index(["trade_id", "option_return"]).loc[("t2", -0.20), "rank_y"] == pytest.approx(0.5)
    assert out.set_index(["trade_id", "option_return"]).loc[("t2", 0.40), "rank_y"] == pytest.approx(1.0)


def test_filter_oracle_entry_options_uses_side_fallback_without_option_action() -> None:
    frame = pd.DataFrame(
        {
            "trade_id": ["t1", "t1", "t2", "t2"],
            "symbol": ["AAPL", "AAPL", "MSFT", "MSFT"],
            "side": ["buy", "buy", "sell", "sell"],
            "option_type": ["call", "put", "call", "put"],
            "option_return": [0.1, 0.2, 0.3, 0.4],
        }
    )

    out = _filter_oracle_entry_options(frame, target_col="rank_y")

    assert list(out["option_type"]) == ["call", "put"]
    assert list(out["side"]) == ["buy", "sell"]


def test_filter_entry_date_tradable_options_drops_expired_contracts() -> None:
    frame = pd.DataFrame(
        {
            "trade_id": ["t1", "t1"],
            "symbol": ["AAPL", "AAPL"],
            "entry_date": ["2026-07-06", "2026-07-06"],
            "expiration": ["2026-07-03", "2026-08-21"],
            "option_return": [0.9, 0.1],
        }
    )

    out = _filter_entry_date_tradable_options(frame)

    assert list(out["expiration"]) == ["2026-08-21"]


def test_filter_entry_date_tradable_options_requires_quote_date_to_match_entry_date_when_present() -> None:
    frame = pd.DataFrame(
        {
            "trade_id": ["t1", "t1"],
            "symbol": ["AAPL", "AAPL"],
            "entry_date": ["2026-07-06", "2026-07-06"],
            "quote_date": ["2026-07-03", "2026-07-06"],
            "expiration": ["2026-08-21", "2026-08-21"],
            "option_return": [0.9, 0.1],
        }
    )

    out = _filter_entry_date_tradable_options(frame)

    assert list(out["quote_date"]) == ["2026-07-06"]


def test_option_family_ranker_defaults_pairwise_off() -> None:
    assert OptionFamilyRankerConfig().disable_pairwise_ranker is True


def test_option_family_ranker_defaults_to_training_on_all_data() -> None:
    assert OptionFamilyRankerConfig().train_on_all_data is True


def test_join_family_features_uses_same_day_asof_by_default() -> None:
    option_panel = pd.DataFrame(
        {
            "trade_id": ["t1"],
            "symbol": ["AAPL"],
            "entry_date": ["2026-01-03"],
        }
    )
    feature_panel = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL"],
            "date": ["2026-01-02", "2026-01-03"],
            "feature_a": [0.2, 0.9],
        }
    )

    joined = _join_family_features(option_panel, feature_panel, ["feature_a"])

    assert joined.loc[0, "feature_date"] == pd.Timestamp("2026-01-03")
    assert joined.loc[0, "feature_a"] == 0.9


def test_training_and_scoring_frames_use_all_rows_without_reserved_eval() -> None:
    frame = pd.DataFrame(
        {
            "trade_id": ["t1", "t2", "t3"],
            "entry_date": ["2020-01-01", "2021-01-01", "2022-01-01"],
            "option_return": [0.1, 0.2, 0.3],
        }
    )

    train, score, reserved_eval_rows, mode = _training_and_scoring_frames(
        frame,
        train_end="2020-12-31",
        eval_start="2021-01-01",
        train_on_all_data=True,
    )

    assert list(train["trade_id"]) == ["t1", "t2", "t3"]
    assert list(score["trade_id"]) == ["t1", "t2", "t3"]
    assert reserved_eval_rows == 0
    assert mode == "train_in_sample_no_reserved_eval"
