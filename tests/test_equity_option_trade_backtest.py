from datetime import datetime, timedelta

import polars as pl
import pytest

from quant_orchestrator.platforms.backtesting_frameworks.equity_option_trade_backtest import (
    choose_ranked_contract,
    model_exit,
)


def test_option_rank_uses_own_long_returns_for_puts_too():
    candidates = pl.DataFrame(
        {"document_symbol": ["put_a", "put_b"], "option_type": ["put", "put"]}
    )
    scores = [
        dict(symbol="put_a", hits_long_return_hub=1.0, hits_long_return_authority=2.0),
        dict(symbol="put_b", hits_long_return_hub=3.0, hits_long_return_authority=2.0),
    ]
    winner, score = choose_ranked_contract(candidates, scores)
    assert winner["document_symbol"] == "put_b"
    assert score == 2.5
    with pytest.raises(ValueError):
        choose_ranked_contract(candidates, scores + scores[:1])


def test_option_exit_uses_own_classifier_and_next_quote():
    start = datetime(2024, 1, 2)
    days = [start + timedelta(days=n) for n in (0, 1, 3, 4)]
    path = pl.DataFrame({"date": days, "low": [2.0, 3.0, 4.0, 5.0]})
    seen = []

    def score_day(day):
        seen.append(day)
        return [
            dict(
                oracle_is_buy=0.8,
                oracle_is_short=0.1,
                oracle_is_sell=0.6 if day == days[1] else 0.1,
            )
        ]

    quote, reason, count = model_exit(path, days[-1], score_day)
    assert quote["date"] == days[2]  # Missing date never causes a backward fill.
    assert reason == "option_oracle_exit"
    assert count == 2
    assert seen == days[:2]  # No scoring after exit.


def test_option_can_hold_to_expiration_with_no_equity_exit_dependency():
    days = [datetime(2024, 1, n) for n in (2, 3, 4)]
    path = pl.DataFrame({"date": days, "low": [2.0, 1.0, 0.0]})
    quote, reason, count = model_exit(
        path, days[-1], lambda _: [dict(oracle_is_buy=0.9, oracle_is_short=0.1, oracle_is_sell=0.1)]
    )
    assert quote["low"] == 0
    assert reason == "expiration"
    assert count == 2


def test_trade_replay_holds_past_equity_exit_and_buys_one_model_selected_identity(tmp_path):
    import pandas as pd
    from quant_orchestrator.platforms.backtesting_frameworks.equity_option_trade_backtest import (
        run_equity_option_trade_backtest,
    )

    days = [datetime(2024, 1, n) for n in (2, 3, 4, 5, 8)]
    members = pl.DataFrame(
        [
            dict(
                document_symbol=s,
                contract_symbol=s,
                option_type="call",
                expiration=days[-1],
                settlement=days[-1],
                strike=100.0,
            )
            for s in ("a", "b")
        ]
    )
    paths = pl.DataFrame(
        [dict(symbol=s, date=d, low=2.0, high=3.0) for s in ("a", "b") for d in days]
    )

    class Stream:
        def prepare_option_year(self, year):
            pass

        def cohorts(self, symbol, year):
            return members, paths

    requests = []

    def predict(symbol, year, date, candidates, paths):
        requests.append((date, candidates["document_symbol"].to_list()))
        return [
            dict(
                symbol=s,
                hits_long_return_hub=2.0 if s == "b" else 0.0,
                hits_long_return_authority=1.0,
                oracle_is_buy=0.9,
                oracle_is_short=0.1,
                oracle_is_sell=0.9 if date == days[2] else 0.1,
            )
            for s in candidates["document_symbol"]
        ]

    windows = pd.DataFrame(
        [
            dict(
                trade_id="t",
                entry_price=100.0,
                exit_price=110.0,
                equity_entry_notional=500.0,
                symbol="A",
                side="long",
                entry_date=days[1],
                exit_date=days[2],
            )
        ]
    )
    # The first trade consumes the budget. Do not infer an unfundable second entry.
    windows = pd.concat([windows, windows.assign(trade_id="t2", symbol="B")], ignore_index=True)
    reports = run_equity_option_trade_backtest(
        windows, Stream(), predict, tmp_path / "bt", year=2024, dates=days, capacity=1
    )
    selected = pd.read_parquet(tmp_path / "bt/call_selected_trades.parquet")
    assert selected.contract_symbol.tolist() == ["b"]
    assert selected.option_exit_date.iloc[0] == days[3]  # After equity exit.
    assert selected.exit_reason.iloc[0] == "option_oracle_exit"
    assert requests == [(days[0], ["a", "b"]), (days[1], ["b"]), (days[2], ["b"])]
    ledger = pd.read_parquet(tmp_path / "bt/call_trade_ledger.parquet")
    assert (ledger.option_units % 100 == 0).all()
    assert ledger.entry_fee.iloc[0] > 0
    assert pd.read_parquet(tmp_path / "bt/call_equity.parquet").cash.min() >= 0
    assert reports[0]["skipped_entry_trades"] == 1
    assert reports[0]["unfunded_trades"] == 0
    assert reports[1]["entries"] == 0
