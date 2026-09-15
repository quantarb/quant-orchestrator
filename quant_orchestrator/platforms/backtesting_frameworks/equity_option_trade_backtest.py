"""Equity trades choose one model-ranked option from the annual filtered sample."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from .optimal_trader.artifact_replay import replay_option_portfolio_from_selected_paths

SELECTION_POLICY = dict(
    universe="annual hindsight-filtered training-style sample",
    max_candidates_per_side=5,
    contracts_per_trade=1,
    hindsight_selection=True,
    ranking="mean predicted option long-return HITS hub and authority",
    entry="ask",
    exit="bid or intrinsic settlement",
    option_scoring="entry candidates, then held contract daily",
    exit_signal="own option Oracle buy <= short or sell >= 0.5; execute next observed session",
)


def eligible_trade_candidates(members, paths, entry, side, budget):
    """At most five surviving, unexpired contracts with an executable entry quote."""
    right = "call" if side == "long" else "put"
    terms = members.filter((pl.col("option_type") == right) & (pl.col("settlement") > entry))
    quotes = paths.filter(
        (pl.col("date") == entry)
        & pl.col("low").is_finite()
        & pl.col("high").is_finite()
        & (pl.col("low") > 0)
        & (pl.col("high") >= pl.col("low"))
        & (pl.col("high") * 100 * (1 + 5.5 / 10000) <= budget)
    )
    return terms.join(
        quotes.select(
            pl.col("symbol").alias("document_symbol"), pl.col("high").alias("trade_entry_ask")
        ),
        on="document_symbol",
        how="inner",
    ).sort("document_symbol")


def choose_ranked_contract(candidates, scores):
    values = {
        r["symbol"]: (r["hits_long_return_hub"] + r["hits_long_return_authority"]) / 2
        for r in scores
    }
    if (
        len(scores) != len(values)
        or set(values) != set(candidates["document_symbol"])
        or any(not np.isfinite(v) for v in values.values())
    ):
        raise ValueError(
            "Every candidate must have exactly one finite instrument-specific rank score"
        )
    winner = min(values, key=lambda s: (-values[s], s))
    return candidates.filter(pl.col("document_symbol") == winner).row(0, named=True), values[winner]


def model_exit(held, settlement, score_day):
    """Own-option classifier exits on the following observed quote; no equity exit."""
    rows = held.sort("date").to_dicts()
    if len(rows) < 2:
        raise ValueError("Entered option needs a subsequent observed exit or settlement")
    pending = False
    count = 0
    for index, quote in enumerate(rows):
        if quote["date"] >= settlement:
            return quote, "expiration", count
        if index and pending:
            return quote, "option_oracle_exit", count
        if index == len(rows) - 1:
            return quote, "evaluation_end", count
        scores = score_day(quote["date"])
        if len(scores) != 1:
            raise ValueError("Held option must produce exactly one daily score")
        signal = scores[0]
        values = [signal[key] for key in ("oracle_is_buy", "oracle_is_short", "oracle_is_sell")]
        if not all(np.isfinite(values)):
            raise ValueError("Held option has nonfinite Oracle predictions")
        count += 1
        pending = values[0] <= values[1] or values[2] >= 0.5
    raise AssertionError("No option exit")


def run_equity_option_trade_backtest(
    trade_windows,
    stream,
    rank_candidates,
    output,
    *,
    year,
    dates,
    initial_cash=100000.0,
    capacity=20,
):
    """Only equity trade triggers request option predictions; pool fixed per year."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    cohorts = {}
    reports = []
    for side, right in [("long", "call"), ("short", "put")]:
        windows = trade_windows.loc[trade_windows.side.eq(side)].sort_values(
            ["entry_date", "symbol"]
        )
        selected_rows = []
        path_rows = []
        statuses = []
        busy_until = {}
        predictions = 0
        cash_events = []
        for trade_number, trade in enumerate(windows.to_dict("records"), 1):
            if trade_number == 1 or trade_number % 25 == 0:
                print(
                    f"[option-trade-replay] year={year} side={side} trade={trade_number}/{len(windows)} option_predictions={predictions}",
                    flush=True,
                )
            symbol = trade["symbol"]
            entry = pd.Timestamp(trade["entry_date"]).to_pydatetime()
            requested_exit = pd.Timestamp(trade["exit_date"]).to_pydatetime()
            status = dict(trade_id=trade["trade_id"], symbol=symbol, side=side, entry_date=entry)
            if symbol in busy_until and entry <= busy_until[symbol]:
                statuses.append(dict(**status, status="previous_option_still_open"))
                continue
            if symbol not in cohorts:
                cohorts[symbol] = stream.cohorts(symbol, year)
            cohort = cohorts[symbol]
            if cohort is None:
                statuses.append(dict(**status, status="no_annual_filter_survivors"))
                continue
            members, paths = cohort
            available_cash = initial_cash + sum(
                amount for date, amount in cash_events if date <= entry
            )
            budget = min(initial_cash / capacity, max(0.0, available_cash))
            if budget <= 0:
                statuses.append(dict(**status, status="insufficient_cash"))
                continue
            candidates = eligible_trade_candidates(members, paths, entry, side, budget)
            if candidates.is_empty():
                statuses.append(dict(**status, status="no_executable_surviving_contract"))
                continue
            # Rank with the prior observed session, then buy at the entry ask.
            prior = paths.filter(
                (pl.col("date") < entry)
                & pl.col("symbol").is_in(candidates["document_symbol"].to_list())
            )
            signal_dates = prior.group_by("symbol").agg(pl.col("date").max())
            score_rows = []
            for signal_date in sorted(signal_dates["date"].unique().to_list()):
                identities = signal_dates.filter(pl.col("date") == signal_date)["symbol"].to_list()
                batch = candidates.filter(pl.col("document_symbol").is_in(identities))
                score_rows.extend(rank_candidates(symbol, year, signal_date, batch, paths))
            candidates = candidates.filter(
                pl.col("document_symbol").is_in([r["symbol"] for r in score_rows])
            )
            if candidates.is_empty():
                statuses.append(dict(**status, status="no_prior_option_signal"))
                continue
            predictions += len(score_rows)
            row, rank = choose_ranked_contract(candidates, score_rows)
            path = paths.filter(pl.col("symbol") == row["document_symbol"]).sort("date")
            held = path.filter(
                (pl.col("date") >= entry)
                & (
                    pl.col("date")
                    <= min(row["settlement"], pd.Timestamp(dates[-1]).to_pydatetime())
                )
            )
            one = candidates.filter(pl.col("document_symbol") == row["document_symbol"])
            exit_row, exit_reason, held_predictions = model_exit(
                held,
                row["settlement"],
                lambda date: rank_candidates(symbol, year, date, one, paths),
            )
            predictions += held_predictions
            units = np.floor(budget / (row["trade_entry_ask"] * (1 + 5.5 / 10000)) / 100) * 100
            cash_events.extend(
                [
                    (entry, -units * row["trade_entry_ask"] * (1 + 5.5 / 10000)),
                    (exit_row["date"], units * exit_row["low"] * (1 - 5.5 / 10000)),
                ]
            )
            busy_until[symbol] = exit_row["date"]
            expired = exit_reason == "expiration"
            selected_rows.append(
                dict(
                    trade,
                    equity_exit_date=requested_exit,
                    exit_date=exit_row["date"],
                    contract_symbol=row["contract_symbol"],
                    option_type=right,
                    expiration=row["expiration"],
                    strike=row["strike"],
                    option_exit_date=exit_row["date"],
                    entry_price=row["trade_entry_ask"],
                    exit_price=exit_row["low"],
                    equity_entry_notional=budget,
                    expired_before_equity_exit=expired and exit_row["date"] <= requested_exit,
                    exit_reason=exit_reason,
                    option_rank_score=rank,
                    candidates_scored=len(score_rows),
                    held_predictions=held_predictions,
                )
            )
            path_rows.append(
                path.filter(pl.col("date").is_between(entry, exit_row["date"]))
                .select(
                    pl.lit(trade["trade_id"]).alias("trade_id"),
                    pl.col("date").alias("snapshot_date"),
                    pl.col("low").alias("mark_price"),
                )
                .to_pandas()
            )
            statuses.append(
                dict(
                    **status,
                    status="expired_worthless" if expired and exit_row["low"] == 0 else "priced",
                    contract_symbol=row["contract_symbol"],
                )
            )
        selected = pd.DataFrame(selected_rows)
        paths = pd.concat(path_rows, ignore_index=True) if path_rows else pd.DataFrame()
        replay = replay_option_portfolio_from_selected_paths(
            selected,
            paths,
            date_index=dates,
            initial_balance=initial_cash,
            whole_contracts=True,
            allow_borrowing=False,
            fee_bps=5.5,
        )
        selected.to_parquet(output / f"{right}_selected_trades.parquet", index=False)
        paths.to_parquet(output / f"{right}_selected_paths.parquet", index=False)
        pd.DataFrame(statuses).to_parquet(output / f"{right}_trade_status.parquet", index=False)
        replay.trade_ledger.to_parquet(output / f"{right}_trade_ledger.parquet", index=False)
        pd.DataFrame(
            dict(equity=replay.equity, net_return=replay.returns, cash=replay.cash)
        ).to_parquet(output / f"{right}_equity.parquet")
        peak = replay.equity.cummax().clip(lower=initial_cash)
        report = dict(
            side="long_calls" if side == "long" else "long_puts",
            capital_return=float(replay.equity.iloc[-1] / initial_cash - 1),
            final_equity=float(replay.equity.iloc[-1]),
            initial_cash=initial_cash,
            sharpe=float(replay.returns.mean() / replay.returns.std() * np.sqrt(252))
            if replay.returns.std() > 0
            else 0.0,
            max_drawdown=float((replay.equity / peak - 1).min()),
            entries=len(replay.trade_ledger),
            exits=len(replay.trade_ledger),
            model_exits=sum(r["exit_reason"] == "option_oracle_exit" for r in selected_rows),
            expiration_exits=sum(r["exit_reason"] == "expiration" for r in selected_rows),
            equity_trade_windows=len(windows),
            priced_trades=len(selected),
            skipped_entry_trades=len(windows) - len(selected),
            unfunded_trades=len(selected) - len(replay.trade_ledger),
            hindsight_selection=True,
            option_predictions=predictions,
            contracts_per_trade=1,
            selection_policy=SELECTION_POLICY,
        )
        reports.append(report)
    (output / "results.json").write_text(json.dumps(reports, indent=2))
    return reports
