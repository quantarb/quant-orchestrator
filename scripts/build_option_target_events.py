"""Generate option-native HITS and Oracle sparse targets from bid/ask baskets."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from quant_warehouse.platforms.data_providers.fmp.target_engineering import (
    HitsLabelSpec,
    LabelBuildSpec,
    build_oracle_labels,
    build_return_and_speed_hits_labels,
)

from scripts.backtest_dte_event_driven import _build_daily_quotes


def _available_date(dates: pd.Series, date: pd.Timestamp, hold: int) -> pd.Timestamp:
    values = pd.DatetimeIndex(sorted(pd.to_datetime(dates).dropna().unique()))
    later = values[values >= date]
    if len(later) == 0:
        return date
    index = min(hold, len(later) - 1)
    return pd.Timestamp(later[index]).normalize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hits-max-hold", type=int, default=120)
    parser.add_argument("--oracle-min-profit", type=float, default=0.01)
    args = parser.parse_args()
    groups = pd.read_parquet(args.groups)
    groups["entry_date"] = pd.to_datetime(groups["entry_date"], errors="coerce").dt.normalize()
    groups["document_symbol"] = groups["document_symbol"].astype(str).str.upper()
    groups["underlying_symbol"] = groups["underlying_symbol"].astype(str).str.upper()
    quotes = _build_daily_quotes(groups, set(groups.underlying_symbol))
    if quotes.empty:
        raise RuntimeError("no raw bid/ask quotes available for option target generation")

    rows: list[pd.DataFrame] = []
    for symbol, frame in quotes.groupby("symbol", sort=False):
        frame = frame.sort_values("date").drop_duplicates("date")
        # HITS consumes high/low. Mapping high=ask and low=bid makes every
        # long edge executable as ask-to-bid and every short edge as
        # bid-to-ask; no midpoint is introduced.
        prices = pd.DataFrame({
            "date": frame["date"], "high": frame["ask"], "low": frame["bid"],
            "adj_high": frame["ask"], "adj_low": frame["bid"],
            "close": frame["bid"], "adj_close": frame["bid"],
        }).dropna(subset=["high", "low"])
        if len(prices) < 2:
            continue
        hits = build_return_and_speed_hits_labels({symbol: prices}, spec=HitsLabelSpec(max_hold=args.hits_max_hold, iterations=50))
        if not hits.empty:
            hits["event_date"] = pd.to_datetime(hits["date"]).dt.normalize()
            hits["date"] = hits["event_date"].map(lambda d: _available_date(prices["date"], d, args.hits_max_hold))
            hits["reported_date"] = hits["date"]
            hits["target_family"] = "equity.strategy.hits_graph"
            channels = ["long_hub", "long_authority", "short_hub", "short_authority", "speed_long_hub", "speed_long_authority", "speed_short_hub", "speed_short_authority"]
            for index, channel in enumerate(channels):
                hits["signal_value" if index == 0 else f"text_{index - 1}"] = hits[channel]
            rows.append(hits[["symbol", "date", "event_date", "reported_date", "target_family", "signal_value", *[f"text_{i}" for i in range(7)]]])
        oracle = build_oracle_labels([symbol], spec=LabelBuildSpec(
            k_params={"YE": [1]}, min_profit_pct=args.oracle_min_profit,
            buy_execution="adj_high", sell_execution="adj_low",
            short_execution="adj_low", cover_execution="adj_high",
        ), price_frames={symbol: prices})
        if oracle.label_rows:
            labels = pd.DataFrame(oracle.label_rows)
            labels["event_date"] = pd.to_datetime(labels["date"]).dt.normalize()
            labels["reported_date"] = pd.to_datetime(labels["exit_date"]).dt.normalize()
            labels["date"] = labels["reported_date"]
            channels = {"buy": "is_oracle_buy", "sell": "is_oracle_sell", "short": "is_oracle_short", "cover": "is_oracle_cover"}
            labels["channel"] = labels["label"].map(channels)
            labels = labels.dropna(subset=["channel"])
            grouped = labels.groupby("event_date", as_index=False).agg(reported_date=("reported_date", "max"))
            for index, channel in enumerate(channels.values()):
                values = labels.loc[labels.channel.eq(channel)].groupby("event_date").size()
                grouped[channel] = grouped.event_date.map(values).fillna(0).gt(0).astype("float32")
            grouped["symbol"] = symbol
            grouped["date"] = grouped["reported_date"]
            grouped["target_family"] = "equity.strategy.oracle_trades"
            for index, channel in enumerate(channels.values()):
                grouped["signal_value" if index == 0 else f"text_{index - 1}"] = grouped[channel]
            for index in range(4, 8):
                grouped[f"text_{index - 1}"] = 0.0
            rows.append(grouped[["symbol", "date", "event_date", "reported_date", "target_family", "signal_value", *[f"text_{i}" for i in range(7)]]])
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)
    print(f"rows={len(out)} symbols={out.symbol.nunique() if not out.empty else 0}")


if __name__ == "__main__":
    main()
