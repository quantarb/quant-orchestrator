"""Add delayed oracle trade actions as a temporal target family."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from quant_warehouse import Warehouse
from quant_warehouse.platforms.data_providers.fmp.target_engineering import (
    LabelBuildSpec,
    build_oracle_labels,
)


ORACLE_TARGET_FAMILY = "equity.strategy.oracle_trades"
ORACLE_CHANNELS = ("is_oracle_buy", "is_oracle_sell", "is_oracle_short", "is_oracle_cover")


def _prices_as_frame(warehouse: Warehouse, symbol: str, start: str) -> pd.DataFrame:
    prices = warehouse.read_prices(symbol, provider="fmp", start=start)
    if prices is None or prices.empty:
        return pd.DataFrame()
    frame = prices.reset_index()
    if "date" not in frame.columns:
        frame = frame.rename(columns={frame.columns[0]: "date"})
    return frame


def _oracle_rows(symbol: str, prices: pd.DataFrame, spec: LabelBuildSpec) -> pd.DataFrame:
    result = build_oracle_labels([symbol], spec=spec, price_frames={symbol: prices})
    if not result.label_rows:
        return pd.DataFrame()
    labels = pd.DataFrame(result.label_rows)
    labels["event_date"] = pd.to_datetime(labels["date"], errors="coerce").dt.normalize()
    labels["reported_date"] = pd.to_datetime(labels["exit_date"], errors="coerce").dt.normalize()
    labels = labels.loc[labels["event_date"].notna() & labels["reported_date"].notna()]
    labels = labels.loc[labels["reported_date"] >= labels["event_date"]]
    if labels.empty:
        return pd.DataFrame()
    labels["channel"] = labels["label"].map({
        "buy": "is_oracle_buy", "sell": "is_oracle_sell",
        "short": "is_oracle_short", "cover": "is_oracle_cover",
    })
    labels = labels.loc[labels["channel"].notna()]
    # Collapse all YE horizons to one binary action vector per symbol/date.
    grouped = labels.groupby(["event_date"], as_index=False).agg(
        reported_date=("reported_date", "max"),
    )
    for channel in ORACLE_CHANNELS:
        grouped[channel] = labels.loc[labels["channel"].eq(channel)].groupby("event_date").size().reindex(grouped["event_date"], fill_value=0).to_numpy() > 0
    output = pd.DataFrame({
        "symbol": symbol,
        "date": grouped["reported_date"].to_numpy(),
        "event_date": grouped["event_date"].to_numpy(),
        "reported_date": grouped["reported_date"].to_numpy(),
        "target_family": ORACLE_TARGET_FAMILY,
    })
    # Sparse target families reserve eight channels. The first four carry the
    # requested oracle actions; the remaining channels are explicit zeros.
    for index, channel in enumerate((*ORACLE_CHANNELS, "unused_0", "unused_1", "unused_2", "unused_3")):
        output["signal_value" if index == 0 else f"text_{index - 1}"] = (
            grouped[channel].astype("float32") if channel in grouped else 0.0
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, action="append", required=True)
    parser.add_argument("--start-date", default="2018-01-01")
    args = parser.parse_args()
    warehouse = Warehouse()
    spec = LabelBuildSpec(
        k_params={"YE": list(range(1, 13))},
        buy_execution="high", sell_execution="low",
        short_execution="low", cover_execution="high",
        start_date=args.start_date,
    )
    for directory in args.input_dir:
        symbols = sorted(pd.read_csv(directory / "symbols.csv")["symbol"].astype(str).str.upper().unique())
        rows: list[pd.DataFrame] = []
        for index, symbol in enumerate(symbols, start=1):
            prices = _prices_as_frame(warehouse, symbol, args.start_date)
            result = _oracle_rows(symbol, prices, spec) if not prices.empty else pd.DataFrame()
            if not result.empty:
                rows.append(result)
            if index % 50 == 0 or index == len(symbols):
                print(f"{directory}: processed {index}/{len(symbols)}", flush=True)
        oracle = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        existing_path = directory / "target_events.parquet"
        existing = pd.read_parquet(existing_path if existing_path.exists() else directory / "sparse_events.parquet")
        existing = existing.loc[existing["target_family"].ne(ORACLE_TARGET_FAMILY)]
        combined = pd.concat([existing, oracle], ignore_index=True, sort=False)
        combined.to_parquet(directory / "target_events.parquet", index=False)
        print(f"{directory}: oracle_rows={len(oracle)} target_events={len(combined)}", flush=True)


if __name__ == "__main__":
    main()
