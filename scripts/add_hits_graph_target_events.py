"""Add delayed HITS return/speed scores as a temporal target family."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from quant_warehouse import Warehouse
from quant_warehouse.platforms.data_providers.fmp.target_engineering import (
    HitsLabelSpec,
    build_return_and_speed_hits_labels,
)


HITS_TARGET_FAMILY = "equity.strategy.hits_graph"
HITS_CHANNELS = (
    "long_return_hub", "long_return_authority",
    "short_return_hub", "short_return_authority",
    "long_speed_hub", "long_speed_authority",
    "short_speed_hub", "short_speed_authority",
)
HITS_SOURCE_COLUMNS = (
    "long_hub", "long_authority", "short_hub", "short_authority",
    "speed_long_hub", "speed_long_authority", "speed_short_hub", "speed_short_authority",
)


def _prices_as_frame(warehouse: Warehouse, symbol: str, start: str) -> pd.DataFrame:
    prices = warehouse.read_prices(symbol, provider="fmp", start=start)
    if prices is None or prices.empty:
        return pd.DataFrame()
    frame = prices.reset_index()
    if "date" not in frame.columns:
        frame = frame.rename(columns={frame.columns[0]: "date"})
    return frame


def _delayed_hits_rows(symbol: str, prices: pd.DataFrame, spec: HitsLabelSpec) -> pd.DataFrame:
    work = prices.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce").dt.normalize()
    work = work.dropna(subset=["date", "high", "low"]).sort_values("date").drop_duplicates("date")
    labels = build_return_and_speed_hits_labels({symbol: work}, spec=spec)
    if labels.empty:
        return pd.DataFrame()
    # HITS is future-derived. Make a label visible only after its graph look-
    # ahead has elapsed, using the actual future trading date rather than a
    # calendar-day approximation.
    availability: dict[pd.Timestamp, pd.Timestamp] = {}
    for _, year_frame in work.groupby(work["date"].dt.year, sort=True):
        dates = year_frame["date"].tolist()
        for index, date in enumerate(dates[:-1]):
            availability[date] = dates[min(index + spec.max_hold, len(dates) - 1)]
    labels["date"] = pd.to_datetime(labels["date"]).dt.normalize()
    labels["reported_date"] = labels["date"].map(availability)
    labels = labels.loc[labels["reported_date"].notna() & (labels["reported_date"] > labels["date"])].copy()
    if labels.empty:
        return pd.DataFrame()
    output = pd.DataFrame({
        "symbol": symbol,
        "date": labels["reported_date"].to_numpy(),
        "event_date": labels["date"].to_numpy(),
        "reported_date": labels["reported_date"].to_numpy(),
        "target_family": HITS_TARGET_FAMILY,
    })
    # The sparse representation has eight channels per target family. Store
    # the eight HITS scores in those channels without text re-encoding.
    for index, source_column in enumerate(HITS_SOURCE_COLUMNS):
        output["signal_value" if index == 0 else f"text_{index - 1}"] = labels[source_column].to_numpy("float32")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, action="append", required=True)
    parser.add_argument("--start-date", default="2018-01-01")
    parser.add_argument("--max-hold", type=int, default=120)
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()
    warehouse = Warehouse()
    spec = HitsLabelSpec(max_hold=args.max_hold, iterations=args.iterations)
    for directory in args.input_dir:
        symbols = sorted(pd.read_csv(directory / "symbols.csv")["symbol"].astype(str).str.upper().unique())
        rows: list[pd.DataFrame] = []
        for index, symbol in enumerate(symbols, start=1):
            prices = _prices_as_frame(warehouse, symbol, args.start_date)
            result = _delayed_hits_rows(symbol, prices, spec) if not prices.empty else pd.DataFrame()
            if not result.empty:
                rows.append(result)
            if index % 50 == 0 or index == len(symbols):
                print(f"{directory}: processed {index}/{len(symbols)}", flush=True)
        hits = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        existing_path = directory / "target_events.parquet"
        existing = pd.read_parquet(existing_path if existing_path.exists() else directory / "sparse_events.parquet")
        existing = existing.loc[existing["target_family"].ne(HITS_TARGET_FAMILY)]
        combined = pd.concat([existing, hits], ignore_index=True, sort=False)
        combined.to_parquet(directory / "target_events.parquet", index=False)
        print(f"{directory}: hits_rows={len(hits)} target_events={len(combined)}", flush=True)


if __name__ == "__main__":
    main()
