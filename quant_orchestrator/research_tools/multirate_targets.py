"""Materialize the warehouse's shared Oracle/HITS tasks for any priced instrument."""

from datetime import datetime

import polars as pl


HITS_CHANNELS = (
    "long_hub",
    "long_authority",
    "short_hub",
    "short_authority",
    "speed_long_hub",
    "speed_long_authority",
    "speed_short_hub",
    "speed_short_authority",
)
VALUE_COLUMNS = ("signal_value", *(f"text_{i}" for i in range(7)))


def materialize_instrument_targets(symbol: str, prices: pl.DataFrame) -> pl.DataFrame:
    """Keep event-only labels and mark whole-year graph labels available at year end.

    Prices for options must be their own basket paths, with high=ask/low=bid.
    Task-family names are storage identifiers shared by all asset classes.
    """
    from quant_warehouse.platforms.data_providers.fmp.target_engineering import (
        HitsLabelSpec,
        LabelBuildSpec,
        build_oracle_labels,
        build_return_and_speed_hits_labels,
    )

    symbol = symbol.strip().upper()
    prices = prices.with_columns(pl.lit(symbol).alias("symbol"))
    rows = []
    for yearly in (
        prices.sort("date")
        .with_columns(pl.col("date").dt.year().alias("_year"))
        .partition_by("_year")
    ):
        year = int(yearly["_year"][0])
        yearly = yearly.drop("_year")
        if yearly.height < 2:
            continue
        available = datetime(year, 12, 31)
        hits = build_return_and_speed_hits_labels({symbol: yearly}, spec=HitsLabelSpec())
        for row in hits.iter_rows(named=True):
            values = {
                dest: float(row[source]) if row.get(f"{source}_tail") else None
                for source, dest in zip(HITS_CHANNELS, VALUE_COLUMNS)
            }
            if any(value is not None for value in values.values()):
                rows.append(
                    dict(
                        symbol=symbol,
                        date=available,
                        event_date=row["date"],
                        target_family="equity.strategy.hits_graph",
                        **values,
                    )
                )
        oracle = build_oracle_labels(
            [symbol],
            price_frames={symbol: yearly},
            spec=LabelBuildSpec(
                k_params={"YE": [1]},
                min_profit_pct=0.01,
                buy_execution="high",
                sell_execution="low",
                short_execution="low",
                cover_execution="high",
            ),
        )
        events = {}
        for row in oracle.label_rows:
            events.setdefault(row["date"], set()).add(row["label"])
        for date, labels in events.items():
            values = dict.fromkeys(VALUE_COLUMNS, None)
            values.update(
                {
                    dest: float(label in labels)
                    for dest, label in zip(VALUE_COLUMNS, ("buy", "sell", "short", "cover"))
                }
            )
            rows.append(
                dict(
                    symbol=symbol,
                    date=available,
                    event_date=datetime.fromisoformat(str(date)),
                    target_family="equity.strategy.oracle_trades",
                    **values,
                )
            )
    schema = {
        "symbol": pl.String,
        "date": pl.Datetime("ns"),
        "event_date": pl.Datetime("ns"),
        "target_family": pl.String,
        **dict.fromkeys(VALUE_COLUMNS, pl.Float32),
    }
    return pl.DataFrame(rows, schema=schema) if rows else pl.DataFrame(schema=schema)
