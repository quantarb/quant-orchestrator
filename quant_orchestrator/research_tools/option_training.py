"""Canonical annual option-cohort and DTE-group training utilities."""

from __future__ import annotations

from typing import Iterable

import polars as pl


def _date_expr(name: str) -> pl.Expr:
    return pl.col(name).cast(pl.Datetime, strict=False).dt.replace_time_zone(None).dt.truncate("1d")


def _weighted_quote(quote: str, volume: str, alias: str) -> pl.Expr:
    q = pl.col(quote).cast(pl.Float64, strict=False)
    w = pl.col(volume).cast(pl.Float64, strict=False)
    valid = q.gt(0) & q.is_not_null() & w.gt(0) & w.is_not_null()
    return pl.when(valid.sum() > 0).then(
        (q.filter(valid) * w.filter(valid)).sum() / w.filter(valid).sum()
    ).otherwise(q.filter(q.gt(0)).mean()).alias(alias)


def load_first_trading_day_option_chains(
    symbols: Iterable[str], *, start_year: int, end_year: int
) -> pl.DataFrame:
    """Load only each symbol's first observed trading-session chain per year."""
    from quant_warehouse.platforms.data_providers.thetadata.options import read_thetadata_eod_option_chain

    columns = [
        "snapshot_date", "underlying_symbol", "contract_symbol", "expiration",
        "option_type", "strike", "bid", "ask", "mid", "volume", "open_interest",
    ]
    frames: list[pl.DataFrame] = []
    for symbol in sorted({str(value).strip().upper() for value in symbols if str(value).strip()}):
        for year in range(int(start_year), int(end_year) + 1):
            chain = read_thetadata_eod_option_chain(
                symbol, start_date=f"{year}-01-01", end_date=f"{year}-01-11", columns=columns,
            )
            if chain is None or chain.is_empty():
                continue
            chain = chain.with_columns(_date_expr("snapshot_date"), _date_expr("expiration"))
            first = chain.select(pl.col("snapshot_date").drop_nulls().min()).item()
            if first is None:
                continue
            chain = chain.filter(pl.col("snapshot_date") == first)
            if chain.is_empty():
                continue
            frames.append(chain.with_columns(
                pl.lit(symbol).alias("symbol"),
                pl.lit(first).alias("entry_date"),
                pl.col("option_type").cast(pl.String).str.to_lowercase().str.strip_chars(),
                pl.when(pl.col("option_type").cast(pl.String).str.to_lowercase() == "call")
                .then(pl.lit("long")).otherwise(pl.lit("short")).alias("side"),
                (pl.col("expiration") - pl.lit(first)).dt.total_days().cast(pl.Int64).alias("dte"),
            ))
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()


def group_option_contracts_by_dte(options: pl.DataFrame) -> pl.DataFrame:
    """Compress a frozen chain into one synthetic row per symbol/year/type/DTE."""
    if options.is_empty():
        return options
    options = options.with_columns(
        pl.col("symbol").cast(pl.String).str.to_uppercase().str.strip_chars(),
        _date_expr("entry_date"),
        pl.col("option_type").cast(pl.String).str.to_lowercase().str.strip_chars(),
        pl.col("volume").cast(pl.Float64, strict=False),
        pl.col("open_interest").cast(pl.Float64, strict=False),
    )
    if "dte" not in options.columns:
        options = options.with_columns(pl.lit(None, dtype=pl.Int64).alias("dte"))
    options = options.with_columns(
        pl.col("expiration").cast(pl.Datetime, strict=False).dt.replace_time_zone(None).dt.truncate("1d")
    ).with_columns(
        pl.when(pl.col("dte").is_null())
        .then((pl.col("expiration") - pl.col("entry_date")).dt.total_days())
        .otherwise(pl.col("dte")).cast(pl.Int64).alias("dte")
    ).filter(
        pl.col("symbol").is_not_null() & pl.col("entry_date").is_not_null()
        & pl.col("contract_symbol").is_not_null() & pl.col("dte").is_not_null()
    ).with_columns(pl.col("entry_date").dt.year().alias("year"))
    groups = options.group_by(["symbol", "year", "option_type", "dte"], maintain_order=True).agg(
        pl.col("snapshot_date").first(), pl.col("underlying_symbol").first(),
        pl.col("expiration").first(), pl.col("strike").first(), pl.col("bid").first(),
        pl.col("ask").first(), pl.col("mid").first(), pl.col("volume").mean(),
        pl.col("open_interest").mean(), pl.col("entry_date").first(), pl.col("side").first(),
        pl.col("contract_symbol").first().alias("contract_symbol_example"),
        pl.col("contract_symbol").n_unique().alias("dte_contract_count"),
        pl.col("contract_symbol").sort().str.join(",").alias("dte_contracts"),
        _weighted_quote("bid", "volume", "entry_bid"),
        _weighted_quote("ask", "volume", "entry_ask"),
    )
    return groups.with_columns(
        pl.concat_str([pl.lit("DTE_"), pl.col("dte").cast(pl.String)]).alias("contract_symbol")
    ).drop("contract_symbol_example")


__all__ = ["group_option_contracts_by_dte", "load_first_trading_day_option_chains"]
