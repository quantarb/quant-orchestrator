"""Canonical annual option-cohort and DTE-group training utilities."""

from __future__ import annotations

from typing import Iterable

import polars as pl


def frozen_ten_option_members(chain: pl.DataFrame, *, first_session) -> pl.DataFrame:
    """Select five observed expiries per right; retain every strike in each.

    Selection uses only the actual first session, never a later fallback.
    Quantile selection spans available positive DTEs without mixing expiries.
    Equal contract weights are fixed at entry and must not be renormalized
    when a constituent quote subsequently disappears.
    """
    first = chain.filter(_date_expr("snapshot_date") == first_session).with_columns(
        ( _date_expr("expiration") - pl.lit(first_session)).dt.total_days().alias("dte"),
        pl.col("option_type").str.to_lowercase(),
    ).filter(pl.col("dte") > 0).unique("contract_symbol")
    selected = []
    for right in ("call", "put"):
        part = first.filter(pl.col("option_type") == right)
        expiries = sorted(part["dte"].unique().to_list())
        if len(expiries) < 5:
            raise ValueError(f"First-session {right} cohort has {len(expiries)} DTEs; five required")
        chosen = [expiries[round(i * (len(expiries) - 1) / 4)] for i in range(5)]
        selected.append(part.filter(pl.col("dte").is_in(chosen)))
    return pl.concat(selected).with_columns(
        pl.concat_str([pl.lit("OPT_"), pl.col("underlying_symbol"),
                       pl.lit(f"_{first_session.year}_"), pl.col("option_type").str.to_uppercase(),
                       pl.lit("_DTE_"), pl.col("dte")]).alias("document_symbol"),
    ).with_columns((1.0 / pl.len().over("document_symbol")).alias("weight"))


def frozen_option_paths(quotes: pl.DataFrame, members: pl.DataFrame) -> pl.DataFrame:
    """Produce executable basket paths only where every frozen member is quoted.

    Missing constituents invalidate a day's basket, rather than changing its
    composition. No forward fill and no post-expiration extension are allowed.
    """
    joined = quotes.join(members.select("contract_symbol", "document_symbol", "weight"),
                         on="contract_symbol", how="inner").with_columns(_date_expr("snapshot_date").alias("date"))
    joined = joined.filter((pl.col("bid") >= 0) & (pl.col("ask") > 0)
                           & (pl.col("ask") >= pl.col("bid"))
                           & pl.col("bid").is_finite() & pl.col("ask").is_finite()
                           & (pl.col("date") <= _date_expr("expiration")))
    expected = members.group_by("document_symbol").len().rename({"len": "expected"})
    return joined.unique(["document_symbol", "contract_symbol", "date"]).group_by("document_symbol", "date").agg(
        (pl.col("bid") * pl.col("weight")).sum().alias("low"),
        (pl.col("ask") * pl.col("weight")).sum().alias("high"),
        pl.col("volume").sum().alias("volume"), pl.len().alias("observed"),
    ).join(expected, on="document_symbol").filter(pl.col("observed") == pl.col("expected")).with_columns(
        ((pl.col("low") + pl.col("high")) / 2).alias("close"),
    ).with_columns(pl.col("close").alias("open")).rename({"document_symbol": "symbol"}).sort("symbol", "date")


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
