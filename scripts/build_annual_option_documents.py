"""Build frozen annual option documents with a Polars-only data path."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import polars as pl


def _date_expr(name: str) -> pl.Expr:
    return pl.col(name).cast(pl.Datetime, strict=False).dt.replace_time_zone(None).dt.truncate("1d")


def _contract_id(underlying: str, year: int, option_type: str, contract: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9]+", "_", str(contract)).strip("_")
    return f"OPT_{underlying}_{year}_{option_type[:1].upper()}_{clean}"


def _weighted_quote(quote: str, volume: str, alias: str) -> pl.Expr:
    q = pl.col(quote).cast(pl.Float64, strict=False)
    w = pl.col(volume).cast(pl.Float64, strict=False)
    valid = q.gt(0) & q.is_not_null() & w.gt(0) & w.is_not_null()
    return pl.when(valid.sum() > 0).then(
        (q.filter(valid) * w.filter(valid)).sum() / w.filter(valid).sum()
    ).otherwise(q.filter(q.gt(0)).mean()).alias(alias)


def _load_raw_first_day(symbols: set[str], *, start_year: int, end_year: int) -> pl.DataFrame:
    """Read first-session chains without converting warehouse frames."""
    from quant_warehouse.platforms.data_providers.thetadata.options import read_thetadata_eod_option_chain

    frames: list[pl.DataFrame] = []
    columns = [
        "snapshot_date", "underlying_symbol", "contract_symbol", "expiration",
        "option_type", "strike", "bid", "ask", "mid", "volume", "open_interest",
    ]
    for symbol in sorted(symbols):
        for year in range(start_year, end_year + 1):
            start = f"{year}-01-01"
            end = f"{year}-01-11"
            chain = read_thetadata_eod_option_chain(
                symbol, start_date=start, end_date=end, columns=columns,
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
            frames.append(
                chain.with_columns(
                    pl.lit(symbol).alias("symbol"),
                    pl.lit(first).alias("entry_date"),
                    pl.col("option_type").cast(pl.String).str.to_lowercase().str.strip_chars(),
                    pl.when(pl.col("option_type").cast(pl.String).str.to_lowercase() == "call")
                    .then(pl.lit("long")).otherwise(pl.lit("short")).alias("side"),
                    (pl.col("expiration") - pl.lit(first)).dt.total_days().cast(pl.Int64).alias("dte"),
                )
            )
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()


def _select(options: pl.DataFrame, *, max_contracts: int, group_by_dte: bool) -> pl.DataFrame:
    if options.is_empty():
        return options
    options = options.with_columns(
        pl.col("symbol").cast(pl.String).str.to_uppercase().str.strip_chars(),
        pl.col("entry_date").cast(pl.Datetime, strict=False).dt.replace_time_zone(None).dt.truncate("1d"),
        pl.col("option_type").cast(pl.String).str.to_lowercase().str.strip_chars(),
        pl.col("volume").cast(pl.Float64, strict=False),
        pl.col("open_interest").cast(pl.Float64, strict=False),
    )
    if "dte" not in options.columns:
        options = options.with_columns(pl.lit(None, dtype=pl.Int64).alias("dte"))
    if "expiration" in options.columns:
        options = options.with_columns(
            pl.col("expiration").cast(pl.Datetime, strict=False).dt.replace_time_zone(None).dt.truncate("1d")
        ).with_columns(
            pl.when(pl.col("dte").is_null())
            .then((pl.col("expiration") - pl.col("entry_date")).dt.total_days())
            .otherwise(pl.col("dte")).cast(pl.Int64).alias("dte")
        )
    options = options.filter(
        pl.col("symbol").is_not_null() & pl.col("entry_date").is_not_null()
        & pl.col("contract_symbol").is_not_null() & pl.col("dte").is_not_null()
    ).with_columns(pl.col("entry_date").dt.year().alias("year"))
    if not group_by_dte:
        return (
            options.sort(["volume", "open_interest", "contract_symbol"], descending=[True, True, False], nulls_last=True)
            .group_by(["symbol", "year", "option_type"], maintain_order=True)
            .head(max_contracts)
        )
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


def _expand_documents(base: pl.DataFrame, options: pl.DataFrame, name: str, base_symbols: set[str]) -> pl.DataFrame:
    base = base.with_columns(pl.col("symbol").cast(pl.String).str.to_uppercase(), _date_expr("date"))
    base_rows = base.filter(pl.col("symbol").is_in(sorted(base_symbols)))
    specs = options.select("underlying_symbol", "document_symbol", "entry_date")
    docs = base.join(specs, left_on="symbol", right_on="underlying_symbol", how="inner").drop("symbol")
    docs = docs.rename({"document_symbol": "symbol"})
    if name != "daily":
        docs = docs.filter(pl.col("date") <= pl.col("entry_date")).sort(["symbol", "date"])
        docs = docs.with_columns(
            pl.len().over("symbol").alias("_count"),
            pl.int_range(pl.len()).over("symbol").alias("_rank"),
        ).filter(pl.col("_rank") >= pl.col("_count") - 252).drop("_count", "_rank")
    else:
        docs = docs.sort(["symbol", "date"])
    return pl.concat([base_rows, docs], how="diagonal_relaxed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--option-panel", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-symbols-file", type=Path, required=True)
    parser.add_argument("--test-symbols-file", type=Path, required=True)
    parser.add_argument("--option-start-date", default="2025-01-01")
    parser.add_argument("--max-contracts", type=int, default=32)
    parser.add_argument("--group-by-dte", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--raw-warehouse", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    root = args.corpus
    manifest = json.loads((root / "manifest.json").read_text())
    train_symbols = set(pl.read_csv(args.train_symbols_file).get_column("symbol").cast(pl.String).str.to_uppercase())
    test_symbols = set(pl.read_csv(args.test_symbols_file).get_column("symbol").cast(pl.String).str.to_uppercase())
    base_symbols = train_symbols | test_symbols
    if args.raw_warehouse:
        options = _load_raw_first_day(base_symbols, start_year=int(args.option_start_date[:4]), end_year=2026)
    else:
        options = pl.read_parquet(args.option_panel).filter(_date_expr("entry_date") >= pl.lit(args.option_start_date).str.to_date())
    options = _select(options, max_contracts=args.max_contracts, group_by_dte=args.group_by_dte)
    options = options.filter(pl.col("symbol").is_in(sorted(base_symbols)))
    options = options.with_columns(
        pl.col("symbol").alias("underlying_symbol"),
        pl.struct(["symbol", "year", "option_type", "contract_symbol"]).map_elements(
            lambda row: _contract_id(row["symbol"], int(row["year"]), row["option_type"], row["contract_symbol"]),
            return_dtype=pl.String,
        ).alias("document_symbol"),
    ).with_columns(pl.col("document_symbol").alias("symbol"))
    options.write_parquet(args.output_dir / "selected_contract_documents.parquet")
    options.select(
        "document_symbol", "underlying_symbol", "year", "entry_date", "option_type", "dte",
        pl.col("dte_contracts").str.split(",").alias("contract_symbol"),
    ).explode("contract_symbol").write_parquet(args.output_dir / "dte_group_members.parquet")

    taxonomy = pl.read_csv(root / "taxonomy.csv").with_columns(pl.col("symbol").cast(pl.String).str.to_uppercase())
    doc_tax = options.select("underlying_symbol", "document_symbol").join(
        taxonomy, left_on="underlying_symbol", right_on="symbol", how="inner"
    ).drop("underlying_symbol").rename({"document_symbol": "symbol"})
    pl.concat([taxonomy.filter(pl.col("symbol").is_in(sorted(base_symbols))), doc_tax], how="diagonal_relaxed").unique("symbol").write_csv(args.output_dir / "taxonomy.csv")
    for name in ("annual", "quarterly", "daily", "sparse_events"):
        _expand_documents(pl.read_parquet(root / f"{name}.parquet"), options, name, base_symbols).write_parquet(args.output_dir / f"{name}.parquet")

    symbols = sorted(base_symbols)
    train_docs = options.filter(pl.col("underlying_symbol").is_in(sorted(train_symbols))).get_column("document_symbol")
    test_docs = options.filter(~pl.col("underlying_symbol").is_in(sorted(train_symbols))).get_column("document_symbol")
    pl.DataFrame({"symbol": sorted(train_symbols | set(train_docs))}).write_csv(args.output_dir / "train_symbols.csv")
    pl.DataFrame({"symbol": sorted((set(symbols) - train_symbols) | set(test_docs))}).write_csv(args.output_dir / "test_symbols.csv")
    pl.DataFrame({"symbol": symbols + options.get_column("document_symbol").to_list()}).unique("symbol").write_csv(args.output_dir / "symbols.csv")
    manifest["symbols"] = len(symbols) + options.get_column("document_symbol").n_unique()
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"selected_contract_documents": options.height, "symbols": len(symbols), "document_symbols": options.get_column("document_symbol").n_unique()}, indent=2))


if __name__ == "__main__":
    main()
