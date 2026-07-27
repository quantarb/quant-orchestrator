"""Build one historical-MTL corpus from all locally available equity symbols."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from quant_warehouse import Warehouse
from quant_warehouse.research_tools.feature_family_eval import (
    FamilyEvaluationConfig,
    build_fundamental_feature_panel,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=250)
    args = parser.parse_args()

    index = pd.read_csv(args.index)
    requested = tuple(str(value) for value in index.strategy_source)
    warehouse = Warehouse()
    profiles = warehouse.catalog.query_symbol_profiles(
        provider="fmp", min_market_cap=0, country="", exchanges=(),
        exclude_etf=True, exclude_fund=True, limit=100_000,
    )
    symbols = tuple(sorted({str(profile.symbol).strip().upper() for profile in profiles if str(profile.symbol).strip()}))
    taxonomy = pd.DataFrame([
        {"symbol": profile.symbol, "sector": profile.sector or "Unknown",
         "subsector": "Unknown", "industry": profile.industry or "Unknown"}
        for profile in profiles
    ]).drop_duplicates("symbol").sort_values("symbol")
    config = FamilyEvaluationConfig(
        market_cap_min=0, country="", exchanges=(), screen_limit=max(100_000, len(symbols)),
        start_date="2018-01-01", end_date=None,
    )
    rows: list[pd.DataFrame] = []
    diagnostics_parts: list[pd.DataFrame] = []
    timings: dict[str, float] = {"raw_panel_build_seconds": 0.0}
    panel_symbols = 0
    for offset in range(0, len(symbols), max(1, args.chunk_size)):
        symbol_chunk = symbols[offset:offset + max(1, args.chunk_size)]
        panel, metadata, diagnostics, chunk_timings = build_fundamental_feature_panel(
            symbol_chunk, config, warehouse=warehouse, strategy_sources=requested, broadcast_to_target=False,
        )
        panel_symbols += int(panel["symbol"].nunique())
        diagnostics_parts.append(diagnostics)
        timings["raw_panel_build_seconds"] += float(chunk_timings.get("raw_panel_build_seconds", 0.0))
        for (source, family), family_meta in metadata.groupby(["source", "family"], sort=True):
            columns = [str(value) for value in family_meta.feature if str(value) in panel.columns]
            if not columns:
                continue
            work = panel[["symbol", "date", *columns]].copy()
            work["symbol"] = work["symbol"].astype(str).str.upper()
            work["year"] = pd.to_datetime(work["date"], errors="coerce").dt.year
            work = work.loc[work["year"].notna()].copy()
            values = work[columns].to_numpy(dtype="float64")
            observed = np.isfinite(values)
            work["_sum"] = np.nansum(np.where(observed, values, 0.0), axis=1)
            work["_sumsq"] = np.nansum(np.where(observed, values * values, 0.0), axis=1)
            work["_abs_sum"] = np.nansum(np.where(observed, np.abs(values), 0.0), axis=1)
            work["_count"] = observed.sum(axis=1)
            work["_min"] = np.where(observed, values, np.inf).min(axis=1)
            work["_max"] = np.where(observed, values, -np.inf).max(axis=1)
            grouped = work.groupby(["symbol", "year"], sort=False)
            aggregate = grouped[["_sum", "_sumsq", "_abs_sum", "_count", "_min", "_max"]].agg("sum")
            aggregate["_min"] = grouped["_min"].min()
            aggregate["_max"] = grouped["_max"].max()
            aggregate["daily_observations"] = grouped.size()
            count = aggregate["_count"].replace(0, np.nan)
            aggregate["document_mean"] = aggregate["_sum"] / count
            aggregate["document_std"] = (aggregate["_sumsq"] / count - aggregate["document_mean"] ** 2).clip(lower=0).pow(0.5)
            aggregate["document_min"] = aggregate["_min"].replace([np.inf, -np.inf], np.nan)
            aggregate["document_max"] = aggregate["_max"].replace([np.inf, -np.inf], np.nan)
            aggregate["document_abs_mean"] = aggregate["_abs_sum"] / count
            aggregate["document_coverage"] = aggregate["_count"] / (aggregate["daily_observations"] * len(columns))
            aggregate = aggregate.reset_index()
            aggregate["feature_family"] = f"{source}.{family}"
            rows.append(aggregate[["symbol", "feature_family", "year", "document_mean", "document_std",
                                   "document_min", "document_max", "document_abs_mean", "document_coverage",
                                   "daily_observations"]])
        print(f"processed {min(offset + len(symbol_chunk), len(symbols))}/{len(symbols)} symbols", flush=True)

    documents = pd.concat(rows, ignore_index=True).sort_values(["symbol", "feature_family", "year"]).reset_index(drop=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    documents.to_parquet(args.output_dir / "annual_documents.parquet", index=False)
    taxonomy.to_csv(args.output_dir / "symbol_taxonomy.csv", index=False)
    pd.concat(diagnostics_parts, ignore_index=True).to_parquet(args.output_dir / "diagnostics.parquet", index=False)
    (args.output_dir / "manifest.json").write_text(json.dumps({
        "universe": "all_catalog_active_equities_no_market_cap_filter",
        "catalog_symbols": len(symbols),
        "panel_symbols": panel_symbols,
        "documents": len(documents),
        "families": int(documents["feature_family"].nunique()),
        "timings": timings,
    }, indent=2))
    print(json.dumps({"documents": len(documents), "symbols": int(documents.symbol.nunique()),
                      "families": int(documents.feature_family.nunique())}, indent=2), flush=True)


if __name__ == "__main__":
    main()
