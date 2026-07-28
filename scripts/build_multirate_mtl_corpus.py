"""Build annual, quarterly, daily, and sparse-event MTL input tables."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from quant_warehouse import Warehouse
from quant_warehouse.research_tools.feature_family_eval import (
    FamilyEvaluationConfig,
    build_fundamental_feature_panel,
)
from quant_orchestrator.platforms.ml_frameworks.torch.models.transformers.multirate.text_embeddings import (
    canonical_row_text,
    encode_frozen_text_rows,
)


FAMILIES = (
    "fmp.economic_indicators", "fmp.fmp_balance_mcap", "fmp.fmp_cash_mcap",
    "fmp.fmp_company_news", "fmp.fmp_daily_ev_multiple", "fmp.fmp_daily_ev_yield",
    "fmp.fmp_daily_mcap_multiple", "fmp.fmp_daily_mcap_yield", "fmp.fmp_employee_count",
    "fmp.fmp_esg_scores", "fmp.fmp_historical_ratings", "fmp.fmp_income_mcap",
    "fmp.fmp_institutional_position_summary", "fmp.fmp_quarterly_financial_estimates",
    "fmp.industry_pe", "fmp.industry_performance", "fmp.sector_pe", "fmp.sector_performance",
    "fmp.time_calendar", "fmp.treasury_rates", "financetoolkit.ft_growth_balance",
    "financetoolkit.ft_growth_cash", "financetoolkit.ft_growth_income",
    "financetoolkit.ft_ratios_efficiency", "financetoolkit.ft_ratios_liquidity",
    "financetoolkit.ft_ratios_profitability", "financetoolkit.ft_ratios_solvency",
    "financetoolkit.ft_ratios_valuation",
)

# These families are backed by FMP/OpenBB routes that accept an explicit
# annual or quarterly period.  The corpus keeps both provider-period versions
# as distinct feature families.
DUAL_PERIOD_FAMILIES = frozenset({
    "fmp.fmp_balance_mcap", "fmp.fmp_cash_mcap", "fmp.fmp_income_mcap",
    "financetoolkit.ft_growth_balance", "financetoolkit.ft_growth_cash",
    "financetoolkit.ft_growth_income", "financetoolkit.ft_ratios_efficiency",
    "financetoolkit.ft_ratios_liquidity", "financetoolkit.ft_ratios_profitability",
    "financetoolkit.ft_ratios_solvency", "financetoolkit.ft_ratios_valuation",
})
QUARTER_ONLY_FAMILIES = frozenset({"fmp.fmp_quarterly_financial_estimates"})


def _canonical_issuer_key(profile: object | None, symbol: str) -> str:
    """Build a stable issuer label independent of share-class symbols."""
    cik = str(getattr(profile, "cik", None) or "").strip()
    if cik and cik.lower() not in {"none", "nan"}:
        return f"cik:{cik}"
    company_name = str(getattr(profile, "company_name", None) or "").strip()
    normalized_name = re.sub(r"\s+", " ", company_name).casefold()
    if normalized_name and normalized_name not in {"none", "nan"}:
        return f"name:{normalized_name}"
    return f"symbol:{str(symbol).strip().upper()}"


def _load_project_credentials() -> None:
    """Load the shared project .env before invoking OpenBB/FMP."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    project_root = Path(__file__).resolve().parents[2]
    candidates = (
        project_root / "quant-warehouse" / ".env",
        project_root / "optimal_trader" / ".env",
        project_root / ".env",
    )
    for path in candidates:
        if path.exists():
            load_dotenv(path, override=False)


def _subsector_map(symbols: tuple[str, ...]) -> dict[str, str]:
    """Read FMP/OpenBB ``subSector`` classifications for the symbol universe."""
    _load_project_credentials()
    try:
        from quant_warehouse.ingest.constituent_fetch import fetch_index_constituents

        rows = fetch_index_constituents(("sp500", "nasdaq", "dowjones"))
    except Exception as exc:
        print(f"warning: unable to load OpenBB/FMP subsectors: {exc}", flush=True)
        return {}
    wanted = set(symbols)
    result: dict[str, str] = {}
    for row in rows:
        symbol = str(row.get("symbol") or row.get("ticker") or "").strip().upper()
        subsector = str(
            row.get("subSector") or row.get("sub_sector") or row.get("subsector") or ""
        ).strip()
        if symbol in wanted and subsector and subsector.lower() not in {"nan", "none"}:
            result.setdefault(symbol, subsector)
    return result


def _family_values(panel: pd.DataFrame, metadata: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    base = panel[["symbol", "date"]].copy()
    base["symbol"] = base["symbol"].astype(str).str.upper()
    families: list[str] = []
    for family, group in metadata.groupby("family", sort=True):
        columns = [str(value) for value in group.feature if str(value) in panel.columns]
        if not columns:
            continue
        numeric = panel[columns].apply(pd.to_numeric, errors="coerce")
        token = pd.DataFrame({
            "symbol": base["symbol"], "date": pd.to_datetime(base["date"], errors="coerce"),
            f"value__{family}": numeric.mean(axis=1),
            f"presence__{family}": numeric.notna().any(axis=1).astype("float32"),
        })
        base = base.merge(token, on=["symbol", "date"], how="left", validate="one_to_one")
        families.append(str(family))
    return base.sort_values(["symbol", "date"]).reset_index(drop=True), families


def _rate_table(daily: pd.DataFrame, families: list[str], rate: str) -> pd.DataFrame:
    work = daily.copy()
    if rate == "daily":
        return work
    dates = pd.to_datetime(work["date"])
    work["_period"] = dates.dt.to_period("Q" if rate == "quarterly" else "Y").astype(str)
    group_columns = ["symbol", "_period"]
    aggregations: dict[str, str] = {f"value__{family}": "mean" for family in families}
    aggregations.update({f"presence__{family}": "max" for family in families})
    out = work.groupby(group_columns, sort=True).agg(aggregations).reset_index()
    dates_by_period = work.groupby(group_columns, sort=True)["date"].max().rename("date").reset_index()
    out = out.merge(dates_by_period, on=group_columns, how="left", validate="one_to_one")
    return out.drop(columns=["_period"], errors="ignore").sort_values(["symbol", "date"]).reset_index(drop=True)


def _build_sparse_events(events: pd.DataFrame, output: Path, device: str) -> list[str]:
    if events.empty:
        return []
    events = events.copy()
    # Rebuilding a corpus from an already encoded sparse-event table should
    # preserve its frozen text vectors rather than treating them as raw
    # numeric event fields and embedding an empty string again.
    encoded_columns = {"symbol", "date", "target_family", "event_date", "signal_value", *[f"text_{i}" for i in range(7)]}
    if encoded_columns.issubset(events.columns):
        encoded = events[list(encoded_columns)].copy()
        encoded["symbol"] = encoded["symbol"].astype(str).str.upper()
        encoded["date"] = pd.to_datetime(encoded["date"], errors="coerce", utc=True)
        encoded["event_date"] = pd.to_datetime(encoded["event_date"], errors="coerce", utc=True)
        encoded = encoded.loc[encoded["event_date"].notna() & encoded["target_family"].notna()]
        encoded.sort_values(["symbol", "date", "event_date"]).to_parquet(output / "sparse_events.parquet", index=False)
        return sorted(encoded["target_family"].astype(str).unique())
    events["symbol"] = events["symbol"].astype(str).str.upper()
    events["event_date"] = pd.to_datetime(events["event_date"], errors="coerce", utc=True)
    if "reported_date" in events:
        events["reported_date"] = pd.to_datetime(events["reported_date"], errors="coerce", utc=True)
    events = events.loc[events["event_date"].notna() & events["target_family"].notna()].copy()
    target_families = sorted(events["target_family"].astype(str).unique())
    excluded = {"symbol", "event_date", "reported_date", "target_family", "raw_json", "source_event_id"}
    text_columns = [
        column for column in events.columns
        if column not in excluded
        and not pd.api.types.is_numeric_dtype(events[column])
        and not pd.api.types.is_datetime64_any_dtype(events[column])
    ]
    numeric_columns = [
        column for column in events.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(events[column])
    ]
    events["signal_value"] = events[numeric_columns].apply(pd.to_numeric, errors="coerce").mean(axis=1) if numeric_columns else 0.0
    events["event_text"] = canonical_row_text(events, text_columns) if text_columns else ""
    text_values = encode_frozen_text_rows(
        events["event_text"].tolist(), model_name="axiotic/ogma-small", device=device, local_files_only=True,
    )[:, :7]
    rows: list[dict[str, object]] = []
    for index, event in events.reset_index(drop=True).iterrows():
        availability = event.get("reported_date") if pd.notna(event.get("reported_date")) else event["event_date"]
        row = {"symbol": event["symbol"], "date": availability, "target_family": str(event["target_family"]), "event_date": event["event_date"]}
        row["signal_value"] = float(event["signal_value"]) if pd.notna(event["signal_value"]) else 0.0
        for dimension, value in enumerate(text_values[index]):
            row[f"text_{dimension}"] = float(value)
        rows.append(row)
    sparse = pd.DataFrame(rows).sort_values(["symbol", "date", "event_date"]).reset_index(drop=True)
    sparse.to_parquet(output / "sparse_events.parquet", index=False)
    return target_families


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=Path, required=True)
    parser.add_argument("--target-events", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--text-device", default="cuda")
    parser.add_argument("--start-date", default="1900-01-01")
    args = parser.parse_args()
    symbols = tuple(sorted(pd.read_csv(args.symbols)["symbol"].astype(str).str.upper().unique()))
    subsectors = _subsector_map(symbols)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    warehouse = Warehouse()
    config = FamilyEvaluationConfig(
        market_cap_min=0, country="", exchanges=(), screen_limit=100_000,
        start_date=args.start_date,
    )
    daily_parts: list[pd.DataFrame] = []
    metadata_parts: list[pd.DataFrame] = []
    for start in range(0, len(symbols), max(1, args.chunk_size)):
        chunk = symbols[start:start + max(1, args.chunk_size)]
        panel_parts: list[pd.DataFrame] = []
        metadata_parts_for_chunk: list[pd.DataFrame] = []
        dual_families = sorted(DUAL_PERIOD_FAMILIES)
        for period in ("quarter", "annual"):
            period_panel, period_metadata, _, _ = build_fundamental_feature_panel(
                chunk,
                config,
                warehouse=warehouse,
                strategy_sources=dual_families,
                broadcast_to_target=False,
                fundamental_period=period,
                family_suffix="quarterly" if period == "quarter" else "annual",
            )
            panel_parts.append(period_panel)
            metadata_parts_for_chunk.append(period_metadata)

        quarterly_only_panel, quarterly_only_metadata, _, _ = build_fundamental_feature_panel(
            chunk,
            config,
            warehouse=warehouse,
            strategy_sources=sorted(QUARTER_ONLY_FAMILIES),
            broadcast_to_target=False,
            fundamental_period="quarter",
            family_suffix="quarterly",
        )
        panel_parts.append(quarterly_only_panel)
        metadata_parts_for_chunk.append(quarterly_only_metadata)

        non_period_families = sorted(set(FAMILIES).difference(DUAL_PERIOD_FAMILIES).difference(QUARTER_ONLY_FAMILIES))
        non_period_panel, non_period_metadata, _, _ = build_fundamental_feature_panel(
            chunk,
            config,
            warehouse=warehouse,
            strategy_sources=non_period_families,
            broadcast_to_target=False,
        )
        panel_parts.append(non_period_panel)
        metadata_parts_for_chunk.append(non_period_metadata)

        panel = panel_parts[0]
        for part in panel_parts[1:]:
            extra = part.drop(columns=["close", "daily_market_cap", *[c for c in part.columns if c.startswith("forward_return_")]], errors="ignore")
            panel = panel.merge(extra, on=["symbol", "date"], how="outer", validate="one_to_one")
        metadata = pd.concat(metadata_parts_for_chunk, ignore_index=True).drop_duplicates()
        values, families = _family_values(panel, metadata)
        daily_parts.append(values)
        metadata_parts.append(pd.DataFrame({"family": families}))
        print(f"processed {min(start + len(chunk), len(symbols))}/{len(symbols)}", flush=True)
    daily = pd.concat(daily_parts, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)
    families = sorted({str(value) for part in metadata_parts for value in part["family"]})
    for rate in ("daily", "quarterly", "annual"):
        _rate_table(daily, families, rate).to_parquet(output / f"{rate}.parquet", index=False)
    target_families = _build_sparse_events(pd.read_parquet(args.target_events), output, args.text_device)
    profiles = warehouse.catalog.query_symbol_profiles(provider="fmp", min_market_cap=0, country="", exchanges=(), exclude_etf=True, exclude_fund=True, limit=100_000)
    profiles_by_symbol = {str(profile.symbol).strip().upper(): profile for profile in profiles}
    taxonomy = pd.DataFrame([
        {"symbol": symbol, "issuer": _canonical_issuer_key(profiles_by_symbol.get(symbol), symbol),
         "sector": (getattr(profiles_by_symbol.get(symbol), "sector", None) or "Unknown"),
         "subsector": subsectors.get(symbol, "Unknown"),
         "industry": (getattr(profiles_by_symbol.get(symbol), "industry", None) or "Unknown")}
        for symbol in symbols
    ])
    taxonomy.to_csv(output / "taxonomy.csv", index=False)
    (output / "manifest.json").write_text(json.dumps({"symbols": len(symbols), "feature_families": families, "target_families": target_families}, indent=2))


if __name__ == "__main__":
    main()
