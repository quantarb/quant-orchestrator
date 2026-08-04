"""Fetch FMP fund/institutional activity events for an MTL symbol universe."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "quant-warehouse"))

from quant_warehouse.research_tools.fund_activity import (
    build_fund_holding_activity_events,
    build_holder_activity_events,
    build_institutional_activity_events,
)


def _load_key() -> str:
    try:
        from dotenv import load_dotenv

        root = Path(__file__).resolve().parents[2]
        for path in (root / "quant-warehouse" / ".env", root / "optimal_trader" / ".env", root / ".env"):
            if path.exists():
                load_dotenv(path, override=False)
    except ImportError:
        pass
    key = os.getenv("FMP_API_KEY") or os.getenv("FMP_API_KEY".upper())
    if not key:
        raise RuntimeError("FMP_API_KEY is not configured")
    return key


def _get(session: requests.Session, url: str, *, params: dict[str, object], key: str) -> list[dict[str, object]]:
    response = session.get(url, params={**params, "apikey": key}, timeout=(5, 30))
    response.raise_for_status()
    payload = response.json()
    return payload if isinstance(payload, list) else []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-year", type=int, default=2013)
    parser.add_argument("--end-year", type=int, default=date.today().year)
    args = parser.parse_args()
    key = _load_key()
    symbols = sorted(pd.read_csv(args.symbols)["symbol"].astype(str).str.upper().unique())
    session = requests.Session()
    parts: list[pd.DataFrame] = []

    for index, symbol in enumerate(symbols, start=1):
        summary_rows: list[dict[str, object]] = []
        for year in range(args.start_year, args.end_year + 1):
            for quarter in range(1, 5):
                try:
                    rows = _get(
                        session,
                        "https://financialmodelingprep.com/stable/institutional-ownership/symbol-positions-summary",
                        params={"symbol": symbol, "year": year, "quarter": quarter},
                        key=key,
                    )
                except requests.RequestException:
                    continue
                summary_rows.extend(rows)
        if summary_rows:
            parts.append(build_institutional_activity_events(pd.DataFrame(summary_rows)))

        holder_rows: list[dict[str, object]] = []
        for year in range(args.start_year, args.end_year + 1):
            for quarter in range(1, 5):
                try:
                    holder_rows.extend(_get(
                        session,
                        "https://financialmodelingprep.com/stable/institutional-ownership/extract-analytics/holder",
                        params={"symbol": symbol, "year": year, "quarter": quarter, "page": 0, "limit": 100,
                        },
                        key=key,
                    ))
                except requests.RequestException:
                    continue
        if holder_rows:
            parts.append(build_holder_activity_events(pd.DataFrame(holder_rows)))

        try:
            exposure = _get(
                session,
                "https://financialmodelingprep.com/stable/etf/asset-exposure",
                params={"symbol": symbol},
                key=key,
            )
        except requests.RequestException:
            exposure = []
        if exposure:
            holdings = pd.DataFrame(exposure).rename(
                columns={"symbol": "fund_symbol", "asset": "symbol", "sharesNumber": "shares"}
            )
            holdings["date"] = pd.Timestamp.utcnow().normalize()
            parts.append(build_fund_holding_activity_events(holdings, fund_type="etf"))
        print(f"processed {index}/{len(symbols)} {symbol}", flush=True)

    events = pd.concat([part for part in parts if not part.empty], ignore_index=True)
    events = events.drop_duplicates(subset=["symbol", "date", "target_family", "fund_id"], keep="last") if "fund_id" in events else events.drop_duplicates()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    events.to_parquet(args.output, index=False)
    print(f"wrote {len(events):,} fund activity events to {args.output}")


if __name__ == "__main__":
    main()
