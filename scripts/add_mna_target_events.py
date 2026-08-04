"""Append FMP merger-and-acquisition events to MTL target-event inputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from quant_warehouse.ingest.corporate_events_fetch import fetch_fmp_corporate_events
from quant_orchestrator.platforms.ml_frameworks.torch.models.transformers.multirate.text_embeddings import (
    canonical_row_text,
    encode_frozen_text_rows,
)


MNA_TARGET_FAMILY = "equity.corporate.merger_acquisition"


def _load_credentials() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    for path in (
        Path(__file__).resolve().parents[2] / "quant-warehouse" / ".env",
        Path(__file__).resolve().parents[1] / ".env",
    ):
        if path.exists():
            load_dotenv(path, override=False)


def _normalise_events(rows: list[dict], symbols: set[str]) -> pd.DataFrame:
    output: list[dict[str, object]] = []
    for row in rows:
        event_date = pd.to_datetime(row.get("transactionDate"), errors="coerce", utc=True)
        accepted_date = pd.to_datetime(row.get("acceptedDate"), errors="coerce", utc=True)
        if pd.isna(event_date):
            continue
        source_symbol = str(row.get("symbol") or "").strip().upper()
        target_symbol = str(row.get("targetedSymbol") or "").strip().upper()
        event_id = "|".join(str(row.get(key) or "") for key in ("transactionDate", "cik", "targetedCik", "link"))
        common = {
            "event_date": event_date,
            "reported_date": accepted_date if pd.notna(accepted_date) else event_date,
            "target_family": MNA_TARGET_FAMILY,
            "source_event_id": event_id,
            "source_id": event_id,
            "company_cik": row.get("cik"),
            "targeted_cik": row.get("targetedCik"),
            "company_name": row.get("companyName"),
            "targeted_company_name": row.get("targetedCompanyName"),
            "targeted_symbol": target_symbol or None,
            "transaction_date": event_date,
            "link": row.get("link"),
        }
        participants = ((source_symbol, "acquirer"), (target_symbol, "target"))
        for symbol, role in participants:
            if symbol and symbol in symbols:
                output.append({"symbol": symbol, "event_role": role, **common})
    return pd.DataFrame(output)


def _encode_events(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=["symbol", "date", "target_family", "event_date", "signal_value", *[f"text_{i}" for i in range(7)]])
    text_columns = [
        "event_role", "company_name", "targeted_company_name", "targeted_symbol", "link",
    ]
    text = canonical_row_text(events, text_columns)
    vectors = encode_frozen_text_rows(
        list(text), model_name="axiotic/ogma-small", device="cuda", local_files_only=True,
    )[:, :7]
    encoded = pd.DataFrame({
        "symbol": events["symbol"].astype(str).str.upper().to_numpy(),
        "date": events["reported_date"].to_numpy(),
        "target_family": events["target_family"].to_numpy(),
        "event_date": events["event_date"].to_numpy(),
        "signal_value": 0.0,
    })
    for index in range(vectors.shape[1]):
        encoded[f"text_{index}"] = vectors[:, index].astype("float32")
    return encoded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, action="append", required=True)
    parser.add_argument("--max-pages", type=int, default=100)
    args = parser.parse_args()
    _load_credentials()
    rows_by_directory: dict[Path, set[str]] = {}
    for directory in args.input_dir:
        rows_by_directory[directory] = set(pd.read_csv(directory / "symbols.csv")["symbol"].astype(str).str.upper())
    rows = fetch_fmp_corporate_events(
        ("merger_acquisition",), page_limit=100, max_pages=args.max_pages,
    )
    for directory in args.input_dir:
        mna = _encode_events(_normalise_events(rows, rows_by_directory[directory]))
        path = directory / "target_events.parquet"
        # sparse_events.parquet is the encoded, complete pre-M&A event set;
        # use it as the recovery/rebuild source so repeated runs never lose
        # same-day events or duplicate rows from older target families.
        existing_path = directory / "target_events.parquet"
        existing = pd.read_parquet(existing_path if existing_path.exists() else directory / "sparse_events.parquet")
        existing = existing.loc[existing["target_family"].ne(MNA_TARGET_FAMILY)]
        # Preserve all existing rows: multiple events for one symbol/date are
        # distinct subtokens and must not be collapsed.
        combined = pd.concat([existing, mna], ignore_index=True, sort=False)
        for column in ("date", "event_date"):
            if column in combined:
                combined[column] = pd.to_datetime(combined[column], errors="coerce", utc=True).dt.tz_localize(None)
        combined.to_parquet(path, index=False)
        print(f"{directory}: mna={len(mna)} target_events={len(combined)}", flush=True)


if __name__ == "__main__":
    main()
