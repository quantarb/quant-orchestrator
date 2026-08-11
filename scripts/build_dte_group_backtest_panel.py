"""Build synthetic annual-DTE basket outcomes from raw bid/ask history."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _dates(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hold-days", type=int, default=30)
    parser.add_argument("--dte", type=int, help="Restrict the group table to one DTE bucket.")
    args = parser.parse_args()

    from quant_warehouse.platforms.data_providers.thetadata.options import read_thetadata_eod_option_chain

    groups = pd.read_parquet(args.groups)
    groups["entry_date"] = _dates(groups["entry_date"])
    groups["option_type"] = groups["option_type"].astype(str).str.lower()
    groups["underlying_symbol"] = groups["underlying_symbol"].astype(str).str.upper()
    groups["dte_contracts"] = groups["dte_contracts"].astype(str)
    if args.dte is not None:
        groups = groups.loc[pd.to_numeric(groups["dte"], errors="coerce").eq(args.dte)].copy()
        if groups.empty:
            raise RuntimeError(f"no groups found for DTE {args.dte}")
    rows: list[dict[str, object]] = []
    for (underlying, entry_date), group_rows in groups.groupby(["underlying_symbol", "entry_date"], sort=True):
        end = entry_date + pd.Timedelta(days=args.hold_days + 7)
        chain = read_thetadata_eod_option_chain(
            underlying, start_date=entry_date, end_date=end,
            columns=["snapshot_date", "contract_symbol", "expiration", "bid", "ask"],
        )
        if chain is None or chain.empty:
            continue
        chain["snapshot_date"] = _dates(chain["snapshot_date"])
        chain["expiration"] = _dates(chain["expiration"])
        chain["contract_symbol"] = chain["contract_symbol"].astype(str)
        chain["bid"] = pd.to_numeric(chain["bid"], errors="coerce")
        chain["ask"] = pd.to_numeric(chain["ask"], errors="coerce")
        for row in group_rows.itertuples(index=False):
            contracts = [x for x in row.dte_contracts.split(",") if x]
            member = chain.loc[chain["contract_symbol"].isin(contracts)].copy()
            if member.empty:
                continue
            entry = member.loc[member["snapshot_date"].eq(entry_date)].copy()
            entry = entry.loc[entry["ask"].gt(0.0)]
            if entry.empty:
                continue
            entry = entry.sort_values("contract_symbol").drop_duplicates("contract_symbol", keep="last")
            exit_target = entry["expiration"].where(
                entry["expiration"].lt(entry_date + pd.Timedelta(days=args.hold_days)),
                entry_date + pd.Timedelta(days=args.hold_days),
            )
            candidate = member.merge(entry[["contract_symbol", "ask"]], on="contract_symbol", suffixes=("", "_entry"))
            candidate["exit_target"] = candidate["contract_symbol"].map(dict(zip(entry.contract_symbol, exit_target)))
            candidate = candidate.loc[
                candidate["snapshot_date"].ge(entry_date)
                & candidate["snapshot_date"].le(candidate["exit_target"])
            ].sort_values(["contract_symbol", "snapshot_date"])
            exit_rows = candidate.groupby("contract_symbol", as_index=False).tail(1)
            exit_rows = exit_rows.loc[exit_rows["bid"].notna() & exit_rows["ask_entry"].gt(0.0)]
            if exit_rows.empty:
                continue
            returns = exit_rows["bid"] / exit_rows["ask_entry"] - 1.0
            rows.append({
                "symbol": row.document_symbol,
                "underlying_symbol": underlying,
                "entry_date": entry_date,
                "side": "long" if row.option_type == "call" else "short",
                "option_type": row.option_type,
                "dte": int(row.dte),
                "group_contract_count": int(row.dte_contract_count),
                "priced_contract_count": int(len(returns)),
                "entry_ask": float(exit_rows["ask_entry"].mean()),
                "exit_bid": float(exit_rows["bid"].mean()),
                "execution_return": float(np.nanmean(returns.to_numpy(float))),
            })
    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("no DTE groups could be priced from raw bid/ask history")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)
    print(out.groupby("entry_date").size().to_string())
    print(f"groups={len(out)} priced_members={int(out.priced_contract_count.sum())}")


if __name__ == "__main__":
    main()
