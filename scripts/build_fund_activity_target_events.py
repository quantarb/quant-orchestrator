"""Convert stored fund/institutional snapshots into MTL target events."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from quant_warehouse.research_tools.fund_activity import (
    build_fund_holding_activity_events,
    build_institutional_activity_events,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--institutional-summary", type=Path, default=None)
    parser.add_argument("--fund-holdings", type=Path, default=None)
    parser.add_argument("--fund-type", default="etf", choices=("etf", "mutual_fund", "institutional"))
    parser.add_argument("--fund-column", default="fund_symbol")
    parser.add_argument("--security-column", default="symbol")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    parts: list[pd.DataFrame] = []
    if args.institutional_summary is not None:
        parts.append(build_institutional_activity_events(pd.read_parquet(args.institutional_summary)))
    if args.fund_holdings is not None:
        parts.append(
            build_fund_holding_activity_events(
                pd.read_parquet(args.fund_holdings),
                fund_type=args.fund_type,
                fund_column=args.fund_column,
                security_column=args.security_column,
            )
        )
    parts = [part for part in parts if not part.empty]
    if not parts:
        raise SystemExit("no fund or institutional activity rows were produced")
    output = pd.concat(parts, ignore_index=True).sort_values(["symbol", "date", "target_family"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(args.output, index=False)
    print(f"wrote {len(output):,} fund activity target events to {args.output}")


if __name__ == "__main__":
    main()
