"""Build a bounded Polars multi-rate corpus from stored warehouse observations."""

from datetime import datetime
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class Parser(argparse.ArgumentParser):
    def error(self, message):
        print("error: " + json.dumps(message))
        self.print_help(sys.stdout)
        raise SystemExit(2)


def main():
    parser = Parser(
        description=__doc__,
        epilog="Example: python scripts/build_multirate_mtl_corpus.py --instrument-roster roster.csv --output-dir artifacts/corpus --end-date 2025-12-31 --option-issuers AAPL",
    )
    parser.add_argument(
        "--instrument-roster",
        type=Path,
        required=True,
        help="CSV with symbol, underlying_symbol, issuer, asset_class and instrument terms",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New output directory; existing corpora are never overwritten",
    )
    parser.add_argument(
        "--start-date", default="1900-01-01",
        help="Inclusive observation start (YYYY-MM-DD); default 1900-01-01 reads all available stored history",
    )
    parser.add_argument("--end-date", required=True, help="Inclusive observation end (YYYY-MM-DD)")
    parser.add_argument(
        "--option-issuers",
        default="",
        help="Comma-separated issuers for actual annual call/put cohorts; default none",
    )
    parser.add_argument(
        "--options-start-year",
        type=int,
        default=2023,
        help="First annual option cohort; default 2023",
    )
    if len(sys.argv) == 1:
        print("bin: " + json.dumps(str(Path(__file__).resolve())))
        print("description: Build a Polars multi-rate corpus from stored warehouse data")
        print("help: Run with --help for required inputs and an example")
        return
    args = parser.parse_args()
    try:
        start, end = datetime.fromisoformat(args.start_date), datetime.fromisoformat(args.end_date)
    except ValueError:
        parser.error("Use ISO dates for --start-date and --end-date")
    if start >= end or not args.instrument_roster.is_file() or args.output_dir.exists():
        parser.error(
            "Require start before end, an existing roster file, and a new output directory"
        )
    import polars as pl
    from quant_orchestrator.research_tools.multirate_corpus import build_fresh_corpus

    try:
        manifest = build_fresh_corpus(
            args.output_dir,
            roster=pl.read_csv(args.instrument_roster),
            start=args.start_date,
            end=args.end_date,
            options_start_year=args.options_start_year,
            option_issuers=tuple(
                s.strip().upper() for s in args.option_issuers.split(",") if s.strip()
            ),
        )
    except Exception as exc:
        print(
            "error: " + json.dumps(str(exc) if isinstance(exc, ValueError) else type(exc).__name__)
        )
        print(
            "help: Inspect warehouse coverage and use a new --output-dir after repairing the input"
        )
        raise SystemExit(1) from None
    print("status: complete")
    print("manifest: " + json.dumps(str(args.output_dir / "manifest.json")))
    print("asset_classes: " + json.dumps(",".join(manifest["asset_classes"])))


if __name__ == "__main__":
    main()
