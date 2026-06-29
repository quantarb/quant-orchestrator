from __future__ import annotations

import argparse

import pandas as pd

from quant_orchestrator.backtests.nautilus_sma import run_nautilus_backtest
from quant_orchestrator.backtests.zipline_sma import run_zipline_backtest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Quant Warehouse strategy examples.")
    parser.add_argument(
        "--framework",
        choices=["all", "zipline", "nautilus"],
        default="all",
        help="Backtesting adapter to run.",
    )
    parser.add_argument(
        "--strategy",
        choices=["sma"],
        default="sma",
        help="Strategy to run.",
    )
    parser.add_argument("--symbols", nargs="+", default=["AAPL"], help="Symbols to backtest.")
    parser.add_argument("--provider", default="yfinance", help="Quant Warehouse price provider.")
    parser.add_argument("--start", default="2020-01-01", help="Inclusive start date.")
    parser.add_argument("--end", default=None, help="Inclusive end date.")
    parser.add_argument("--fast-window", type=int, default=20, help="Fast SMA window.")
    parser.add_argument("--slow-window", type=int, default=50, help="Slow SMA window.")
    parser.add_argument("--capital-base", type=float, default=100_000, help="Starting capital.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = []

    if args.framework in {"all", "zipline"}:
        results.append(
            run_zipline_backtest(
                symbol=args.symbols[0],
                provider=args.provider,
                start=args.start,
                end=args.end,
                fast_window=args.fast_window,
                slow_window=args.slow_window,
                capital_base=args.capital_base,
            ),
        )
    if args.framework in {"all", "nautilus"}:
        results.append(
            run_nautilus_backtest(
                symbol=args.symbols[0],
                provider=args.provider,
                start=args.start,
                end=args.end,
                fast_window=args.fast_window,
                slow_window=args.slow_window,
                capital_base=args.capital_base,
            ),
        )

    print(pd.concat(results, ignore_index=True))
