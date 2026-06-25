from __future__ import annotations

import argparse

import pandas as pd

from quant_orchestrator.backtests.nautilus_sma import run_nautilus_backtest
from quant_orchestrator.backtests.pandas_sma import run_pandas_backtest
from quant_orchestrator.backtests.zipline_sma import run_zipline_backtest
from quant_orchestrator.trading_app_equity import (
    run_trading_app_nautilus,
    run_trading_app_zipline,
    train_price_model_artifact,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Quant Warehouse strategy examples.")
    parser.add_argument(
        "--framework",
        choices=["all", "pandas", "zipline", "nautilus"],
        default="pandas",
        help="Backtesting adapter to run.",
    )
    parser.add_argument(
        "--strategy",
        choices=["sma", "trading-app-equity"],
        default="sma",
        help="Strategy to run.",
    )
    parser.add_argument("--symbols", nargs="+", default=["AAPL"], help="Symbols to backtest.")
    parser.add_argument("--provider", default="yfinance", help="Quant Warehouse price provider.")
    parser.add_argument("--start", default="2020-01-01", help="Inclusive start date.")
    parser.add_argument("--end", default=None, help="Inclusive end date.")
    parser.add_argument("--fast-window", type=int, default=20, help="Fast SMA window.")
    parser.add_argument("--slow-window", type=int, default=50, help="Slow SMA window.")
    parser.add_argument("--top-k", type=int, default=3, help="Trading app equity top-k names.")
    parser.add_argument(
        "--gross-exposure",
        type=float,
        default=0.50,
        help="Trading app equity gross exposure.",
    )
    parser.add_argument(
        "--prediction-artifact",
        default=None,
        help="Path to an optimal_trader ml_predictions CSV.",
    )
    parser.add_argument(
        "--train-model",
        action="store_true",
        help="Train a price-feature model before backtesting.",
    )
    parser.add_argument("--train-start", default="2020-01-01", help="Training window start.")
    parser.add_argument("--train-end", default="2020-12-31", help="Training window end.")
    parser.add_argument("--backtest-start", default="2021-01-01", help="Scoring/backtest start.")
    parser.add_argument("--max-symbols", type=int, default=None, help="Optional training universe cap.")
    parser.add_argument(
        "--train-symbols",
        nargs="+",
        default=None,
        help="Optional symbols to train on for trading-app-equity.",
    )
    parser.add_argument(
        "--backtest-symbols",
        nargs="+",
        default=None,
        help="Optional symbols to backtest on for trading-app-equity.",
    )
    parser.add_argument("--capital-base", type=float, default=100_000, help="Starting capital.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = []

    if args.strategy == "trading-app-equity":
        prediction_artifact = args.prediction_artifact
        if args.train_model:
            prediction_artifact = str(
                train_price_model_artifact(
                    provider=args.provider,
                    train_start=args.train_start,
                    train_end=args.train_end,
                    backtest_start=args.backtest_start,
                    end=args.end,
                    max_symbols=args.max_symbols,
                    symbols=args.train_symbols,
                ),
            )
        if args.framework in {"all", "zipline"}:
            results.append(
                run_trading_app_zipline(
                    prediction_artifact=prediction_artifact,
                    provider=args.provider,
                    top_k=args.top_k,
                    gross_exposure=args.gross_exposure,
                    capital_base=args.capital_base,
                    end=args.end,
                    symbols=args.backtest_symbols,
                ),
            )
        if args.framework in {"all", "nautilus"}:
            results.append(
                run_trading_app_nautilus(
                    prediction_artifact=prediction_artifact,
                    provider=args.provider,
                    top_k=args.top_k,
                    gross_exposure=args.gross_exposure,
                    capital_base=args.capital_base,
                    end=args.end,
                    symbols=args.backtest_symbols,
                ),
            )
        if not results:
            raise ValueError("trading-app-equity supports --framework zipline, nautilus, or all")
        print(pd.concat(results, ignore_index=True))
        return

    if args.framework in {"all", "pandas"}:
        results.append(
            run_pandas_backtest(
                symbols=args.symbols,
                provider=args.provider,
                start=args.start,
                end=args.end,
                fast_window=args.fast_window,
                slow_window=args.slow_window,
                capital_base=args.capital_base,
            ),
        )
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
