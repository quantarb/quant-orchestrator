from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd

from quant_orchestrator.pipeline import FunctionStage, Pipeline, PipelineContext
from quant_orchestrator.platforms.backtesting_frameworks.nautilus.runner import (
    run_nautilus_signal_strategy,
)
from quant_orchestrator.platforms.backtesting_frameworks.reporting import (
    build_common_summary,
    normalize_equity_curve,
)
from quant_orchestrator.platforms.backtesting_frameworks.shared import (
    build_sma_crossover_frame as build_shared_sma_crossover_frame,
    combine_equity_curves,
    MAG7_SYMBOLS,
    load_price_frame,
    normalize_session_label,
)
from quant_orchestrator.platforms.backtesting_frameworks.zipline.runner import (
    run_zipline_signal_strategy,
)


FAST_WINDOWS = (5, 10, 15, 20, 25, 30, 35, 40, 45, 50)
SLOW_WINDOWS = (60, 70, 80, 90, 100, 120, 140, 160, 180, 200)


@dataclass(frozen=True)
class FrameworkRun:
    raw_result: object
    summary: pd.DataFrame
    equity: pd.Series


@dataclass(frozen=True)
class FrameworkComparisonResult:
    comparison: pd.DataFrame
    factor_report: pd.DataFrame
    framework_summary: pd.DataFrame
    provider_summary: pd.DataFrame
    runs: dict[str, dict[str, FrameworkRun]]


def build_sma_frame(prices: pd.DataFrame, *, fast_window: int, slow_window: int) -> pd.DataFrame:
    return build_shared_sma_crossover_frame(prices, fast_window=fast_window, slow_window=slow_window)


def _combine_equity_sleeves(sleeves: list[pd.Series]) -> pd.Series:
    return combine_equity_curves(sleeves)


def _framework_comparison_rows(table: pd.DataFrame, *, metric: str) -> pd.DataFrame:
    pivot = table.pivot(index="framework", columns="provider", values=metric)
    grand_mean = float(pivot.to_numpy(dtype=float).mean())
    framework_means = pivot.mean(axis=1)
    provider_means = pivot.mean(axis=0)
    framework_ss = float(len(provider_means) * ((framework_means - grand_mean) ** 2).sum())
    provider_ss = float(len(framework_means) * ((provider_means - grand_mean) ** 2).sum())
    total_ss = float(((pivot - grand_mean) ** 2).to_numpy().sum())
    residual_ss = max(0.0, total_ss - framework_ss - provider_ss)
    total = framework_ss + provider_ss + residual_ss
    return pd.DataFrame(
        [
            {
                "metric": metric,
                "framework_ss": framework_ss,
                "provider_ss": provider_ss,
                "residual_ss": residual_ss,
                "framework_share": framework_ss / total if total else 0.0,
                "provider_share": provider_ss / total if total else 0.0,
                "residual_share": residual_ss / total if total else 0.0,
                "dominant_factor": "provider" if provider_ss > framework_ss else "framework",
                "provider_to_framework_ratio": (provider_ss / framework_ss) if framework_ss else float("inf"),
            }
        ],
    )


def _coerce_framework_run(result: object) -> FrameworkRun:
    if isinstance(result, FrameworkRun):
        return result
    if isinstance(result, tuple) and len(result) == 3:
        raw_result, summary, equity = result
        return FrameworkRun(raw_result=raw_result, summary=summary, equity=equity)
    raise TypeError(f"Unsupported framework run result: {type(result)!r}")


def _run_nautilus_isolated(
    frame: pd.DataFrame,
    *,
    symbol: str,
    fast_window: int,
    slow_window: int,
    capital_base: float,
) -> FrameworkRun:
    payload = {
        "frame_json": frame.to_json(orient="split", date_format="iso"),
        "symbol": symbol,
        "fast_window": fast_window,
        "slow_window": slow_window,
        "capital_base": capital_base,
    }
    code = """
from __future__ import annotations

import json
import sys
from io import StringIO
from pathlib import Path

import pandas as pd

repo_root = Path.cwd().resolve()
if not (repo_root / "quant_orchestrator").exists():
    repo_root = repo_root.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from quant_orchestrator.backtests.research import build_sma_frame, run_nautilus

payload = json.loads(sys.stdin.read())
frame = pd.read_json(StringIO(payload["frame_json"]), orient="split")
frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index))
signal_frame = build_sma_frame(
    frame,
    fast_window=int(payload["fast_window"]),
    slow_window=int(payload["slow_window"]),
)
result = run_nautilus(
    signal_frame,
    symbol=payload["symbol"],
    capital_base=float(payload["capital_base"]),
)
summary = result.summary
equity = result.equity
row = summary.iloc[0].to_dict()
row["framework"] = "nautilus"
row["provider"] = payload.get("provider", "")
clean = {}
for key, value in row.items():
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    if hasattr(value, "isoformat"):
        try:
            value = value.isoformat()
        except Exception:
            pass
    clean[key] = value
print(json.dumps(clean, default=str))
print("__EQUITY__")
print(equity.to_json(date_format="iso"))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[2],
        input=json.dumps(payload),
    )
    stdout = completed.stdout.strip().splitlines()
    if "__EQUITY__" not in stdout:
        raise RuntimeError("Nautilus isolated run did not emit an equity payload")
    marker = stdout.index("__EQUITY__")
    row = json.loads(stdout[marker - 1])
    equity = pd.read_json(StringIO(stdout[marker + 1]), typ="series")
    equity.index = pd.DatetimeIndex(pd.to_datetime(equity.index))
    return FrameworkRun(pd.DataFrame([row]), pd.DataFrame([row]), normalize_equity_curve(equity))


def run_framework_comparison(
    *,
    symbol: str,
    providers: tuple[str, ...] = ("yfinance", "fmp"),
    frameworks: tuple[str, ...] = ("backtesting.py", "zipline", "nautilus"),
    start: str = "2020-01-01",
    end: str | None = None,
    fast_window: int = 50,
    slow_window: int = 200,
    capital_base: float = 100_000.0,
) -> FrameworkComparisonResult:
    from quant_orchestrator.platforms.backtesting_frameworks.backtesting_py.sma_crossover import (
        run_sma_crossover_backtest as run_backtesting_py_runner,
    )
    from quant_orchestrator.platforms.backtesting_frameworks.zipline.sma_crossover import (
        run_sma_crossover_backtest as run_zipline_runner,
    )

    def load_provider_frames(context: PipelineContext) -> dict[str, Any]:
        return {
            "provider_frames": {
                provider: load_price_frame(symbol, provider=provider, start=start, end=end)
                for provider in providers
            }
        }

    def run_frameworks(context: PipelineContext) -> dict[str, Any]:
        provider_frames = context.require("provider_frames")
        runs: dict[str, dict[str, FrameworkRun]] = {}
        rows = []
        for provider in providers:
            provider_runs: dict[str, FrameworkRun] = {}
            frame = provider_frames[provider]
            for framework in frameworks:
                started = perf_counter()
                if framework == "nautilus":
                    run = _run_nautilus_isolated(
                        frame,
                        symbol=symbol,
                        fast_window=fast_window,
                        slow_window=slow_window,
                        capital_base=capital_base,
                    )
                else:
                    runners = {
                        "backtesting.py": run_backtesting_py_runner,
                        "zipline": run_zipline_runner,
                    }
                    runner = runners.get(framework)
                    if runner is None:
                        raise ValueError(f"Unsupported framework for comparison: {framework}")
                    run = _coerce_framework_run(
                        runner(
                            frame.loc[:, ["open", "high", "low", "close", "volume"]].copy(),
                            symbol=symbol,
                            fast_window=fast_window,
                            slow_window=slow_window,
                            capital_base=capital_base,
                        ),
                    )
                wall_clock_seconds = perf_counter() - started
                summary = run.summary.copy()
                summary["framework"] = framework
                summary["provider"] = provider
                summary["performance_score"] = summary["total_return"] + summary["max_drawdown"]
                summary["wall_clock_seconds"] = wall_clock_seconds
                rows.append(summary)
                provider_runs[framework] = FrameworkRun(run.raw_result, summary, run.equity)
            runs[provider] = provider_runs
        return {"framework_runs": runs, "comparison_rows": rows}

    def build_reports(context: PipelineContext) -> dict[str, Any]:
        comparison = (
            pd.concat(context.require("comparison_rows"), ignore_index=True)
            .sort_values(["provider", "framework"])
            .reset_index(drop=True)
        )
        factor_report = pd.concat(
            [
                _framework_comparison_rows(comparison, metric="performance_score"),
                _framework_comparison_rows(comparison, metric="total_return"),
                _framework_comparison_rows(comparison, metric="elapsed_seconds"),
                _framework_comparison_rows(comparison, metric="wall_clock_seconds"),
            ],
            ignore_index=True,
        )
        framework_summary = comparison.groupby("framework").agg(
            mean_total_return=("total_return", "mean"),
            mean_max_drawdown=("max_drawdown", "mean"),
            mean_elapsed_seconds=("elapsed_seconds", "mean"),
            mean_wall_clock_seconds=("wall_clock_seconds", "mean"),
            mean_bars_per_second=("bars_per_second", "mean"),
        )
        provider_summary = comparison.groupby("provider").agg(
            mean_total_return=("total_return", "mean"),
            mean_max_drawdown=("max_drawdown", "mean"),
            mean_elapsed_seconds=("elapsed_seconds", "mean"),
            mean_wall_clock_seconds=("wall_clock_seconds", "mean"),
            mean_bars_per_second=("bars_per_second", "mean"),
        )
        return {
            "comparison": comparison,
            "factor_report": factor_report,
            "framework_summary": framework_summary,
            "provider_summary": provider_summary,
        }

    context = Pipeline(
        [
            FunctionStage(
                name="load_provider_frames",
                function=load_provider_frames,
                produced_outputs=("provider_frames",),
            ),
            FunctionStage(
                name="run_frameworks",
                function=run_frameworks,
                required_inputs=("provider_frames",),
                produced_outputs=("framework_runs", "comparison_rows"),
            ),
            FunctionStage(
                name="build_reports",
                function=build_reports,
                required_inputs=("comparison_rows",),
                produced_outputs=("comparison", "factor_report", "framework_summary", "provider_summary"),
            ),
        ],
        name="framework_comparison",
    ).run().context
    return FrameworkComparisonResult(
        comparison=context.require("comparison"),
        factor_report=context.require("factor_report"),
        framework_summary=context.require("framework_summary"),
        provider_summary=context.require("provider_summary"),
        runs=context.require("framework_runs"),
    )


def run_backtesting_py(frame: pd.DataFrame, *, symbol: str, capital_base: float) -> FrameworkRun:
    from backtesting import Backtest, Strategy

    trade_size = max(1, int((capital_base * 0.25) / float(frame["close"].iloc[0])))
    bt_frame = frame.rename(
        columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume", "signal": "Signal"}
    ).copy()
    signal_map = {
        normalize_session_label(date): bool(signal)
        for date, signal in frame["signal"].items()
    }

    class SignalStrategy(Strategy):
        def init(self):
            self.trade_size = trade_size

        def next(self):
            bullish = signal_map.get(normalize_session_label(self.data.index[-1]), False)
            if bullish and not self.position:
                self.buy(size=self.trade_size)
            elif not bullish and self.position:
                self.position.close()

    started = perf_counter()
    stats = Backtest(
        bt_frame,
        SignalStrategy,
        cash=capital_base,
        commission=0.0,
        trade_on_close=False,
        exclusive_orders=True,
    ).run()
    elapsed = perf_counter() - started
    equity = stats["_equity_curve"]["Equity"].rename("portfolio_value")
    summary = build_common_summary(
        framework="backtesting.py",
        symbol=symbol,
        equity=equity,
        elapsed_seconds=elapsed,
        bars=len(bt_frame),
        trades=int(stats["# Trades"]),
    )
    summary["native_return_pct"] = float(stats["Return [%]"])
    summary["native_sharpe"] = float(stats["Sharpe Ratio"]) if pd.notna(stats["Sharpe Ratio"]) else None
    summary["native_max_drawdown_pct"] = float(stats["Max. Drawdown [%]"])
    return FrameworkRun(stats, summary, normalize_equity_curve(equity))


def run_zipline(frame: pd.DataFrame, *, symbol: str, capital_base: float) -> FrameworkRun:
    perf, summary, equity = run_zipline_signal_strategy(
        frame,
        symbol=symbol,
        capital_base=capital_base,
    )
    summary["native_last_value"] = float(perf["portfolio_value"].iloc[-1])
    return FrameworkRun(perf, summary, normalize_equity_curve(equity))


def run_nautilus(frame: pd.DataFrame, *, symbol: str, capital_base: float) -> FrameworkRun:
    fills_report, summary, equity = run_nautilus_signal_strategy(
        frame,
        symbol=symbol,
        capital_base=capital_base,
    )
    summary["native_fills"] = int(len(fills_report))
    summary["native_last_value"] = float(equity.iloc[-1])
    return FrameworkRun(fills_report, summary, normalize_equity_curve(equity))


def run_framework_portfolio(
    framework: str,
    *,
    symbol_frames: dict[str, pd.DataFrame],
    capital_base: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, FrameworkRun]]:
    capital_per_symbol = capital_base / max(1, len(symbol_frames))
    runs: dict[str, FrameworkRun] = {}
    summaries = []
    sleeves = []
    elapsed = 0.0
    trades = 0

    for symbol, frame in symbol_frames.items():
        started = perf_counter()
        run = {
            "backtesting.py": run_backtesting_py,
            "zipline": run_zipline,
            "nautilus": run_nautilus,
        }[framework](frame, symbol=symbol, capital_base=capital_per_symbol)
        elapsed += perf_counter() - started
        summaries.append(run.summary)
        sleeves.append(run.equity.rename(symbol))
        trades += int(run.summary["trades"].iloc[0])
        runs[symbol] = run

    summary = pd.concat(summaries, ignore_index=True)
    portfolio_equity = _combine_equity_sleeves(sleeves)
    portfolio_summary = build_common_summary(
        framework=framework,
        symbol="PORTFOLIO",
        equity=portfolio_equity,
        elapsed_seconds=elapsed,
        bars=len(portfolio_equity),
        trades=trades,
    )
    return summary, portfolio_summary, runs


def build_symbol_frames(
    *,
    provider: str,
    fast_window: int,
    slow_window: int,
    start: str,
    end: str | None,
    symbols: tuple[str, ...] = MAG7_SYMBOLS,
) -> dict[str, pd.DataFrame]:
    frames = {}
    for symbol in symbols:
        prices = load_price_frame(symbol, provider=provider, start=start, end=end)
        frames[symbol] = build_sma_frame(prices, fast_window=fast_window, slow_window=slow_window)
    return frames
