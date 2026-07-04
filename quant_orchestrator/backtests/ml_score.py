from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd

from quant_orchestrator.platforms.backtesting_frameworks.backtesting_py.data_adapter import build_backtesting_frame


@dataclass(frozen=True)
class MlScoreBacktestResult:
    summary: dict[str, object]
    stats: object


def build_ml_score_signal_frame(
    prices: pd.DataFrame,
    scores: pd.DataFrame,
    *,
    variant: str,
    entry_threshold: float,
    exit_threshold: float,
) -> pd.DataFrame:
    frame = prices.rename(columns=str.lower).copy()
    required = ["open", "high", "low", "close", "volume"]
    missing = set(required) - set(frame.columns)
    if missing:
        raise KeyError(f"prices missing required columns: {sorted(missing)}")
    frame = frame[required].apply(pd.to_numeric, errors="coerce")
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index, errors="coerce")).normalize()
    frame = frame.dropna(subset=["open", "high", "low", "close"]).sort_index()

    score_frame = scores.copy()
    score_frame["date"] = pd.to_datetime(score_frame["date"], errors="coerce").dt.normalize()
    score_frame = score_frame.dropna(subset=["date"]).drop_duplicates("date").set_index("date").sort_index()
    aligned = score_frame.reindex(frame.index)
    positions = _target_positions(
        aligned,
        variant=variant,
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
    )
    previous = positions.shift(1).fillna(0.0)
    entry_size = pd.Series(0.0, index=frame.index)
    exit_portion = pd.Series(0.0, index=frame.index)

    changed = positions.ne(previous)
    for idx in frame.index[changed]:
        old = float(previous.loc[idx])
        new = float(positions.loc[idx])
        if old > 0:
            exit_portion.loc[idx] = 1.0
        elif old < 0:
            exit_portion.loc[idx] = -1.0
        if new != 0:
            entry_size.loc[idx] = 0.999 * np.sign(new)

    out = frame.copy()
    # backtesting.py acts on the current bar's signal for the next order cycle.
    # Shift score-change signals one bar forward to match the vectorized
    # no-lookahead convention used elsewhere in the ML trading diagnostics.
    out["signal_entry_size"] = entry_size.shift(1).fillna(0.0)
    out["signal_exit_portion"] = exit_portion.shift(1).fillna(0.0)
    return build_backtesting_frame(out)


def run_ml_score_signal_backtest(
    prices: pd.DataFrame,
    scores: pd.DataFrame,
    *,
    symbol: str,
    strategy_source: str,
    variant: str,
    entry_threshold: float = 0.5,
    exit_threshold: float = 0.5,
    cash: float = 100_000.0,
    commission_bps: float = 0.5,
    spread_bps: float = 5.0,
) -> MlScoreBacktestResult:
    from backtesting import Backtest
    from backtesting.lib import SignalStrategy

    data = build_ml_score_signal_frame(
        prices,
        scores,
        variant=variant,
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
    )

    class MlScoreSignalStrategy(SignalStrategy):
        def init(self):
            super().init()
            self.set_signal(
                self.data.signal_entry_size,
                self.data.signal_exit_portion,
                plot=False,
            )

    started = perf_counter()
    stats = Backtest(
        data,
        MlScoreSignalStrategy,
        cash=float(cash),
        commission=float(commission_bps) / 10_000.0,
        spread=float(spread_bps) / 10_000.0,
        trade_on_close=False,
        exclusive_orders=True,
        finalize_trades=True,
    ).run()
    elapsed = perf_counter() - started
    buy_hold_return = float(data["Close"].iloc[-1] / data["Close"].iloc[0] - 1.0) if len(data) > 1 else np.nan
    summary = {
        "framework": "backtesting_py_signal",
        "strategy_source": strategy_source,
        "symbol": str(symbol).upper(),
        "variant": variant,
        "days": int(len(data)),
        "trades": int(stats.get("# Trades", 0)),
        "total_return": float(stats.get("Return [%]", np.nan)) / 100.0,
        "buy_hold_total_return": buy_hold_return,
        "excess_total_return": float(stats.get("Return [%]", np.nan)) / 100.0 - buy_hold_return,
        "sharpe": float(stats.get("Sharpe Ratio", np.nan)),
        "max_drawdown": float(stats.get("Max. Drawdown [%]", np.nan)) / 100.0,
        "win_rate": float(stats.get("Win Rate [%]", np.nan)) / 100.0,
        "elapsed_seconds": float(elapsed),
    }
    return MlScoreBacktestResult(summary=summary, stats=stats)


def _target_positions(
    scores: pd.DataFrame,
    *,
    variant: str,
    entry_threshold: float,
    exit_threshold: float,
) -> pd.Series:
    long_score = pd.to_numeric(scores.get("long_score", 0.0), errors="coerce").fillna(0.0)
    short_score = pd.to_numeric(scores.get("short_score", 0.0), errors="coerce").fillna(0.0)
    if {"long_agree_count", "short_agree_count", "model_count"}.issubset(scores.columns):
        model_count = pd.to_numeric(scores.get("model_count"), errors="coerce").fillna(0.0)
        long_agree = pd.to_numeric(scores.get("long_agree_count"), errors="coerce").fillna(0.0)
        short_agree = pd.to_numeric(scores.get("short_agree_count"), errors="coerce").fillna(0.0)
    else:
        model_count = long_agree = short_agree = None
    position = []
    current = 0.0
    for idx, (long_value, short_value) in enumerate(zip(long_score, short_score)):
        if model_count is not None and long_agree is not None and short_agree is not None:
            models = float(model_count.iloc[idx])
            long_ok = models > 0 and float(long_agree.iloc[idx]) == models
            short_ok = models > 0 and float(short_agree.iloc[idx]) == models
        else:
            long_ok = long_value >= short_value
            short_ok = short_value > long_value
        if variant == "long_only":
            if current > 0 and not long_ok:
                current = 0.0
            elif current == 0 and long_value > entry_threshold and long_ok:
                current = 1.0
        elif variant == "short_only":
            if current < 0 and not short_ok:
                current = 0.0
            elif current == 0 and short_value > entry_threshold and short_ok:
                current = -1.0
        elif variant == "long_short":
            if current > 0 and not long_ok:
                current = 0.0
            elif current < 0 and not short_ok:
                current = 0.0
            if current == 0 and long_value > entry_threshold and long_ok:
                current = 1.0
            elif current == 0 and short_value > entry_threshold and short_ok:
                current = -1.0
        else:
            raise ValueError(f"unknown variant {variant!r}")
        position.append(current)
    return pd.Series(position, index=scores.index, dtype="float64")
