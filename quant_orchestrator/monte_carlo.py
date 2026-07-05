from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from quant_orchestrator.platforms.backtesting_frameworks.strategy_artifacts import (
    combine_trade_lists,
    normalize_trade_list,
    read_trade_list_artifact,
)


@dataclass(frozen=True)
class MonteCarloResult:
    paths: pd.DataFrame
    summary: pd.DataFrame


def simulate_return_paths(
    returns: pd.Series,
    *,
    iterations: int = 1_000,
    horizon: int | None = None,
    seed: int = 1337,
    block_size: int = 1,
) -> MonteCarloResult:
    clean = pd.to_numeric(returns, errors="coerce").dropna()
    if clean.empty:
        raise ValueError("Monte Carlo simulation requires at least one return observation")
    horizon = int(horizon or len(clean))
    iterations = max(1, int(iterations))
    block_size = max(1, int(block_size))

    rng = np.random.default_rng(seed)
    sampled = np.vstack(
        [
            _sample_path(clean.to_numpy(dtype=float), horizon, block_size, rng)
            for _ in range(iterations)
        ]
    )
    equity = np.cumprod(1.0 + sampled, axis=1)
    paths = pd.DataFrame(equity).T
    terminal = paths.iloc[-1] - 1.0
    max_drawdown = paths.div(paths.cummax()).sub(1.0).min()
    summary = pd.DataFrame(
        [
            {
                "iterations": iterations,
                "horizon": horizon,
                "terminal_return_mean": float(terminal.mean()),
                "terminal_return_p05": float(terminal.quantile(0.05)),
                "terminal_return_p50": float(terminal.quantile(0.50)),
                "terminal_return_p95": float(terminal.quantile(0.95)),
                "max_drawdown_mean": float(max_drawdown.mean()),
                "max_drawdown_p05": float(max_drawdown.quantile(0.05)),
            }
        ]
    )
    return MonteCarloResult(paths=paths, summary=summary)


def trade_list_return_series(
    trades: pd.DataFrame,
    *,
    return_column: str | None = None,
    capital_column: str | None = None,
) -> pd.Series:
    """Extract realized returns from a standard trade-list artifact."""

    normalized = normalize_trade_list(trades)
    if normalized.empty:
        raise ValueError("Trade-list Monte Carlo requires at least one trade")

    candidates = [return_column] if return_column else []
    candidates.extend(
        [
            "portfolio_return_contribution",
            "ret_dec",
            "return_pct",
            "option_return",
        ]
    )
    for column in candidates:
        if not column or column not in normalized.columns:
            continue
        returns = pd.to_numeric(normalized[column], errors="coerce")
        returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
        if returns.empty:
            continue
        if column == "return_pct" and returns.abs().gt(10).any():
            returns = returns / 100.0
        return returns.rename("trade_return")

    pnl = pd.to_numeric(normalized.get("pnl"), errors="coerce")
    if capital_column and capital_column in normalized.columns:
        capital = pd.to_numeric(normalized[capital_column], errors="coerce")
    elif "equity_entry_notional" in normalized.columns:
        capital = pd.to_numeric(normalized["equity_entry_notional"], errors="coerce")
    elif "notional" in normalized.columns:
        capital = pd.to_numeric(normalized["notional"], errors="coerce")
    else:
        capital = pd.Series(np.nan, index=normalized.index)
    returns = (pnl / capital).replace([np.inf, -np.inf], np.nan).dropna()
    if returns.empty:
        raise ValueError(
            "Trade-list Monte Carlo requires a return column or pnl plus notional/equity_entry_notional"
        )
    return returns.rename("trade_return")


def simulate_trade_list_paths(
    trade_lists: pd.DataFrame | str | Path | Mapping[str, pd.DataFrame | str | Path],
    *,
    return_column: str | None = None,
    capital_column: str | None = None,
    iterations: int = 1_000,
    horizon: int | None = None,
    seed: int = 1337,
    block_size: int = 1,
) -> MonteCarloResult:
    """Run Monte Carlo directly from one or more standard trade-list artifacts."""

    if isinstance(trade_lists, Mapping):
        trades = combine_trade_lists(trade_lists)
    elif isinstance(trade_lists, pd.DataFrame):
        trades = normalize_trade_list(trade_lists)
    else:
        trades = read_trade_list_artifact(trade_lists)
    returns = trade_list_return_series(
        trades,
        return_column=return_column,
        capital_column=capital_column,
    )
    return simulate_return_paths(
        returns,
        iterations=iterations,
        horizon=horizon,
        seed=seed,
        block_size=block_size,
    )


def _sample_path(
    values: np.ndarray,
    horizon: int,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if block_size == 1:
        return rng.choice(values, size=horizon, replace=True)
    starts = rng.integers(0, len(values), size=max(1, int(np.ceil(horizon / block_size))))
    blocks = [np.take(values, range(start, start + block_size), mode="wrap") for start in starts]
    return np.concatenate(blocks)[:horizon]
