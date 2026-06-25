from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


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
