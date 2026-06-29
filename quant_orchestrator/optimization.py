from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass(frozen=True)
class MetricFilter:
    column: str
    operator: str
    value: float


def build_parameter_grid(
    parameters: Mapping[str, Iterable[Any]],
    *,
    constraints: Iterable[Callable[[dict[str, Any]], bool]] = (),
) -> pd.DataFrame:
    """Build a parameter grid as a dataframe, optionally filtering combinations."""

    names = tuple(parameters.keys())
    rows = []
    for values in product(*(tuple(parameters[name]) for name in names)):
        row = dict(zip(names, values, strict=True))
        if all(constraint(row) for constraint in constraints):
            rows.append(row)
    return pd.DataFrame(rows, columns=names)


def filter_results(table: pd.DataFrame, filters: Iterable[MetricFilter]) -> pd.DataFrame:
    """Apply simple metric filters without hardcoding research-specific thresholds."""

    result = table.copy()
    for metric_filter in filters:
        if metric_filter.column not in result.columns:
            raise KeyError(f"Cannot filter on missing column: {metric_filter.column}")
        values = pd.to_numeric(result[metric_filter.column], errors="coerce")
        if metric_filter.operator == ">":
            mask = values > metric_filter.value
        elif metric_filter.operator == ">=":
            mask = values >= metric_filter.value
        elif metric_filter.operator == "<":
            mask = values < metric_filter.value
        elif metric_filter.operator == "<=":
            mask = values <= metric_filter.value
        elif metric_filter.operator == "==":
            mask = values == metric_filter.value
        elif metric_filter.operator == "!=":
            mask = values != metric_filter.value
        else:
            raise ValueError(f"Unsupported metric filter operator: {metric_filter.operator}")
        result = result[mask.fillna(False)].copy()
    return result


def rank_results(
    table: pd.DataFrame,
    *,
    by: str | list[str],
    ascending: bool | list[bool] = False,
    rank_column: str = "rank",
) -> pd.DataFrame:
    """Sort a result table and attach a one-based rank column."""

    ranked = table.sort_values(by=by, ascending=ascending).reset_index(drop=True).copy()
    ranked[rank_column] = np.arange(1, len(ranked) + 1)
    return ranked


def returns_matrix_from_equity_curves(equity_curves: Mapping[str, pd.Series]) -> pd.DataFrame:
    """Convert named equity curves to an aligned returns matrix."""

    if not equity_curves:
        raise ValueError("At least one equity curve is required")
    columns = {
        name: pd.to_numeric(curve, errors="coerce").pct_change().dropna()
        for name, curve in equity_curves.items()
    }
    matrix = pd.DataFrame(columns).dropna(how="all").fillna(0.0)
    if matrix.empty:
        raise ValueError("Equity curves produced an empty returns matrix")
    return matrix


def optimize_long_only_sharpe_weights(
    returns: pd.DataFrame,
    *,
    max_weight: float = 1.0,
    min_weight: float = 0.0,
    min_active_weight: float = 1e-6,
) -> pd.Series:
    """Optimize long-only weights by in-sample Sharpe ratio."""

    clean = returns.replace([np.inf, -np.inf], np.nan).dropna(how="all").fillna(0.0)
    if clean.empty:
        raise ValueError("returns matrix is empty")
    n = clean.shape[1]
    if n == 1:
        return pd.Series([1.0], index=clean.columns, dtype=float)
    lower = max(0.0, float(min_weight))
    upper = min(1.0, float(max_weight))
    if upper * n < 1.0:
        upper = 1.0 / n
    if lower * n > 1.0:
        lower = 0.0

    mean = clean.mean().to_numpy(dtype=float)
    cov = clean.cov().to_numpy(dtype=float) + np.eye(n) * 1e-8

    def neg_sharpe(weights: np.ndarray) -> float:
        port_mean = float(weights @ mean)
        port_vol = float(np.sqrt(weights @ cov @ weights))
        return -(port_mean / port_vol) if port_vol > 0 else 1e6

    result = minimize(
        neg_sharpe,
        x0=np.full(n, 1.0 / n),
        method="SLSQP",
        bounds=[(lower, upper)] * n,
        constraints=[{"type": "eq", "fun": lambda weights: float(weights.sum() - 1.0)}],
        options={"maxiter": 1_000, "ftol": 1e-12},
    )
    weights = result.x if result.success else np.full(n, 1.0 / n)
    return _normalize_active_weights(pd.Series(weights, index=clean.columns), min_active_weight)


@dataclass(frozen=True)
class RandomMeanVarianceResult:
    weights: pd.Series
    score: float
    accepted_portfolios: int


def optimize_random_mean_variance_weights(
    returns: pd.DataFrame,
    *,
    iterations: int = 20_000,
    max_weight: float = 1.0,
    min_active_weight: float = 0.0,
    risk_aversion: float = 1.0,
    seed: int = 1337,
    group_constraints: Mapping[str, pd.Series] | None = None,
) -> RandomMeanVarianceResult:
    """Random-search mean-variance weights with optional active group constraints."""

    clean = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        raise ValueError("returns matrix is empty")
    n = clean.shape[1]
    rng = np.random.default_rng(seed)
    mean_returns = clean.mean().to_numpy(dtype=float) * 252.0
    covariance = clean.cov().to_numpy(dtype=float) * 252.0
    max_weight = min(1.0, float(max_weight))
    min_active_weight = max(0.0, float(min_active_weight))

    best_score = -np.inf
    best_weights: np.ndarray | None = None
    accepted = 0
    groups = group_constraints or {}

    for _ in range(max(1, int(iterations))):
        weights = rng.dirichlet(np.ones(n))
        if weights.max() > max_weight:
            continue
        active = weights >= min_active_weight
        if groups and any(series.loc[active].nunique() < 2 for series in groups.values()):
            continue
        expected_return = float(weights @ mean_returns)
        expected_variance = float(weights @ covariance @ weights)
        score = expected_return - float(risk_aversion) * expected_variance
        accepted += 1
        if score > best_score:
            best_score = score
            best_weights = weights

    if best_weights is None:
        best_weights = np.full(n, 1.0 / n)
        best_score = float(best_weights @ mean_returns - float(risk_aversion) * (best_weights @ covariance @ best_weights))
    weights = _normalize_active_weights(pd.Series(best_weights, index=clean.columns), min_active_weight)
    return RandomMeanVarianceResult(weights=weights, score=best_score, accepted_portfolios=accepted)


def _normalize_active_weights(weights: pd.Series, min_active_weight: float) -> pd.Series:
    clipped = weights.clip(lower=0.0)
    active = clipped[clipped >= min_active_weight]
    if active.empty:
        active = clipped.sort_values(ascending=False).head(1)
    return (active / active.sum()).sort_values(ascending=False)

