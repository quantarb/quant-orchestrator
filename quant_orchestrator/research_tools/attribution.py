from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


DEFAULT_ATTRIBUTION_DIMENSIONS = (
    "calendar_year",
    "sector",
    "industry",
    "market_cap_bucket",
    "volatility_regime",
)


def attach_security_context(
    observations: pd.DataFrame,
    context: pd.DataFrame,
    *,
    dimensions: Sequence[str] = DEFAULT_ATTRIBUTION_DIMENSIONS,
) -> pd.DataFrame:
    """Attach exact-date attribution dimensions without treating them as features."""

    required = {"symbol", "date"}
    for name, frame in (("observations", observations), ("context", context)):
        missing = required.difference(frame.columns)
        if missing:
            raise KeyError(f"{name} missing columns: {sorted(missing)}")
    selected_dimensions = [str(column) for column in dimensions]
    missing_dimensions = set(selected_dimensions).difference(context.columns)
    if missing_dimensions:
        raise KeyError(f"context missing attribution dimensions: {sorted(missing_dimensions)}")
    right = context[["symbol", "date", *selected_dimensions]].copy()
    right["symbol"] = right["symbol"].astype(str).str.strip().str.upper()
    right["date"] = pd.to_datetime(right["date"], errors="coerce").dt.normalize()
    if right.duplicated(["symbol", "date"]).any():
        raise ValueError("security context must have one row per symbol and date")
    left = observations.copy()
    left["symbol"] = left["symbol"].astype(str).str.strip().str.upper()
    left["date"] = pd.to_datetime(left["date"], errors="coerce").dt.normalize()
    return left.merge(right, on=["symbol", "date"], how="left", validate="many_to_one")


def attribute_model_scores(
    scores: pd.DataFrame,
    labels: pd.DataFrame,
    context: pd.DataFrame,
    *,
    group_by: Sequence[str] = ("calendar_year", "sector", "industry"),
    target_col: str = "collapsed_label",
) -> pd.DataFrame:
    """Report directional score quality by model and contextual dimensions."""

    score_required = {"model_id", "symbol", "date", "long_score", "short_score"}
    missing = score_required.difference(scores.columns)
    if missing:
        raise KeyError(f"scores missing columns: {sorted(missing)}")
    label_required = {"symbol", "date", target_col}
    missing = label_required.difference(labels.columns)
    if missing:
        raise KeyError(f"labels missing columns: {sorted(missing)}")
    targets = labels[["symbol", "date", target_col]].copy()
    targets["symbol"] = targets["symbol"].astype(str).str.strip().str.upper()
    targets["date"] = pd.to_datetime(targets["date"], errors="coerce").dt.normalize()
    targets["actual_side"] = targets[target_col].map(_target_side)
    targets = targets.dropna(subset=["actual_side"]).drop_duplicates(["symbol", "date", "actual_side"])
    if targets.duplicated(["symbol", "date"], keep=False).any():
        raise ValueError("labels contain ambiguous long and short targets for the same symbol and date")
    merged = scores.copy()
    merged["symbol"] = merged["symbol"].astype(str).str.strip().str.upper()
    merged["date"] = pd.to_datetime(merged["date"], errors="coerce").dt.normalize()
    merged = merged.merge(targets[["symbol", "date", "actual_side"]], on=["symbol", "date"], how="inner")
    merged = attach_security_context(merged, context, dimensions=group_by)
    merged["long_score"] = pd.to_numeric(merged["long_score"], errors="coerce")
    merged["short_score"] = pd.to_numeric(merged["short_score"], errors="coerce")
    merged = merged.dropna(subset=["long_score", "short_score", "actual_side"])
    if merged.empty:
        return pd.DataFrame()
    merged["predicted_side"] = np.where(merged["long_score"].ge(merged["short_score"]), "long", "short")
    merged["correct"] = merged["predicted_side"].eq(merged["actual_side"])
    merged["confidence"] = merged[["long_score", "short_score"]].max(axis=1)
    merged["long_actual"] = merged["actual_side"].eq("long").astype(float)
    merged["long_brier"] = (merged["long_score"] - merged["long_actual"]) ** 2
    keys = ["model_id", *group_by]
    return (
        merged.groupby(keys, dropna=False, as_index=False)
        .agg(
            observations=("correct", "size"),
            symbols=("symbol", "nunique"),
            directional_accuracy=("correct", "mean"),
            mean_confidence=("confidence", "mean"),
            brier_long=("long_brier", "mean"),
            mean_long_score=("long_score", "mean"),
            mean_short_score=("short_score", "mean"),
            actual_long_share=("long_actual", "mean"),
        )
        .sort_values(keys, kind="stable")
        .reset_index(drop=True)
    )


def attribute_strategy_returns(
    observations: pd.DataFrame,
    context: pd.DataFrame,
    *,
    strategy_col: str = "strategy_id",
    return_col: str = "asset_return",
    position_col: str = "position",
    cost_col: str | None = "cost",
    pnl_col: str | None = None,
    group_by: Sequence[str] = ("calendar_year", "sector", "industry"),
) -> pd.DataFrame:
    """Aggregate additive return or P&L contribution across context buckets."""

    required = {strategy_col, "symbol", "date"}
    if pnl_col is None:
        required.update({return_col, position_col})
    else:
        required.add(pnl_col)
    missing = required.difference(observations.columns)
    if missing:
        raise KeyError(f"strategy observations missing columns: {sorted(missing)}")
    merged = attach_security_context(observations, context, dimensions=group_by)
    if pnl_col is None:
        returns = pd.to_numeric(merged[return_col], errors="coerce")
        positions = pd.to_numeric(merged[position_col], errors="coerce")
        merged["gross_contribution"] = returns * positions
    else:
        merged["gross_contribution"] = pd.to_numeric(merged[pnl_col], errors="coerce")
        position_values = merged[position_col] if position_col in merged.columns else pd.Series(0.0, index=merged.index)
        positions = pd.to_numeric(position_values, errors="coerce").fillna(0.0)
    if cost_col is not None and cost_col in merged.columns:
        merged["cost"] = pd.to_numeric(merged[cost_col], errors="coerce").fillna(0.0)
    else:
        merged["cost"] = 0.0
    merged["net_contribution"] = merged["gross_contribution"] - merged["cost"]
    merged["absolute_exposure"] = positions.abs()
    merged["win"] = merged["net_contribution"].gt(0)
    merged = merged.dropna(subset=["gross_contribution"])
    if merged.empty:
        return pd.DataFrame()
    keys = [strategy_col, *group_by]
    return (
        merged.groupby(keys, dropna=False, as_index=False)
        .agg(
            observations=("net_contribution", "size"),
            symbols=("symbol", "nunique"),
            gross_contribution=("gross_contribution", "sum"),
            costs=("cost", "sum"),
            net_contribution=("net_contribution", "sum"),
            average_absolute_exposure=("absolute_exposure", "mean"),
            win_rate=("win", "mean"),
        )
        .sort_values(keys, kind="stable")
        .reset_index(drop=True)
    )


def _target_side(value: object) -> str | None:
    text = str(value or "").strip().lower()
    if text in {"event_bullish", "oracle_long", "long", "buy"}:
        return "long"
    if text in {"event_bearish", "oracle_short", "short", "sell"}:
        return "short"
    return None
