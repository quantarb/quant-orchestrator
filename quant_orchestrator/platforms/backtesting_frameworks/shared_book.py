from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


StrategyVariant = Literal["long_only", "short_only", "long_short"]


@dataclass(frozen=True)
class SharedBookCostModel:
    commission_bps: float
    slippage_bps: float

    @property
    def total_bps(self) -> float:
        return float(self.commission_bps) + float(self.slippage_bps)


DEFAULT_SHARED_BOOK_COST_MODELS: dict[str, SharedBookCostModel] = {
    "backtesting_py_shared_book": SharedBookCostModel(commission_bps=1.0, slippage_bps=5.0),
    "zipline_shared_book": SharedBookCostModel(commission_bps=0.5, slippage_bps=5.0),
    "nautilus_shared_book": SharedBookCostModel(commission_bps=1.0, slippage_bps=7.5),
}


def build_shared_book_weights(
    scores: pd.DataFrame,
    symbols: tuple[str, ...] | list[str],
    dates: pd.DatetimeIndex | list[pd.Timestamp],
    *,
    top_k: int,
    variant: StrategyVariant,
    entry_threshold: float = 0.5,
    exit_threshold: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create one shared multi-asset target-weight book from daily ML scores.

    The book has one capacity of ``top_k`` positions. Each active position gets
    fixed weight ``+/- 1 / top_k``. Unfilled slots remain cash. The long/short
    variant ranks long and short candidates together and does not maintain two
    independent books.
    """

    if int(top_k) <= 0:
        raise ValueError("top_k must be positive")
    if variant not in {"long_only", "short_only", "long_short"}:
        raise ValueError(f"unknown variant {variant!r}")
    required = {"symbol", "date", "long_score", "short_score"}
    missing = required - set(scores.columns)
    if missing:
        raise KeyError(f"scores missing required columns: {sorted(missing)}")

    symbol_list = tuple(dict.fromkeys(str(symbol).upper() for symbol in symbols))
    date_index = pd.DatetimeIndex(pd.to_datetime(dates, errors="coerce")).dropna().sort_values()
    normalized_scores = scores.copy()
    normalized_scores["symbol"] = normalized_scores["symbol"].astype(str).str.upper()
    normalized_scores["date"] = pd.to_datetime(normalized_scores["date"], errors="coerce").dt.normalize()
    by_date = {pd.Timestamp(date): group.set_index("symbol") for date, group in normalized_scores.groupby("date", sort=False)}

    positions: dict[str, int] = {}
    rows: list[dict[str, object]] = []
    events: list[dict[str, object]] = []
    top_k_int = int(top_k)

    for date in date_index:
        day = by_date.get(pd.Timestamp(date))
        if day is None or day.empty:
            rows.extend({"date": date, "symbol": symbol, "target_weight": positions.get(symbol, 0) / top_k_int} for symbol in symbol_list)
            continue

        for symbol, side in list(positions.items()):
            if symbol not in day.index:
                continue
            long_score = float(day.at[symbol, "long_score"])
            short_score = float(day.at[symbol, "short_score"])
            if side == 1 and short_score > exit_threshold:
                del positions[symbol]
                events.append({"date": date, "symbol": symbol, "action": "exit_long", "score": short_score})
            elif side == -1 and long_score > exit_threshold:
                del positions[symbol]
                events.append({"date": date, "symbol": symbol, "action": "exit_short", "score": long_score})

        open_slots = max(0, top_k_int - len(positions))
        if open_slots > 0:
            candidates = _entry_candidates(
                day,
                held_symbols=set(positions),
                variant=variant,
                entry_threshold=float(entry_threshold),
            )
            for score, symbol, side, action in candidates[:open_slots]:
                positions[symbol] = side
                events.append({"date": date, "symbol": symbol, "action": action, "score": score})

        rows.extend({"date": date, "symbol": symbol, "target_weight": positions.get(symbol, 0) / top_k_int} for symbol in symbol_list)

    weights = (
        pd.DataFrame(rows)
        .pivot(index="date", columns="symbol", values="target_weight")
        .reindex(index=date_index, columns=symbol_list)
        .fillna(0.0)
    )
    trades = pd.DataFrame(events, columns=["date", "symbol", "action", "score"])
    return weights, trades


def run_shared_book_backtest(
    weights: pd.DataFrame,
    next_returns: pd.DataFrame,
    *,
    cost_bps: float,
    capital_base: float = 1_000_000.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Backtest target weights with costs charged on absolute notional turnover."""

    aligned_returns = next_returns.reindex(index=weights.index, columns=weights.columns).fillna(0.0)
    prev_weights = weights.shift(1).fillna(0.0)
    turnover = weights.sub(prev_weights).abs().sum(axis=1)
    gross_returns = (weights * aligned_returns).sum(axis=1)
    costs = turnover * (float(cost_bps) / 10_000.0)
    net_returns = gross_returns - costs
    equity = float(capital_base) * (1 + net_returns.fillna(0.0)).cumprod()
    equity.name = "portfolio_value"
    turnover.name = "turnover"
    net_returns.name = "strategy_return"
    return net_returns, equity, turnover


def shared_book_performance_metrics(
    returns: pd.Series,
    equity: pd.Series,
    weights: pd.DataFrame,
    trades: pd.DataFrame,
    *,
    framework: str,
    variant: str,
    top_k: int,
    cost_bps: float,
) -> dict[str, object]:
    clean = returns.dropna()
    years = len(clean) / 252 if len(clean) else np.nan
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1) if len(equity) > 1 else np.nan
    annualized_return = float((1 + total_return) ** (1 / years) - 1) if years and years > 0 and total_return > -1 else np.nan
    annualized_vol = float(clean.std() * np.sqrt(252)) if len(clean) else np.nan
    sharpe = float(clean.mean() / clean.std() * np.sqrt(252)) if len(clean) and clean.std() else np.nan
    drawdown = equity / equity.cummax() - 1
    gross = weights.abs().sum(axis=1)
    net = weights.sum(axis=1)
    return {
        "framework": framework,
        "variant": variant,
        "top_k": int(top_k),
        "cost_bps": float(cost_bps),
        "days": int(len(clean)),
        "trades": int(len(trades)),
        "final_equity": float(equity.iloc[-1]),
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_vol": annualized_vol,
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()),
        "win_rate": float(clean.gt(0).mean()) if len(clean) else np.nan,
        "avg_gross_exposure": float(gross.mean()),
        "avg_net_exposure": float(net.mean()),
        "fully_invested_days": float(gross.ge(0.999).mean()),
        "cash_days": float(gross.eq(0).mean()),
    }


def run_shared_book_framework_comparison(
    *,
    scores: pd.DataFrame,
    next_returns: pd.DataFrame,
    symbols: tuple[str, ...] | list[str],
    dates: pd.DatetimeIndex | list[pd.Timestamp],
    variants: tuple[StrategyVariant, ...] = ("long_only", "short_only", "long_short"),
    top_k_values: tuple[int, ...] = (5, 10, 20, 40),
    entry_threshold: float = 0.5,
    exit_threshold: float = 0.5,
    cost_models: dict[str, SharedBookCostModel] | None = None,
    capital_base: float = 1_000_000.0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[str, int], pd.DataFrame]]:
    """Run shared-book variants through common cost/accounting models."""

    models = cost_models or DEFAULT_SHARED_BOOK_COST_MODELS
    weight_artifacts: dict[tuple[str, int], pd.DataFrame] = {}
    trade_logs: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    for variant in variants:
        for top_k in top_k_values:
            weights, trades = build_shared_book_weights(
                scores,
                symbols,
                dates,
                top_k=int(top_k),
                variant=variant,
                entry_threshold=entry_threshold,
                exit_threshold=exit_threshold,
            )
            weight_artifacts[(variant, int(top_k))] = weights
            trades = trades.assign(variant=variant, top_k=int(top_k))
            trade_logs.append(trades)
            for framework, cost_model in models.items():
                returns, equity, _ = run_shared_book_backtest(
                    weights,
                    next_returns,
                    cost_bps=cost_model.total_bps,
                    capital_base=capital_base,
                )
                summary_rows.append(
                    shared_book_performance_metrics(
                        returns,
                        equity,
                        weights,
                        trades,
                        framework=framework,
                        variant=variant,
                        top_k=int(top_k),
                        cost_bps=cost_model.total_bps,
                    )
                )
    trade_log = pd.concat(trade_logs, ignore_index=True) if trade_logs else pd.DataFrame(columns=["date", "symbol", "action", "score", "variant", "top_k"])
    return pd.DataFrame(summary_rows), trade_log, weight_artifacts


def _entry_candidates(
    day: pd.DataFrame,
    *,
    held_symbols: set[str],
    variant: StrategyVariant,
    entry_threshold: float,
) -> list[tuple[float, str, int, str]]:
    candidates: list[tuple[float, str, int, str]] = []
    for symbol, row in day.iterrows():
        symbol = str(symbol).upper()
        if symbol in held_symbols:
            continue
        long_score = float(row["long_score"])
        short_score = float(row["short_score"])
        if variant == "long_only":
            if long_score > entry_threshold:
                candidates.append((long_score, symbol, 1, "enter_long"))
        elif variant == "short_only":
            if short_score > entry_threshold:
                candidates.append((short_score, symbol, -1, "enter_short"))
        elif variant == "long_short":
            best_side = 1 if long_score >= short_score else -1
            best_score = max(long_score, short_score)
            if best_score > entry_threshold:
                candidates.append((best_score, symbol, best_side, "enter_long" if best_side == 1 else "enter_short"))
    candidates.sort(key=lambda candidate: candidate[0], reverse=True)
    return candidates
