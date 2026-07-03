from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import numpy as np
import pandas as pd


class Strategy(Protocol):
    @property
    def name(self) -> str: ...

    def compute_weights(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Return target weights indexed by date with symbols as columns."""


@dataclass(frozen=True)
class ExecutionConfig:
    price_col: str = "close"
    fee_bps: float = 2.0
    slippage_bps: float = 2.0
    use_lagged_weights: bool = True
    turnover_half_l1: bool = True


@dataclass(frozen=True)
class BacktestResult:
    strategy_name: str
    stats: dict[str, Any]
    equity_curve: pd.Series
    returns: pd.Series
    turnover: pd.Series
    costs: pd.Series


def ensure_panel_index(panel: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(panel.index, pd.MultiIndex):
        raise TypeError("panel must use a MultiIndex with date and symbol levels")
    names = list(panel.index.names)
    if "date" not in names or "symbol" not in names:
        if len(names) < 2:
            raise ValueError("panel MultiIndex must have date and symbol levels")
        panel = panel.copy()
        panel.index = panel.index.set_names(["date", "symbol"] + names[2:])
    out = panel.copy()
    dates = pd.to_datetime(out.index.get_level_values("date"), errors="coerce").normalize()
    symbols = out.index.get_level_values("symbol").astype(str).str.upper()
    out.index = pd.MultiIndex.from_arrays([dates, symbols], names=["date", "symbol"])
    out = out.loc[~out.index.get_level_values("date").isna()]
    return out.sort_index()


def build_panel_from_daily_by_symbol(
    daily_by_symbol: dict[str, pd.DataFrame],
    *,
    include_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for symbol, frame in daily_by_symbol.items():
        if frame is None or frame.empty:
            continue
        daily = frame.copy()
        daily.columns = [str(column).strip().lower() for column in daily.columns]
        if not isinstance(daily.index, pd.DatetimeIndex):
            if "date" in daily.columns:
                daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
                daily = daily.set_index("date")
            else:
                daily.index = pd.to_datetime(daily.index, errors="coerce")
        daily = daily.sort_index()
        if daily.index.has_duplicates:
            daily = daily[~daily.index.duplicated(keep="last")]
        if include_cols is not None:
            keep = [column for column in include_cols if column in daily.columns]
            daily = daily.loc[:, keep].copy()
        daily["symbol"] = str(symbol).strip().upper()
        daily = daily.set_index("symbol", append=True)
        daily.index = daily.index.set_names(["date", "symbol"])
        frames.append(daily)
    if not frames:
        empty = pd.DataFrame()
        empty.index = pd.MultiIndex.from_arrays([[], []], names=["date", "symbol"])
        return empty
    return ensure_panel_index(pd.concat(frames, axis=0))


def backtest_panel(
    panel: pd.DataFrame,
    *,
    strategy: Strategy,
    cfg: ExecutionConfig | None = None,
) -> BacktestResult:
    cfg = cfg or ExecutionConfig()
    panel0 = ensure_panel_index(panel)
    prices = _align_prices(panel0, cfg.price_col)
    symbol_returns = _compute_symbol_returns(prices)
    weights = strategy.compute_weights(panel0)
    weights = weights.reindex(symbol_returns.index).reindex(columns=symbol_returns.columns).fillna(0.0)
    turnover = _compute_turnover(weights, half_l1=cfg.turnover_half_l1)
    costs = (turnover * ((float(cfg.fee_bps) + float(cfg.slippage_bps)) / 10000.0)).astype(float)
    costs.name = "costs"
    gross_returns = _portfolio_returns(
        weights,
        symbol_returns,
        lag_weights=cfg.use_lagged_weights,
    )
    net_returns = (gross_returns - costs).astype(float)
    net_returns.name = "portfolio_return_net"
    equity = (1.0 + net_returns).cumprod()
    equity.name = "equity"
    return BacktestResult(
        strategy_name=strategy.name,
        stats=_summarize_stats(net_returns, equity, turnover, cfg),
        equity_curve=equity,
        returns=net_returns,
        turnover=turnover,
        costs=costs,
    )


def run_backtest(
    *,
    panel: pd.DataFrame,
    spec: Any,
    title: str | None = None,
) -> tuple[BacktestResult, pd.DataFrame]:
    strategy = getattr(spec, "strategy", None)
    if strategy is None:
        raise ValueError("spec.strategy is required for panel_weight run_backtest")
    cfg = getattr(spec, "execution", None) or getattr(spec, "execution_params", None)
    result = backtest_panel(panel=panel, strategy=strategy, cfg=cfg)
    table = pd.DataFrame([result.stats])
    if title is not None:
        table.insert(0, "title", title)
    return result, table


def _align_prices(panel: pd.DataFrame, price_col: str) -> pd.DataFrame:
    if price_col not in panel.columns:
        raise ValueError(f"panel missing required price column {price_col!r}")
    prices = panel[price_col].unstack("symbol").sort_index()
    return prices.apply(pd.to_numeric, errors="coerce")


def _compute_symbol_returns(prices: pd.DataFrame) -> pd.DataFrame:
    returns = prices.pct_change(fill_method=None)
    return returns.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _compute_turnover(weights: pd.DataFrame, *, half_l1: bool) -> pd.Series:
    turnover = weights.fillna(0.0).diff().abs().fillna(0.0).sum(axis=1)
    if half_l1:
        turnover = turnover * 0.5
    return turnover


def _portfolio_returns(
    weights: pd.DataFrame,
    symbol_returns: pd.DataFrame,
    *,
    lag_weights: bool,
) -> pd.Series:
    aligned = weights.reindex(symbol_returns.index).reindex(columns=symbol_returns.columns).fillna(0.0)
    if lag_weights:
        aligned = aligned.shift(1).fillna(0.0)
    returns = (aligned * symbol_returns).sum(axis=1)
    returns.name = "portfolio_return"
    return returns


def _summarize_stats(
    returns: pd.Series,
    equity: pd.Series,
    turnover: pd.Series,
    cfg: ExecutionConfig,
) -> dict[str, Any]:
    clean_returns = returns.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    total_return = float(equity.iloc[-1] - 1.0) if len(equity) else 0.0
    volatility = float(clean_returns.std(ddof=0))
    sharpe = (
        float(clean_returns.mean() / volatility * np.sqrt(252.0))
        if volatility > 1e-12
        else np.nan
    )
    if len(equity):
        drawdown = (equity / equity.cummax()) - 1.0
        max_drawdown = float(drawdown.min())
    else:
        max_drawdown = 0.0
    return {
        "total_return_pct": 100.0 * total_return,
        "sharpe": sharpe,
        "max_drawdown_pct": 100.0 * max_drawdown,
        "avg_turnover": float(turnover.mean()) if len(turnover) else 0.0,
        "fee_bps": float(cfg.fee_bps),
        "slippage_bps": float(cfg.slippage_bps),
    }
