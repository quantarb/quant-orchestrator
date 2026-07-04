from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd


ACTIVE_WEIGHT_EPSILON = 1e-12
BPS_TO_DECIMAL = 10000.0
TRADING_DAYS_PER_YEAR = 252.0
SUMMARY_DECIMALS = 8


@dataclass(frozen=True)
class OptimalTraderBacktestConfig:
    start_date: str | None = None
    end_date: str | None = None
    fee_bps: float = 0.0
    slippage_bps: float = 0.0
    transaction_cost_bps: float = 0.0
    max_position_weight: float = 0.0
    min_price: float = 0.0
    min_dollar_volume: float = 0.0
    short_borrow_bps_annual: float = 0.0
    execution_delay_days: int = 1
    turnover_half_l1: bool = True
    use_lagged_weights: bool = True
    allowed_symbols: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any] | None = None) -> "OptimalTraderBacktestConfig":
        raw = dict(config or {})
        return cls(
            start_date=str(raw.get("backtest_start_date") or raw.get("start_date") or "").strip() or None,
            end_date=str(raw.get("backtest_end_date") or raw.get("end_date") or "").strip() or None,
            fee_bps=max(0.0, _as_float(raw.get("fee_bps"), 0.0)),
            slippage_bps=max(0.0, _as_float(raw.get("slippage_bps"), 0.0)),
            transaction_cost_bps=max(0.0, _as_float(raw.get("transaction_cost_bps"), 0.0)),
            max_position_weight=max(0.0, _as_float(raw.get("max_position_weight"), 0.0)),
            min_price=max(0.0, _as_float(raw.get("min_price"), 0.0)),
            min_dollar_volume=max(0.0, _as_float(raw.get("min_dollar_volume"), 0.0)),
            short_borrow_bps_annual=max(0.0, _as_float(raw.get("short_borrow_bps_annual"), 0.0)),
            execution_delay_days=_as_int(raw.get("execution_delay_days"), 1, minimum=0),
            turnover_half_l1=_as_bool(raw.get("turnover_half_l1"), True),
            use_lagged_weights=_as_bool(raw.get("use_lagged_weights"), True),
            allowed_symbols=_normalize_symbols(raw.get("allowed_symbols")),
        )

    def effective_slippage_bps(self) -> float:
        if self.fee_bps <= 0.0 and self.slippage_bps <= 0.0:
            return float(self.transaction_cost_bps)
        return float(self.slippage_bps)


@dataclass(frozen=True)
class OptimalTraderBacktestMatrices:
    date_index: pd.DatetimeIndex
    symbol_order: list[str]
    weights: pd.DataFrame
    returns: pd.DataFrame
    strategy_scores: pd.DataFrame
    strategy_signals: pd.DataFrame


@dataclass(frozen=True)
class OptimalTraderBacktestComputation:
    effective_weights: pd.DataFrame
    turnover: pd.Series
    turnover_cost: pd.Series
    short_borrow_cost: pd.Series
    realized_matrix: pd.DataFrame
    daily_return: pd.Series
    net_daily_return: pd.Series
    equity_curve: pd.Series


@dataclass(frozen=True)
class OptimalTraderBacktestResult:
    trade_frame: pd.DataFrame
    daily_rows: list[dict[str, Any]]
    trades: int
    wins: int
    losses: int
    avg_return: float
    cumulative_return: float
    final_equity: float
    max_drawdown: float
    has_liquidity_data: bool
    start_date: str
    end_date: str


def run_optimal_trader_equity_backtest(
    strategy_df: pd.DataFrame,
    *,
    config: OptimalTraderBacktestConfig | Mapping[str, Any] | None = None,
) -> OptimalTraderBacktestResult:
    """Run optimal_trader-compatible vectorized equity accounting.

    This consumes the strategy dataset shape produced by optimal_trader:
    one row per `(date, symbol)` with `target_weight` and `ret_1` (or
    `asset_return`). The goal is exact accounting parity, not a new strategy
    engine.
    """

    cfg = config if isinstance(config, OptimalTraderBacktestConfig) else OptimalTraderBacktestConfig.from_mapping(config)
    prepared, has_liquidity_data = prepare_optimal_trader_backtest_frame(strategy_df, config=cfg)
    filtered = apply_optimal_trader_backtest_filters(prepared, config=cfg, has_liquidity_data=has_liquidity_data)
    matrices = build_optimal_trader_backtest_matrices(filtered)
    position_state = prepare_optimal_trader_position_state(matrices.weights)
    computation = compute_optimal_trader_backtest(matrices, config=cfg, position_state=position_state)
    trade_frame = build_optimal_trader_trade_frame(matrices, computation)
    daily_rows = build_optimal_trader_daily_rows(matrices, computation)
    trades, wins, losses, avg_return, cumulative_return, final_equity, max_drawdown = summarize_optimal_trader_backtest(
        trade_frame,
        daily_rows,
        computation.equity_curve,
    )
    return OptimalTraderBacktestResult(
        trade_frame=trade_frame,
        daily_rows=daily_rows,
        trades=trades,
        wins=wins,
        losses=losses,
        avg_return=avg_return,
        cumulative_return=cumulative_return,
        final_equity=final_equity,
        max_drawdown=max_drawdown,
        has_liquidity_data=has_liquidity_data,
        start_date=str(cfg.start_date or ""),
        end_date=str(cfg.end_date or ""),
    )


def prepare_optimal_trader_backtest_frame(
    strategy_df: pd.DataFrame,
    *,
    config: OptimalTraderBacktestConfig,
) -> tuple[pd.DataFrame, bool]:
    if strategy_df is None or strategy_df.empty:
        raise ValueError("No strategy dataset rows available for backtest.")
    required = {"date", "symbol", "target_weight"}
    missing = sorted(required.difference(strategy_df.columns))
    if missing:
        raise ValueError(f"optimal_trader equity backtest missing required columns: {missing}")

    out = strategy_df.copy()
    out["date"] = pd.to_datetime(out["date"].astype(str).str[:10], errors="coerce")
    out = out.dropna(subset=["date"]).copy()
    out = _filter_frame_by_date(out, start_date=config.start_date, end_date=config.end_date)
    if out.empty:
        raise ValueError("Strategy dataset produced no rows for backtest.")
    out["symbol"] = out["symbol"].astype(str).str.strip().str.upper()
    out["strategy_signal"] = pd.to_numeric(out.get("strategy_signal"), errors="coerce").fillna(0).astype(int)
    out["target_weight"] = pd.to_numeric(out.get("target_weight"), errors="coerce").fillna(0.0)
    out["strategy_score"] = pd.to_numeric(out.get("strategy_score"), errors="coerce")
    return_col = "asset_return" if "asset_return" in out.columns else "ret_1"
    if return_col not in out.columns:
        raise ValueError("optimal_trader equity backtest requires 'asset_return' or 'ret_1'")
    out["asset_return"] = pd.to_numeric(out.get(return_col), errors="coerce").fillna(0.0)

    close_series = _numeric_series(out, "close", default=float("nan"))
    if close_series.isna().all() and "px__close" in out.columns:
        close_series = pd.to_numeric(out.get("px__close"), errors="coerce")
    out["close"] = close_series
    volume_series = _numeric_series(out, "volume", default=float("nan"))
    if volume_series.isna().all() and "px__volume" in out.columns:
        volume_series = pd.to_numeric(out.get("px__volume"), errors="coerce")
    out["volume"] = volume_series
    if "dollar_vol" in out.columns:
        out["dollar_volume"] = pd.to_numeric(out.get("dollar_vol"), errors="coerce")
    elif "px__dollar_vol" in out.columns:
        out["dollar_volume"] = pd.to_numeric(out.get("px__dollar_vol"), errors="coerce")
    else:
        out["dollar_volume"] = out["close"] * out["volume"]
    return out, bool(out["dollar_volume"].notna().any())


def apply_optimal_trader_backtest_filters(
    strategy_df: pd.DataFrame,
    *,
    config: OptimalTraderBacktestConfig,
    has_liquidity_data: bool,
) -> pd.DataFrame:
    out = strategy_df.copy()
    if config.allowed_symbols:
        out = out[out["symbol"].isin(set(config.allowed_symbols))].copy()
    if config.max_position_weight > 0.0:
        out["target_weight"] = out["target_weight"].clip(
            lower=-float(config.max_position_weight),
            upper=float(config.max_position_weight),
        )
    if config.min_price > 0.0:
        out.loc[out["close"].fillna(0.0) < float(config.min_price), "target_weight"] = 0.0
    if config.min_dollar_volume > 0.0 and has_liquidity_data:
        out.loc[out["dollar_volume"].fillna(0.0) < float(config.min_dollar_volume), "target_weight"] = 0.0
    return out


def build_optimal_trader_backtest_matrices(strategy_df: pd.DataFrame) -> OptimalTraderBacktestMatrices:
    symbol_order = sorted(strategy_df["symbol"].dropna().unique().tolist())
    date_index = pd.DatetimeIndex(sorted(strategy_df["date"].dropna().unique().tolist()))
    weights = (
        strategy_df.pivot_table(index="date", columns="symbol", values="target_weight", aggfunc="last")
        .reindex(index=date_index, columns=symbol_order)
        .fillna(0.0)
    )
    returns = (
        strategy_df.pivot_table(index="date", columns="symbol", values="asset_return", aggfunc="last")
        .reindex(index=date_index, columns=symbol_order)
        .fillna(0.0)
    )
    strategy_scores = (
        strategy_df.pivot_table(index="date", columns="symbol", values="strategy_score", aggfunc="last")
        .reindex(index=date_index, columns=symbol_order)
    )
    strategy_signals = (
        strategy_df.pivot_table(index="date", columns="symbol", values="strategy_signal", aggfunc="last")
        .reindex(index=date_index, columns=symbol_order)
        .fillna(0)
        .astype(int)
    )
    return OptimalTraderBacktestMatrices(
        date_index=date_index,
        symbol_order=symbol_order,
        weights=weights,
        returns=returns,
        strategy_scores=strategy_scores,
        strategy_signals=strategy_signals,
    )


def prepare_optimal_trader_position_state(positions: pd.DataFrame) -> dict[str, pd.DataFrame | pd.Series]:
    positions = positions.astype(float).copy()
    gross_weight_basis = positions.abs().sum(axis=1).replace(0.0, np.nan)
    weights = positions.div(gross_weight_basis, axis=0).fillna(0.0)
    lagged_weights = weights.shift(1).fillna(0.0)
    long_weights = lagged_weights.clip(lower=0.0)
    short_weights = -lagged_weights.clip(upper=0.0)
    turnover = 0.5 * weights.diff().abs().sum(axis=1).fillna(weights.abs().sum(axis=1))
    return {
        "positions": positions,
        "weights": weights,
        "lagged_weights": lagged_weights,
        "long_weights": long_weights,
        "short_weights": short_weights,
        "turnover": turnover,
    }


def compute_optimal_trader_backtest(
    matrices: OptimalTraderBacktestMatrices,
    *,
    config: OptimalTraderBacktestConfig,
    position_state: dict[str, pd.DataFrame | pd.Series] | None = None,
) -> OptimalTraderBacktestComputation:
    if position_state is None:
        position_state = prepare_optimal_trader_position_state(matrices.weights)
    effective_weights = _require_frame(position_state["weights"], "weights").copy()
    turnover = _require_series(position_state["turnover"], "turnover").copy()
    if config.use_lagged_weights:
        effective_weights = effective_weights.shift(config.execution_delay_days).fillna(0.0)
        turnover = effective_weights.diff().abs().fillna(effective_weights.abs()).sum(axis=1)
    if (effective_weights.abs().sum(axis=1) <= 0).all():
        raise ValueError("Strategy dataset produced no active portfolio rows.")
    if config.turnover_half_l1:
        turnover = turnover * 0.5
    turnover_cost = turnover * ((float(config.fee_bps) + float(config.effective_slippage_bps())) / BPS_TO_DECIMAL)
    short_borrow_cost = effective_weights.clip(upper=0.0).abs().sum(axis=1) * (
        float(config.short_borrow_bps_annual) / BPS_TO_DECIMAL / TRADING_DAYS_PER_YEAR
    )
    realized_matrix = effective_weights * matrices.returns
    daily_return = realized_matrix.sum(axis=1)
    net_daily_return = daily_return - turnover_cost - short_borrow_cost
    equity_curve = (1.0 + net_daily_return).cumprod()
    return OptimalTraderBacktestComputation(
        effective_weights=effective_weights,
        turnover=turnover,
        turnover_cost=turnover_cost,
        short_borrow_cost=short_borrow_cost,
        realized_matrix=realized_matrix,
        daily_return=daily_return,
        net_daily_return=net_daily_return,
        equity_curve=equity_curve,
    )


def build_optimal_trader_trade_frame(
    matrices: OptimalTraderBacktestMatrices,
    computation: OptimalTraderBacktestComputation,
) -> pd.DataFrame:
    trade_frame = pd.DataFrame(
        {
            "target_weight": matrices.weights.stack(future_stack=True),
            "effective_weight": computation.effective_weights.stack(future_stack=True),
            "asset_return": matrices.returns.stack(future_stack=True),
            "strategy_score": matrices.strategy_scores.stack(future_stack=True),
            "strategy_signal": matrices.strategy_signals.stack(future_stack=True),
            "realized_return": computation.realized_matrix.stack(future_stack=True),
        }
    ).reset_index()
    trade_frame = trade_frame.rename(columns={"level_0": "date", "level_1": "symbol"})
    trade_frame["gross_exposure"] = trade_frame["effective_weight"].abs()
    trade_frame["turnover"] = trade_frame["date"].map(computation.turnover.to_dict())
    trade_frame["turnover_cost"] = trade_frame["date"].map(computation.turnover_cost.to_dict())
    trade_frame["short_borrow_cost"] = trade_frame["date"].map(computation.short_borrow_cost.to_dict())
    trade_frame = trade_frame[
        (trade_frame["target_weight"].abs() > ACTIVE_WEIGHT_EPSILON)
        | (trade_frame["effective_weight"].abs() > ACTIVE_WEIGHT_EPSILON)
    ].copy()
    trade_frame["date"] = pd.to_datetime(trade_frame["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return trade_frame.sort_values(["date", "symbol"]).reset_index(drop=True)


def build_optimal_trader_daily_rows(
    matrices: OptimalTraderBacktestMatrices,
    computation: OptimalTraderBacktestComputation,
) -> list[dict[str, Any]]:
    return [
        {
            "date": str(date_value.date()),
            "positions": int((computation.effective_weights.loc[date_value].abs() > ACTIVE_WEIGHT_EPSILON).sum()),
            "gross_exposure": round(float(computation.effective_weights.loc[date_value].abs().sum()), SUMMARY_DECIMALS),
            "turnover": round(float(computation.turnover.loc[date_value]), SUMMARY_DECIMALS),
            "turnover_cost": round(float(computation.turnover_cost.loc[date_value]), SUMMARY_DECIMALS),
            "short_borrow_cost": round(float(computation.short_borrow_cost.loc[date_value]), SUMMARY_DECIMALS),
            "daily_return": round(float(computation.daily_return.loc[date_value]), SUMMARY_DECIMALS),
            "net_daily_return": round(float(computation.net_daily_return.loc[date_value]), SUMMARY_DECIMALS),
            "equity": round(float(computation.equity_curve.loc[date_value]), SUMMARY_DECIMALS),
        }
        for date_value in matrices.date_index
    ]


def summarize_optimal_trader_backtest(
    trade_frame: pd.DataFrame,
    daily_rows: list[dict[str, Any]],
    equity_curve: pd.Series,
) -> tuple[int, int, int, float, float, float, float]:
    realized_returns = pd.to_numeric(trade_frame.get("realized_return"), errors="coerce").fillna(0.0).tolist()
    net_daily_returns = [float(row["net_daily_return"]) for row in daily_rows]
    trades = int(len(trade_frame))
    wins = sum(1 for value in realized_returns if value > 0)
    losses = sum(1 for value in realized_returns if value < 0)
    avg_return = (sum(net_daily_returns) / float(len(net_daily_returns))) if net_daily_returns else 0.0
    cumulative_return = float(equity_curve.iloc[-1] - 1.0)
    rolling_max = equity_curve.cummax()
    drawdown_series = (equity_curve / rolling_max) - 1.0
    max_drawdown = float(drawdown_series.min()) if not drawdown_series.empty else 0.0
    final_equity = float(equity_curve.iloc[-1])
    return trades, wins, losses, avg_return, cumulative_return, final_equity, max_drawdown


def _filter_frame_by_date(
    frame: pd.DataFrame,
    *,
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    out = frame
    if start_date:
        out = out[out["date"] >= pd.Timestamp(start_date)]
    if end_date:
        out = out[out["date"] <= pd.Timestamp(end_date)]
    return out.copy()


def _numeric_series(frame: pd.DataFrame, column: str, *, default: float) -> pd.Series:
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce").fillna(default)
    return pd.Series(default, index=frame.index, dtype=float)


def _as_float(value: Any, default: float) -> float:
    try:
        if value in (None, ""):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _as_int(value: Any, default: int, *, minimum: int | None = None) -> int:
    try:
        parsed = int(value) if value not in (None, "") else int(default)
    except Exception:
        parsed = int(default)
    if minimum is not None:
        parsed = max(int(minimum), parsed)
    return parsed


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _normalize_symbols(values: Any) -> tuple[str, ...]:
    if values in (None, ""):
        return ()
    raw_values: list[Any]
    if isinstance(values, str):
        raw_values = values.split(",")
    elif isinstance(values, (list, tuple, set)):
        raw_values = list(values)
    else:
        raw_values = [values]
    out: list[str] = []
    for value in raw_values:
        normalized = str(value or "").strip().upper()
        if normalized and normalized not in out:
            out.append(normalized)
    return tuple(out)


def _require_frame(value: pd.DataFrame | pd.Series, name: str) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise TypeError(f"position_state['{name}'] must be a DataFrame")
    return value


def _require_series(value: pd.DataFrame | pd.Series, name: str) -> pd.Series:
    if not isinstance(value, pd.Series):
        raise TypeError(f"position_state['{name}'] must be a Series")
    return value
