from __future__ import annotations

import numpy as np
import pandas as pd


def summarize_curve(returns: pd.Series, years: list[int], mode: str) -> dict[str, object]:
    returns = pd.Series(returns).fillna(0.0)
    equity = (1.0 + returns).cumprod()
    total_return_pct = float((equity.iloc[-1] - 1.0) * 100.0) if len(equity) else np.nan
    volatility = returns.std(ddof=0)
    sharpe = (
        float((returns.mean() / volatility) * np.sqrt(252.0))
        if len(returns) and volatility > 1e-12
        else np.nan
    )
    max_drawdown_pct = (
        float((((equity / equity.cummax()) - 1.0).min()) * 100.0)
        if len(equity)
        else np.nan
    )
    yearly_rows = []
    for year in years:
        yearly_returns = returns.loc[
            (returns.index >= pd.Timestamp(f"{year}-01-01"))
            & (returns.index <= pd.Timestamp(f"{year}-12-31"))
        ]
        yearly_equity = (1.0 + yearly_returns).cumprod()
        yearly_volatility = yearly_returns.std(ddof=0)
        yearly_rows.append(
            {
                "mode": str(mode),
                "test_year": int(year),
                "total_return_pct": (
                    float((yearly_equity.iloc[-1] - 1.0) * 100.0)
                    if len(yearly_equity)
                    else np.nan
                ),
                "sharpe": (
                    float((yearly_returns.mean() / yearly_volatility) * np.sqrt(252.0))
                    if len(yearly_returns) and yearly_volatility > 1e-12
                    else np.nan
                ),
                "max_drawdown_pct": (
                    float((((yearly_equity / yearly_equity.cummax()) - 1.0).min()) * 100.0)
                    if len(yearly_equity)
                    else np.nan
                ),
            },
        )
    return {
        "total_return_pct": total_return_pct,
        "sharpe": sharpe,
        "max_drawdown_pct": max_drawdown_pct,
        "equity_curve": equity,
        "yearly_df": pd.DataFrame(yearly_rows),
    }


def run_top_k_long_only_score_rule(
    *,
    panel: pd.DataFrame,
    score_col: str,
    component_cols: list[str],
    component_threshold: float,
    price_col: str,
    top_k: int,
    rebalance_freq: str | None = None,
) -> dict[str, pd.DataFrame]:
    _ = rebalance_freq
    return _run_capacity_limited_long_only_rule(
        panel=panel,
        score_col=score_col,
        component_cols=component_cols,
        component_threshold=component_threshold,
        price_col=price_col,
        top_k=int(top_k),
    )


def run_top_k_long_short_score_rule(
    *,
    panel: pd.DataFrame,
    long_score_col: str,
    short_score_col: str,
    long_component_cols: list[str],
    short_component_cols: list[str],
    component_threshold: float,
    price_col: str,
    top_k: int,
    rebalance_freq: str | None = None,
) -> dict[str, pd.DataFrame]:
    _ = rebalance_freq
    return _run_capacity_limited_long_short_rule(
        panel=panel,
        long_score_col=long_score_col,
        short_score_col=short_score_col,
        long_component_cols=long_component_cols,
        short_component_cols=short_component_cols,
        component_threshold=component_threshold,
        price_col=price_col,
        top_k=int(top_k),
    )


def run_top_k_momentum_baseline(
    *,
    panel: pd.DataFrame,
    price_col: str,
    top_k: int,
    lookback_days: int = 21,
    rebalance_freq: str | None = None,
) -> dict[str, pd.DataFrame]:
    _ = rebalance_freq
    symbols = sorted(panel.index.get_level_values("symbol").unique())
    close = _pivot_rule_panel(panel, price_col, symbols=symbols).replace([np.inf, -np.inf], np.nan).ffill()
    score = close.pct_change(int(lookback_days), fill_method=None).shift(1).replace([np.inf, -np.inf], np.nan)
    common_dates = score.index.intersection(close.index)
    score = score.loc[common_dates]
    close = close.loc[common_dates].fillna(0.0)
    held_side_by_idx: dict[int, int] = {}
    symbol_to_idx = {symbol: idx for idx, symbol in enumerate(symbols)}
    positions = pd.DataFrame(0, index=common_dates, columns=symbols, dtype=int)
    for dt in common_dates:
        score_t = score.loc[dt]
        price_ok_t = close.loc[dt].gt(0.0)
        next_held = {}
        for idx, side in sorted(held_side_by_idx.items()):
            if not bool(price_ok_t.iloc[idx]) or not np.isfinite(score_t.iloc[idx]):
                continue
            current_score = float(score_t.iloc[idx])
            if side > 0 and current_score <= 0.0:
                continue
            if side < 0 and current_score >= 0.0:
                continue
            next_held[idx] = side
        held_side_by_idx = next_held
        capacity = max(0, int(top_k))
        slots_left = max(0, capacity - len(held_side_by_idx))
        if slots_left:
            candidates = []
            for symbol in symbols:
                idx = symbol_to_idx[str(symbol)]
                if idx in held_side_by_idx or not bool(price_ok_t.iloc[idx]) or not np.isfinite(score_t.iloc[idx]):
                    continue
                value = float(score_t.iloc[idx])
                if value > 0.0:
                    candidates.append((abs(value), str(symbol), idx, 1))
                elif value < 0.0:
                    candidates.append((abs(value), str(symbol), idx, -1))
            for _score, _symbol, idx, side in sorted(candidates, key=lambda row: (row[0], row[1]), reverse=True):
                held_side_by_idx[idx] = int(side)
                if len(held_side_by_idx) >= capacity:
                    break
        for idx, side in sorted(held_side_by_idx.items()):
            positions.loc[dt, symbols[idx]] = int(side)
    return {"positions": positions, "score": score, "close": close}


def prepare_backtest_position_state(positions: pd.DataFrame) -> dict[str, pd.DataFrame | pd.Series]:
    positions = positions.astype(float).copy()
    gross_weight_basis = positions.abs().sum(axis=1).replace(0.0, np.nan)
    weights = positions.div(gross_weight_basis, axis=0).fillna(0.0)
    lagged_weights = weights.shift(1).fillna(0.0)
    turnover = 0.5 * weights.diff().abs().sum(axis=1).fillna(weights.abs().sum(axis=1))
    return {
        "positions": positions,
        "weights": weights,
        "lagged_weights": lagged_weights,
        "long_weights": lagged_weights.clip(lower=0.0),
        "short_weights": -lagged_weights.clip(upper=0.0),
        "turnover": turnover,
    }


def backtest_positions_with_directional_asset_returns(
    positions: pd.DataFrame,
    long_asset_returns: pd.DataFrame,
    *,
    short_asset_returns: pd.DataFrame | None = None,
    initial_balance: float = 100000.0,
    fee_bps: float = 5.0,
    slippage_bps: float = 5.0,
    position_state: dict[str, pd.DataFrame | pd.Series] | None = None,
) -> dict[str, pd.DataFrame | pd.Series]:
    if position_state is None:
        position_state = prepare_backtest_position_state(positions)
    weights = position_state["weights"]
    if not isinstance(weights, pd.DataFrame):
        raise TypeError("position_state['weights'] must be a DataFrame")
    aligned_index = weights.index
    aligned_columns = weights.columns
    long_asset_returns = long_asset_returns.reindex(index=aligned_index, columns=aligned_columns).fillna(0.0)
    if short_asset_returns is None:
        short_asset_returns = -long_asset_returns
    short_asset_returns = short_asset_returns.reindex(index=aligned_index, columns=aligned_columns).fillna(0.0)
    long_weights = position_state["long_weights"]
    short_weights = position_state["short_weights"]
    turnover = position_state["turnover"]
    if not isinstance(long_weights, pd.DataFrame) or not isinstance(short_weights, pd.DataFrame) or not isinstance(turnover, pd.Series):
        raise TypeError("position_state has invalid weight or turnover values")
    gross_returns = (long_weights * long_asset_returns + short_weights * short_asset_returns).sum(axis=1)
    net_returns = gross_returns - turnover * ((float(fee_bps) + float(slippage_bps)) / 10000.0)
    equity = (1.0 + net_returns).cumprod() * float(initial_balance)
    return {
        "weights": weights,
        "turnover": turnover,
        "returns": net_returns,
        "equity": equity,
    }


def resolve_component_cols(score_col: str) -> list[str]:
    mapping = {
        "prob_buy": ["prob_buy"],
        "prob_short": ["prob_short"],
        "buy_score_mean_raw3": ["prob_buy", "pred_rf_reg", "ae_familiarity"],
        "buy_score_mean_raw_pct6": [
            "prob_buy",
            "pred_rf_reg",
            "ae_familiarity",
            "prob_buy_pct",
            "pred_rf_reg_pct",
            "ae_familiarity_pct",
        ],
        "buy_score_pct_mean": ["prob_buy_pct", "pred_rf_reg_pct", "ae_familiarity_pct"],
        "buy_score_pct_product": ["prob_buy_pct", "pred_rf_reg_pct", "ae_familiarity_pct"],
        "buy_score_raw": ["prob_buy", "pred_rf_reg", "ae_familiarity"],
        "buy_score": ["prob_buy", "pred_rf_reg", "ae_familiarity"],
        "short_score_mean_raw3": ["prob_short", "pred_rf_reg", "ae_familiarity"],
        "short_score_mean_raw_pct6": [
            "prob_short",
            "pred_rf_reg",
            "ae_familiarity",
            "prob_short_pct",
            "pred_rf_reg_pct",
            "ae_familiarity_pct",
        ],
        "short_score_pct_mean": ["prob_short_pct", "pred_rf_reg_pct", "ae_familiarity_pct"],
        "short_score_pct_product": ["prob_short_pct", "pred_rf_reg_pct", "ae_familiarity_pct"],
        "short_score_raw": ["prob_short", "pred_rf_reg", "ae_familiarity"],
        "short_score": ["prob_short", "pred_rf_reg", "ae_familiarity"],
        "__momentum_21d__": [],
    }
    return list(mapping.get(str(score_col), [str(score_col)]))


def resolve_short_score_col(score_col: str) -> str:
    mapping = {
        "prob_buy": "prob_short",
        "buy_score_mean_raw3": "short_score_mean_raw3",
        "buy_score_mean_raw_pct6": "short_score_mean_raw_pct6",
        "buy_score_pct_mean": "short_score_pct_mean",
        "buy_score_pct_product": "short_score_pct_product",
        "buy_score_raw": "short_score_raw",
        "buy_score": "short_score",
    }
    key = str(score_col)
    if key in mapping:
        return str(mapping[key])
    if key.startswith("buy_"):
        return "short_" + key[len("buy_") :]
    raise KeyError(f"No short-score mapping configured for: {score_col}")


def _run_capacity_limited_long_only_rule(
    *,
    panel: pd.DataFrame,
    score_col: str,
    component_cols: list[str],
    component_threshold: float,
    price_col: str,
    top_k: int | None,
) -> dict[str, pd.DataFrame]:
    inputs = _prepare_capacity_rule_inputs(panel, score_col, component_cols, price_col)
    symbols = inputs["symbols"]
    score = inputs["score"]
    prob_buy = inputs["prob_buy"]
    prob_short = inputs["prob_short"]
    close = inputs["close"]
    entry_ok = _build_entry_ok_matrix(inputs, component_threshold)
    held_idx: set[int] = set()
    symbol_to_idx = {symbol: idx for idx, symbol in enumerate(symbols)}
    positions = pd.DataFrame(0, index=inputs["common_dates"], columns=symbols, dtype=int)
    for dt in inputs["common_dates"]:
        classifier_short = (prob_short.loc[dt] > prob_buy.loc[dt]).fillna(False)
        held_idx -= {idx for idx in held_idx if bool(classifier_short.iloc[idx])}
        slots_left = None if top_k is None else max(0, int(top_k) - len(held_idx))
        if slots_left != 0:
            candidate_mask = entry_ok.loc[dt] & close.loc[dt].gt(0.0) & (~classifier_short)
            ranked = score.loc[dt][candidate_mask].sort_values(ascending=False, kind="stable")
            enter_idx = []
            for symbol in ranked.index:
                idx = symbol_to_idx[str(symbol)]
                if idx in held_idx:
                    continue
                enter_idx.append(idx)
                if slots_left is not None and len(enter_idx) >= slots_left:
                    break
            held_idx |= set(enter_idx)
        if held_idx:
            positions.loc[dt, [symbols[idx] for idx in sorted(held_idx)]] = 1
    return {"positions": positions, "score": score, "close": close}


def _run_capacity_limited_long_short_rule(
    *,
    panel: pd.DataFrame,
    long_score_col: str,
    short_score_col: str,
    long_component_cols: list[str],
    short_component_cols: list[str],
    component_threshold: float,
    price_col: str,
    top_k: int | None,
) -> dict[str, pd.DataFrame]:
    long_inputs = _prepare_capacity_rule_inputs(panel, long_score_col, long_component_cols, price_col)
    short_inputs = _prepare_capacity_rule_inputs(panel, short_score_col, short_component_cols, price_col)
    symbols = long_inputs["symbols"]
    close = long_inputs["close"]
    long_score = long_inputs["score"]
    short_score = short_inputs["score"]
    prob_buy = long_inputs["prob_buy"]
    prob_short = long_inputs["prob_short"]
    long_entry_ok = _build_entry_ok_matrix(long_inputs, component_threshold)
    short_entry_ok = _build_entry_ok_matrix(short_inputs, component_threshold)
    held_side_by_idx: dict[int, int] = {}
    symbol_to_idx = {symbol: idx for idx, symbol in enumerate(symbols)}
    positions = pd.DataFrame(0, index=long_inputs["common_dates"], columns=symbols, dtype=int)
    for dt in long_inputs["common_dates"]:
        price_ok_t = close.loc[dt].gt(0.0)
        next_held = {}
        for idx, side in sorted(held_side_by_idx.items()):
            if side > 0 and bool(prob_short.loc[dt].iloc[idx] > prob_buy.loc[dt].iloc[idx]):
                continue
            if side < 0 and bool(prob_buy.loc[dt].iloc[idx] > prob_short.loc[dt].iloc[idx]):
                continue
            next_held[idx] = side
        held_side_by_idx = next_held
        capacity = None if top_k is None else max(0, int(top_k))
        slots_left = None if capacity is None else max(0, capacity - len(held_side_by_idx))
        if slots_left != 0:
            candidates = []
            for symbol in symbols:
                idx = symbol_to_idx[str(symbol)]
                if idx in held_side_by_idx or not bool(price_ok_t.iloc[idx]):
                    continue
                long_ok = bool(long_entry_ok.loc[dt].iloc[idx]) and np.isfinite(long_score.loc[dt].iloc[idx])
                short_ok = bool(short_entry_ok.loc[dt].iloc[idx]) and np.isfinite(short_score.loc[dt].iloc[idx])
                if not long_ok and not short_ok:
                    continue
                if long_ok and short_ok:
                    long_value = float(long_score.loc[dt].iloc[idx])
                    short_value = float(short_score.loc[dt].iloc[idx])
                    best_side, best_score = (1, long_value) if long_value >= short_value else (-1, short_value)
                elif long_ok:
                    best_side, best_score = 1, float(long_score.loc[dt].iloc[idx])
                else:
                    best_side, best_score = -1, float(short_score.loc[dt].iloc[idx])
                candidates.append((best_score, str(symbol), idx, best_side))
            for _score, _symbol, idx, side in sorted(candidates, key=lambda row: (row[0], row[1]), reverse=True):
                held_side_by_idx[idx] = int(side)
                if slots_left is not None and len(held_side_by_idx) >= int(capacity):
                    break
        for idx, side in sorted(held_side_by_idx.items()):
            positions.loc[dt, symbols[idx]] = int(side)
    return {"positions": positions, "long_score": long_score, "short_score": short_score, "close": close}


def _prepare_capacity_rule_inputs(
    panel: pd.DataFrame,
    score_col: str,
    component_cols: list[str],
    price_col: str,
) -> dict[str, object]:
    if panel.index.has_duplicates:
        panel = panel[~panel.index.duplicated(keep="last")]
    symbols = sorted(panel.index.get_level_values("symbol").unique())
    score = _pivot_rule_panel(panel, score_col, symbols=symbols).shift(1)
    prob_buy = _pivot_rule_panel(panel, "prob_buy", symbols=symbols).shift(1)
    prob_short = _pivot_rule_panel(panel, "prob_short", symbols=symbols).shift(1)
    close = _pivot_rule_panel(panel, price_col, symbols=symbols)
    common_dates = score.index.intersection(prob_buy.index).intersection(prob_short.index).intersection(close.index)
    component_frames = {}
    for column in component_cols:
        component_frames[str(column)] = (
            _pivot_rule_panel(panel, column, symbols=symbols)
            .shift(1)
            .reindex(index=common_dates, columns=symbols)
            .replace([np.inf, -np.inf], np.nan)
        )
    return {
        "symbols": symbols,
        "common_dates": common_dates,
        "score": score.loc[common_dates].replace([np.inf, -np.inf], np.nan),
        "prob_buy": prob_buy.loc[common_dates].replace([np.inf, -np.inf], np.nan),
        "prob_short": prob_short.loc[common_dates].replace([np.inf, -np.inf], np.nan),
        "close": close.loc[common_dates].replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0),
        "component_cols": [str(column) for column in component_cols],
        "component_frames": component_frames,
    }


def _build_entry_ok_matrix(inputs: dict[str, object], component_threshold: float) -> pd.DataFrame:
    score = inputs["score"]
    close = inputs["close"]
    if not isinstance(score, pd.DataFrame) or not isinstance(close, pd.DataFrame):
        raise TypeError("capacity inputs are invalid")
    entry_ok = (score.notna() & np.isfinite(score) & close.gt(0.0)).fillna(False)
    for column in inputs["component_cols"]:
        component = inputs["component_frames"][column]
        component_valid = component.notna() & np.isfinite(component)
        entry_ok &= (component.gt(float(component_threshold)) & component_valid).fillna(False)
    return entry_ok


def _pivot_rule_panel(panel: pd.DataFrame, column: str, *, symbols: list[str]) -> pd.DataFrame:
    work = panel[[column]].reset_index()
    if work.duplicated(subset=["date", "symbol"]).any():
        work = work.sort_values(["date", "symbol"]).groupby(["date", "symbol"], as_index=False, sort=False).last()
    return work.pivot(index="date", columns="symbol", values=column).reindex(columns=symbols).sort_index()
