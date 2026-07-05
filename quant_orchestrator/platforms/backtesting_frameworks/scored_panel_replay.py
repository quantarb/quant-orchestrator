from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from quant_orchestrator.platforms.backtesting_frameworks.optimal_trader.artifact_replay import (
    TradingAppRuleReplay,
    _merge_action_tape_executions,
    _nan_float,
    action_tape_to_trade_windows,
    discrete_backtest_with_executions,
    summarize_returns,
)
from quant_orchestrator.platforms.backtesting_frameworks.strategy_artifacts import (
    StrategyArtifactBundle,
    write_strategy_artifacts,
)


@dataclass(frozen=True)
class ScoredPanelTopKReplayConfig:
    score_col: str = "prob_buy"
    threshold: float = 0.50
    top_k: int = 20
    initial_balance: float = 100_000.0
    fee_bps: float = 5.0
    slippage_bps: float = 5.0
    strategy_name: str = "scored_panel_top_k"
    output_dir: Path | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScoredPanelTopKReplayResult:
    scored_panel: pd.DataFrame
    rule_replay: TradingAppRuleReplay
    summary: dict[str, Any]


def replay_scored_panel_top_k(
    scored_panel: pd.DataFrame,
    *,
    config: ScoredPanelTopKReplayConfig | None = None,
) -> ScoredPanelTopKReplayResult:
    cfg = config or ScoredPanelTopKReplayConfig()
    panel = _normalize_scored_panel(scored_panel)
    replay = _replay_top_k(
        panel,
        score_col=str(cfg.score_col),
        threshold=float(cfg.threshold),
        top_k=int(cfg.top_k),
        initial_balance=float(cfg.initial_balance),
        fee_bps=float(cfg.fee_bps),
        slippage_bps=float(cfg.slippage_bps),
    )
    summary = {
        "strategy_name": str(cfg.strategy_name),
        "score_col": str(cfg.score_col),
        "threshold": float(cfg.threshold),
        "top_k": int(cfg.top_k),
        "scored_rows": int(len(panel)),
        "scored_symbols": int(panel.index.get_level_values("symbol").nunique()) if not panel.empty else 0,
        "trade_windows": int(len(replay.trade_windows)),
        "rule_meta": replay.meta,
        "performance": summarize_returns(replay.returns, float(cfg.initial_balance)) if len(replay.returns) else {},
        "metadata": dict(cfg.metadata),
    }
    result = ScoredPanelTopKReplayResult(scored_panel=panel, rule_replay=replay, summary=summary)
    if cfg.output_dir is not None:
        write_scored_panel_top_k_outputs(result, cfg.output_dir)
    return result


def write_scored_panel_top_k_outputs(
    result: ScoredPanelTopKReplayResult,
    output_dir: str | Path,
) -> dict[str, Path]:
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {
        "equity_curve": out_dir / "equity_curve.csv",
        "returns": out_dir / "returns.csv",
        "cash": out_dir / "cash.csv",
        "equity_executions": out_dir / "equity_executions.csv",
    }
    result.rule_replay.equity.rename("equity").to_csv(paths["equity_curve"])
    result.rule_replay.returns.rename("returns").to_csv(paths["returns"])
    result.rule_replay.cash.rename("cash").to_csv(paths["cash"])
    result.rule_replay.executions.to_csv(paths["equity_executions"], index=False)
    standard_paths = write_strategy_artifacts(
        StrategyArtifactBundle(
            scored_panel=result.scored_panel,
            action_tape=result.rule_replay.action_tape,
            trade_windows=result.rule_replay.trade_windows,
            summary=result.summary,
            strategy_name=str(result.summary.get("strategy_name") or "scored_panel_top_k"),
        ),
        out_dir,
        extra_paths=paths,
    )
    paths.update({f"standard_{key}": value for key, value in standard_paths.items()})
    return paths


def _normalize_scored_panel(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["close"])
    out = frame.copy()
    if isinstance(out.index, pd.MultiIndex) and {"date", "symbol"}.issubset(set(out.index.names)):
        out = out.reorder_levels(["date", "symbol"]).sort_index()
    else:
        if "date" not in out.columns or "symbol" not in out.columns:
            raise ValueError("scored_panel requires a MultiIndex(date, symbol) or date/symbol columns")
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
        out["symbol"] = out["symbol"].astype(str).str.strip().str.upper()
        out = out.dropna(subset=["date", "symbol"]).set_index(["date", "symbol"]).sort_index()
    dates = pd.to_datetime(out.index.get_level_values("date"), errors="coerce").normalize()
    symbols = out.index.get_level_values("symbol").astype(str).str.strip().str.upper()
    out.index = pd.MultiIndex.from_arrays([dates, symbols], names=["date", "symbol"])
    if "close" not in out.columns:
        raise ValueError("scored_panel missing required 'close' column")
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    return out.sort_index()


def _replay_top_k(
    panel: pd.DataFrame,
    *,
    score_col: str,
    threshold: float,
    top_k: int,
    initial_balance: float,
    fee_bps: float,
    slippage_bps: float,
) -> TradingAppRuleReplay:
    if score_col not in panel.columns:
        raise ValueError(f"scored_panel missing score_col {score_col!r}")
    symbols = sorted(panel.index.get_level_values("symbol").unique())
    close = _pivot(panel, "close", symbols).replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    score = _pivot(panel, score_col, symbols).shift(1).reindex(index=close.index, columns=symbols)
    common_dates = close.index.intersection(score.index)
    close = close.loc[common_dates]
    score = score.loc[common_dates].replace([np.inf, -np.inf], np.nan)
    action_type = np.zeros((len(common_dates), len(symbols)), dtype=int)
    positions = pd.DataFrame(0, index=common_dates, columns=symbols, dtype=int)
    symbol_to_idx = {symbol: idx for idx, symbol in enumerate(symbols)}
    held: set[int] = set()
    for row_idx, dt in enumerate(common_dates):
        price_ok = close.loc[dt].gt(0.0)
        eligible = (score.loc[dt].gt(float(threshold)) & score.loc[dt].notna() & price_ok).fillna(False)
        target_symbols = list(score.loc[dt][eligible].sort_values(ascending=False, kind="stable").head(max(int(top_k), 0)).index)
        target = {symbol_to_idx[str(symbol)] for symbol in target_symbols}
        exits = sorted(idx for idx in held if idx not in target or not bool(price_ok.iloc[idx]))
        if exits:
            action_type[row_idx, exits] = 2
            held -= set(exits)
        entries = sorted(target - held, key=lambda idx: target_symbols.index(symbols[idx]))
        if entries:
            action_type[row_idx, entries] = 1
            held |= set(entries)
        if held:
            positions.iloc[row_idx, sorted(held)] = 1
    action_tape = _build_action_tape(
        action_type=action_type,
        close=close,
        score=score,
        symbols=symbols,
        score_col=score_col,
        top_k=top_k,
        threshold=threshold,
    )
    equity, returns, cash, details, executions = discrete_backtest_with_executions(
        action_type=action_type,
        close=close,
        symbol_order=symbols,
        initial_balance=float(initial_balance),
        fee_bps=float(fee_bps),
        slippage_bps=float(slippage_bps),
    )
    action_tape = _merge_action_tape_executions(action_tape, executions)
    details.update(
        {
            "top_k": int(top_k),
            "threshold": float(threshold),
            "score_col": str(score_col),
            "avg_positions": float((positions > 0).sum(axis=1).mean()) if len(positions) else np.nan,
            "median_positions": float((positions > 0).sum(axis=1).median()) if len(positions) else np.nan,
        }
    )
    return TradingAppRuleReplay(
        equity=equity,
        returns=returns,
        cash=cash,
        action_tape=action_tape,
        executions=executions,
        positions=positions,
        trade_windows=action_tape_to_trade_windows(action_tape, prices=close),
        meta=details,
    )


def _pivot(panel: pd.DataFrame, column: str, symbols: list[str]) -> pd.DataFrame:
    work = panel[[column]].reset_index()
    if work.duplicated(["date", "symbol"]).any():
        work = work.sort_values(["date", "symbol"]).groupby(["date", "symbol"], as_index=False, sort=False).last()
    return work.pivot(index="date", columns="symbol", values=column).reindex(columns=symbols).sort_index()


def _build_action_tape(
    *,
    action_type: np.ndarray,
    close: pd.DataFrame,
    score: pd.DataFrame,
    symbols: list[str],
    score_col: str,
    top_k: int,
    threshold: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row_idx, dt in enumerate(close.index):
        actions = np.asarray(action_type[row_idx], dtype=int)
        for col_idx in np.where(actions != 0)[0]:
            symbol = symbols[col_idx]
            action = "buy" if actions[col_idx] == 1 else "sell"
            rows.append(
                {
                    "date": pd.Timestamp(dt).normalize(),
                    "symbol": symbol,
                    "action": action,
                    "side": "long",
                    "target_position": 1 if action == "buy" else 0,
                    "price": float(close.iloc[row_idx, col_idx]),
                    "score": _nan_float(score.iloc[row_idx, col_idx]),
                    "score_col": str(score_col),
                    "top_k": int(top_k),
                    "threshold": float(threshold),
                    "reason": "entry_scored_panel_top_k" if action == "buy" else "exit_scored_panel_top_k_or_invalid",
                }
            )
    return pd.DataFrame(rows)

