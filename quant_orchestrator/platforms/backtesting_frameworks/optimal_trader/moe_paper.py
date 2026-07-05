from __future__ import annotations

import json
import pickle
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import numpy as np
import pandas as pd

from quant_orchestrator.platforms.backtesting_frameworks.optimal_trader.artifact_replay import (
    TradingAppRuleReplay,
    _merge_action_tape_executions,
    _nan_float,
    action_tape_to_trade_windows,
    discrete_backtest_with_executions,
    replay_option_portfolio_from_selected_paths,
    summarize_option_trade_returns,
    summarize_returns,
)
from quant_orchestrator.platforms.backtesting_frameworks.strategy_artifacts import (
    StrategyArtifactBundle,
    write_strategy_artifacts,
)


@dataclass(frozen=True)
class MoePaperReplayConfig:
    score_artifact: Path
    model_artifact_dir: Path | None = None
    output_dir: Path | None = None
    feature_panel_path: Path | None = None
    scored_panel_path: Path | None = None
    backtest_start: str = "2021-01-01"
    end_date: str = "2026-06-24"
    top_k: int = 40
    threshold: float = 0.50
    initial_balance: float = 100_000.0
    fee_bps: float = 5.0
    slippage_bps: float = 5.0
    run_fmp_synthetic_options: bool = False
    option_workers: int = 1


@dataclass(frozen=True)
class MoePaperReplayResult:
    latest_ranked: pd.DataFrame
    feature_panel: pd.DataFrame
    scored_panel: pd.DataFrame
    rule_replay: TradingAppRuleReplay
    summary: dict[str, Any]
    option_execution: Any | None = None
    option_portfolio: Any | None = None


def install_moe_pickle_compat_modules() -> None:
    """Provide enough old optimal_trader class names to unpickle MoE artifacts."""

    def module(name: str) -> types.ModuleType:
        mod = sys.modules.get(name)
        if mod is None:
            mod = types.ModuleType(name)
            sys.modules[name] = mod
        return mod

    for name in (
        "ml",
        "ml.frameworks",
        "ml.frameworks.sklearn",
        "ml.frameworks.sklearn.cuml_classifier",
        "ml.frameworks.sklearn.classifier",
    ):
        module(name)

    class CumlRFClassifier:
        def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
            model = getattr(self, "model", None)
            if model is None:
                raise RuntimeError("Unpickled CumlRFClassifier has no model")
            return np.asarray(model.predict_proba(frame), dtype=float)

        def predict(self, frame: pd.DataFrame, *, feature_cols: list[str] | None = None) -> np.ndarray:
            cols = feature_cols or getattr(self, "_used_features", None)
            x = frame[cols] if cols else frame
            model = getattr(self, "model", None)
            if model is None:
                raise RuntimeError("Unpickled CumlRFClassifier has no model")
            return np.asarray(model.predict(x))

    class SklearnRFClassifier:
        pass

    bindings = {
        "ml.frameworks.sklearn.cuml_classifier": {"CumlRFClassifier": CumlRFClassifier},
        "ml.frameworks.sklearn.classifier": {"SklearnRFClassifier": SklearnRFClassifier},
    }
    for mod_name, names in bindings.items():
        mod = module(mod_name)
        for attr, cls in names.items():
            cls.__module__ = mod_name
            setattr(mod, attr, cls)


def load_moe_family_models(model_artifact_dir: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    artifact_dir = Path(model_artifact_dir).expanduser().resolve()
    install_moe_pickle_compat_modules()
    with (artifact_dir / "classifier_families.pkl").open("rb") as handle:
        models = pickle.load(handle)
    meta = json.loads((artifact_dir / "classifier_families_meta.json").read_text(encoding="utf-8"))
    if not isinstance(models, dict):
        raise TypeError(f"MoE classifier artifact must be a family->model dict, got {type(models)!r}")
    return {str(key): value for key, value in models.items()}, dict(meta)


def load_moe_score_artifact(path: str | Path) -> pd.DataFrame:
    scored = pd.read_pickle(Path(path).expanduser().resolve())
    if not isinstance(scored, pd.DataFrame):
        raise TypeError("MoE score artifact must contain a pandas DataFrame")
    out = scored.copy()
    if "symbol" in out.columns:
        out["symbol"] = out["symbol"].astype(str).str.strip().str.upper()
        out = out.set_index("symbol", drop=True)
    out.index = pd.Index(out.index.astype(str).str.strip().str.upper(), name="symbol")
    for col in ("prob_buy", "close"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.sort_index()


def build_moe_ranked_scores(
    latest_scored: pd.DataFrame,
    *,
    top_k: int,
    threshold: float = 0.50,
) -> pd.DataFrame:
    frame = latest_scored.copy()
    if "symbol" in frame.columns:
        frame["symbol"] = frame["symbol"].astype(str).str.strip().str.upper()
        frame = frame.set_index("symbol", drop=True)
    frame.index = pd.Index(frame.index.astype(str).str.strip().str.upper(), name="symbol")
    frame["prob_buy"] = pd.to_numeric(frame.get("prob_buy"), errors="coerce")
    frame["close"] = pd.to_numeric(frame.get("close"), errors="coerce")
    frame = frame.loc[frame["prob_buy"].notna() & frame["close"].gt(0.0)].copy()
    frame = frame.sort_values(["prob_buy"], ascending=False, kind="stable")
    frame["eligible"] = frame["prob_buy"].gt(float(threshold))
    selected = frame.loc[frame["eligible"]].head(max(int(top_k), 0)).index
    frame["selected"] = frame.index.isin(selected)
    return frame


def score_moe_family_panel(
    feature_panel: pd.DataFrame,
    *,
    models: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> pd.DataFrame:
    if feature_panel.empty:
        return pd.DataFrame()
    panel = _normalize_panel_index(feature_panel)
    families = [str(family) for family in metadata.get("trained_families") or models.keys()]
    features_by_family = {
        str(family): [str(col) for col in cols]
        for family, cols in dict(metadata.get("feature_list_by_family") or {}).items()
    }
    weights = {str(k): float(v) for k, v in dict(metadata.get("feature_family_weights") or {}).items()}
    scored = pd.DataFrame(index=panel.index)
    weighted_sum = pd.Series(0.0, index=panel.index, dtype=float)
    weight_sum = pd.Series(0.0, index=panel.index, dtype=float)
    available_count = pd.Series(0, index=panel.index, dtype=int)
    active = 0
    for family in families:
        model = models.get(family)
        if model is None:
            continue
        features = features_by_family.get(family) or [str(col) for col in getattr(model, "_used_features", [])]
        if not features:
            continue
        x_raw = pd.DataFrame(index=panel.index)
        for col in features:
            x_raw[col] = pd.to_numeric(panel[col], errors="coerce") if col in panel.columns else np.nan
        available = x_raw.notna().any(axis=1)
        if not bool(available.any()):
            continue
        x = x_raw.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        prob = _positive_probability(model, x[features])
        prob_s = pd.Series(prob, index=panel.index, dtype=float).where(available)
        scored[f"{family}__prob_buy"] = prob_s
        valid = prob_s.notna()
        weight = float(weights.get(family, 1.0))
        weighted_sum.loc[valid] += weight * prob_s.loc[valid]
        weight_sum.loc[valid] += weight
        available_count.loc[valid] += 1
        active += 1
    scored["clf__prob_1"] = (weighted_sum / weight_sum.replace(0.0, np.nan)).clip(0.0, 1.0)
    scored["clf"] = scored["clf__prob_1"]
    scored["classifier_available_families"] = available_count
    scored["classifier_active_families"] = active
    scored["ranking"] = scored["clf__prob_1"]
    scored["ae_familiarity"] = 1.0
    scored["close"] = pd.to_numeric(panel.get("close"), errors="coerce")
    return enrich_moe_scored_panel(scored)


def enrich_moe_scored_panel(scored: pd.DataFrame) -> pd.DataFrame:
    out = _normalize_panel_index(scored)
    out["prob_buy"] = pd.to_numeric(out.get("clf__prob_1", out.get("prob_buy")), errors="coerce").fillna(0.0)
    out["prob_short"] = (1.0 - out["prob_buy"]).clip(0.0, 1.0)
    out["pred_rf_reg"] = pd.to_numeric(out.get("ranking", out["prob_buy"]), errors="coerce").fillna(out["prob_buy"])
    out["ae_familiarity"] = pd.to_numeric(out.get("ae_familiarity", 1.0), errors="coerce").fillna(1.0)

    def pct_rank(series: pd.Series) -> pd.Series:
        values = pd.to_numeric(series, errors="coerce")
        if values.notna().sum() <= 1:
            return pd.Series(np.where(values.notna(), 1.0, np.nan), index=series.index, dtype=float)
        return values.rank(pct=True, method="average")

    for col in ("prob_buy", "prob_short", "pred_rf_reg", "ae_familiarity"):
        out[f"{col}_pct"] = out.groupby(level="date", sort=False)[col].transform(pct_rank)
    out["buy_score_raw"] = out["prob_buy"] * out["pred_rf_reg"] * out["ae_familiarity"]
    out["short_score_raw"] = out["prob_short"] * out["pred_rf_reg"] * out["ae_familiarity"]
    out["buy_score_pct_product"] = out["prob_buy_pct"] * out["pred_rf_reg_pct"] * out["ae_familiarity_pct"]
    out["short_score_pct_product"] = out["prob_short_pct"] * out["pred_rf_reg_pct"] * out["ae_familiarity_pct"]
    out["buy_score_pct_mean"] = out[["prob_buy_pct", "pred_rf_reg_pct", "ae_familiarity_pct"]].mean(axis=1, skipna=True)
    out["short_score_pct_mean"] = out[["prob_short_pct", "pred_rf_reg_pct", "ae_familiarity_pct"]].mean(axis=1, skipna=True)
    out["buy_score_mean_raw3"] = out[["prob_buy", "pred_rf_reg", "ae_familiarity"]].mean(axis=1, skipna=True)
    out["buy_score_mean_raw_pct6"] = out[
        ["prob_buy", "pred_rf_reg", "ae_familiarity", "prob_buy_pct", "pred_rf_reg_pct", "ae_familiarity_pct"]
    ].mean(axis=1, skipna=True)
    out["short_score_mean_raw3"] = out[["prob_short", "pred_rf_reg", "ae_familiarity"]].mean(axis=1, skipna=True)
    out["short_score_mean_raw_pct6"] = out[
        ["prob_short", "pred_rf_reg", "ae_familiarity", "prob_short_pct", "pred_rf_reg_pct", "ae_familiarity_pct"]
    ].mean(axis=1, skipna=True)
    out["buy_score"] = out["buy_score_raw"]
    out["short_score"] = out["short_score_raw"]
    return out


def replay_moe_paper_top_k_rule(
    panel: pd.DataFrame,
    *,
    top_k: int,
    threshold: float,
    initial_balance: float,
    fee_bps: float,
    slippage_bps: float,
) -> TradingAppRuleReplay:
    work = _normalize_panel_index(panel)
    symbols = sorted(work.index.get_level_values("symbol").unique())
    close = _pivot(work, "close", symbols).replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    score = _pivot(work, "prob_buy", symbols).shift(1).reindex(index=close.index, columns=symbols)
    common_dates = close.index.intersection(score.index)
    close = close.loc[common_dates]
    score = score.loc[common_dates].replace([np.inf, -np.inf], np.nan)
    action_type = np.zeros((len(common_dates), len(symbols)), dtype=int)
    positions = pd.DataFrame(0, index=common_dates, columns=symbols, dtype=int)
    symbol_to_idx = {symbol: idx for idx, symbol in enumerate(symbols)}
    held: set[int] = set()
    for row_idx, dt in enumerate(common_dates):
        price_ok = close.loc[dt].gt(0.0)
        candidates = (score.loc[dt].gt(float(threshold)) & score.loc[dt].notna() & price_ok).fillna(False)
        target_symbols = list(score.loc[dt][candidates].sort_values(ascending=False, kind="stable").head(max(int(top_k), 0)).index)
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
    action_tape = _build_moe_action_tape(action_type=action_type, close=close, score=score, symbols=symbols, top_k=top_k, threshold=threshold)
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


def run_moe_paper_artifact_replay(config: MoePaperReplayConfig) -> MoePaperReplayResult:
    started = perf_counter()
    out_dir = Path(config.output_dir).expanduser().resolve() if config.output_dir is not None else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
    latest = load_moe_score_artifact(config.score_artifact)
    latest_ranked = build_moe_ranked_scores(latest, top_k=int(config.top_k), threshold=float(config.threshold))
    feature_panel = pd.DataFrame()
    if config.scored_panel_path is not None and Path(config.scored_panel_path).exists():
        scored = _read_panel_parquet(config.scored_panel_path)
    elif config.feature_panel_path is not None and Path(config.feature_panel_path).exists():
        feature_panel = _read_panel_parquet(config.feature_panel_path)
        if config.model_artifact_dir is None:
            raise ValueError("model_artifact_dir is required when scoring a feature_panel_path")
        models, metadata = load_moe_family_models(config.model_artifact_dir)
        scored = score_moe_family_panel(feature_panel, models=models, metadata=metadata)
    else:
        scored = _latest_snapshot_to_panel(latest)
    dates = scored.index.get_level_values("date") if not scored.empty else pd.DatetimeIndex([])
    scored = scored.loc[(dates >= pd.Timestamp(config.backtest_start)) & (dates <= pd.Timestamp(config.end_date))].copy() if not scored.empty else scored
    rule_replay = replay_moe_paper_top_k_rule(
        scored,
        top_k=int(config.top_k),
        threshold=float(config.threshold),
        initial_balance=float(config.initial_balance),
        fee_bps=float(config.fee_bps),
        slippage_bps=float(config.slippage_bps),
    ) if not scored.empty else TradingAppRuleReplay(pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {})
    option_execution = None
    option_portfolio = None
    option_summary: dict[str, Any] = {}
    if bool(config.run_fmp_synthetic_options) and not rule_replay.trade_windows.empty:
        from quant_orchestrator.platforms.backtesting_frameworks.optimal_trader.synthetic_options import (
            FmpSyntheticOptionReplayConfig,
            run_fmp_synthetic_option_trade_replay,
        )

        option_execution = run_fmp_synthetic_option_trade_replay(
            rule_replay.trade_windows,
            config=FmpSyntheticOptionReplayConfig(workers=max(1, int(config.option_workers))),
        )
        option_portfolio = replay_option_portfolio_from_selected_paths(
            option_execution.selected_option_trades,
            option_execution.selected_option_paths,
            date_index=rule_replay.equity.index,
            initial_balance=float(config.initial_balance),
        )
        option_summary = {
            "selected_option_trades": int(len(option_execution.selected_option_trades)),
            "selected_option_paths": int(len(option_execution.selected_option_paths)),
            "option_trade_returns": summarize_option_trade_returns(option_execution.selected_option_trades),
            "option_portfolio": option_portfolio.summary,
            "metrics": option_execution.metrics,
        }
    summary = {
        "score_artifact": str(Path(config.score_artifact).expanduser().resolve()),
        "model_artifact_dir": str(Path(config.model_artifact_dir).expanduser().resolve()) if config.model_artifact_dir else "",
        "feature_panel_path": str(Path(config.feature_panel_path).expanduser().resolve()) if config.feature_panel_path else "",
        "scored_panel_path": str(Path(config.scored_panel_path).expanduser().resolve()) if config.scored_panel_path else "",
        "optimal_trader_imported": False,
        "latest_symbols": int(len(latest_ranked)),
        "latest_selected": int(latest_ranked["selected"].sum()) if "selected" in latest_ranked else 0,
        "backtest_start": str(config.backtest_start),
        "end_date": str(config.end_date),
        "top_k": int(config.top_k),
        "threshold": float(config.threshold),
        "scored_rows": int(len(scored)),
        "trade_windows": int(len(rule_replay.trade_windows)),
        "rule_meta": rule_replay.meta,
        "performance": summarize_returns(rule_replay.returns, float(config.initial_balance)) if len(rule_replay.returns) else {},
        "fmp_synthetic_options": option_summary,
        "elapsed_seconds": float(perf_counter() - started),
    }
    result = MoePaperReplayResult(
        latest_ranked=latest_ranked,
        feature_panel=feature_panel,
        scored_panel=scored,
        rule_replay=rule_replay,
        summary=summary,
        option_execution=option_execution,
        option_portfolio=option_portfolio,
    )
    if out_dir is not None:
        write_moe_paper_replay_outputs(result, out_dir)
    return result


def write_moe_paper_replay_outputs(result: MoePaperReplayResult, output_dir: str | Path) -> dict[str, Path]:
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "latest_ranked": out_dir / "latest_ranked.csv",
        "scored_panel": out_dir / "scored_panel.parquet",
        "equity_curve": out_dir / "equity_curve.csv",
        "returns": out_dir / "returns.csv",
        "cash": out_dir / "cash.csv",
        "action_tape": out_dir / "action_tape.parquet",
        "positions": out_dir / "positions.parquet",
        "trade_windows": out_dir / "trade_windows.parquet",
        "equity_executions": out_dir / "equity_executions.csv",
        "summary": out_dir / "summary.json",
    }
    result.latest_ranked.to_csv(paths["latest_ranked"])
    result.scored_panel.reset_index().to_parquet(paths["scored_panel"], index=False)
    result.rule_replay.equity.rename("equity").to_csv(paths["equity_curve"])
    result.rule_replay.returns.rename("returns").to_csv(paths["returns"])
    result.rule_replay.cash.rename("cash").to_csv(paths["cash"])
    result.rule_replay.action_tape.to_parquet(paths["action_tape"], index=False)
    result.rule_replay.positions.to_parquet(paths["positions"])
    result.rule_replay.trade_windows.to_parquet(paths["trade_windows"], index=False)
    result.rule_replay.executions.to_csv(paths["equity_executions"], index=False)
    (paths["summary"]).write_text(json.dumps(result.summary, indent=2, default=str), encoding="utf-8")
    if result.option_execution is not None:
        paths["selected_option_trades"] = out_dir / "selected_option_trades.parquet"
        paths["selected_option_paths"] = out_dir / "selected_option_paths.parquet"
        paths["option_trade_status"] = out_dir / "option_trade_status.csv"
        result.option_execution.selected_option_trades.to_parquet(paths["selected_option_trades"], index=False)
        result.option_execution.selected_option_paths.to_parquet(paths["selected_option_paths"], index=False)
        result.option_execution.trade_status.to_csv(paths["option_trade_status"], index=False)
    if result.option_portfolio is not None:
        paths["option_equity_curve"] = out_dir / "option_equity_curve.csv"
        paths["option_cash"] = out_dir / "option_cash.csv"
        result.option_portfolio.equity.rename("equity").to_csv(paths["option_equity_curve"])
        result.option_portfolio.cash.rename("cash").to_csv(paths["option_cash"])
    standard_paths = write_strategy_artifacts(
        StrategyArtifactBundle(
            feature_panel=result.feature_panel if not result.feature_panel.empty else None,
            scored_panel=result.scored_panel,
            action_tape=result.rule_replay.action_tape,
            trade_windows=result.rule_replay.trade_windows,
            summary=result.summary,
            strategy_name="optimal_trader.moe_paper_trading",
        ),
        out_dir,
        extra_paths=paths,
    )
    paths.update({f"standard_{key}": value for key, value in standard_paths.items()})
    return paths


def _positive_probability(model: Any, frame: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_positive_proba"):
        return np.asarray(model.predict_positive_proba(frame), dtype=float)
    proba = np.asarray(model.predict_proba(frame), dtype=float)
    classes = getattr(getattr(model, "model", model), "classes_", None)
    if classes is None:
        classes = getattr(model, "_classes", None)
    if classes is None:
        idx = min(1, proba.shape[1] - 1)
    else:
        class_values = [str(value) for value in list(classes)]
        idx = class_values.index("1") if "1" in class_values else min(1, proba.shape[1] - 1)
    return proba[:, idx]


def _normalize_panel_index(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if isinstance(out.index, pd.MultiIndex) and {"date", "symbol"}.issubset(set(out.index.names)):
        out = out.reorder_levels(["date", "symbol"]).sort_index()
    else:
        if "date" not in out.columns or "symbol" not in out.columns:
            raise ValueError("MoE panel requires a MultiIndex(date, symbol) or date/symbol columns")
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
        out["symbol"] = out["symbol"].astype(str).str.strip().str.upper()
        out = out.dropna(subset=["date", "symbol"]).set_index(["date", "symbol"]).sort_index()
    dates = pd.to_datetime(out.index.get_level_values("date"), errors="coerce").normalize()
    symbols = out.index.get_level_values("symbol").astype(str).str.strip().str.upper()
    out.index = pd.MultiIndex.from_arrays([dates, symbols], names=["date", "symbol"])
    return out.sort_index()


def _read_panel_parquet(path: str | Path) -> pd.DataFrame:
    return _normalize_panel_index(pd.read_parquet(Path(path).expanduser().resolve()))


def _latest_snapshot_to_panel(latest: pd.DataFrame) -> pd.DataFrame:
    frame = latest.copy()
    if "date" not in frame.columns:
        raise ValueError("latest MoE score artifact has no date column; provide scored_panel_path for historical replay")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["symbol"] = frame.index.astype(str).str.upper()
    return enrich_moe_scored_panel(frame)


def _pivot(panel: pd.DataFrame, column: str, symbols: list[str]) -> pd.DataFrame:
    work = panel[[column]].reset_index()
    if work.duplicated(["date", "symbol"]).any():
        work = work.sort_values(["date", "symbol"]).groupby(["date", "symbol"], as_index=False, sort=False).last()
    return work.pivot(index="date", columns="symbol", values=column).reindex(columns=symbols).sort_index()


def _build_moe_action_tape(
    *,
    action_type: np.ndarray,
    close: pd.DataFrame,
    score: pd.DataFrame,
    symbols: list[str],
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
                    "prob_buy": _nan_float(score.iloc[row_idx, col_idx]),
                    "top_k": int(top_k),
                    "threshold": float(threshold),
                    "reason": "entry_moe_top_k" if action == "buy" else "exit_moe_top_k_or_invalid",
                }
            )
    return pd.DataFrame(rows)
