from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from quant_orchestrator.data import load_ohlcv
from quant_warehouse import Warehouse
from quant_warehouse.migrate.backfill_thetadata_options import backfill_thetadata_options
from quant_warehouse.target_engineering import (
    LabelBuildSpec,
    OptionLabelSpec,
    OptionMlDatasetSpec,
    ThetaDataDownloadSpec,
    build_option_ml_dataset,
    build_trade_results,
)

MAG7_SYMBOLS: tuple[str, ...] = ("AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA", "TSLA")


@dataclass(frozen=True)
class TradeModelBundle:
    pipeline: Any
    feature_columns: list[str]
    categorical_columns: list[str]
    numeric_columns: list[str]
    metrics: dict[str, Any]


@dataclass(frozen=True)
class PairwiseRankerBundle:
    preprocessor: Any
    model: Any
    feature_columns: list[str]
    categorical_columns: list[str]
    numeric_columns: list[str]
    metrics: dict[str, Any]


def resolve_option_dir() -> Path:
    return Warehouse().config.home / "options" / "thetadata"


def _normalize_datetime_index(index: pd.Index) -> pd.DatetimeIndex:
    dt_index = pd.DatetimeIndex(pd.to_datetime(index, errors="coerce"))
    if getattr(dt_index, "tz", None) is not None:
        dt_index = dt_index.tz_convert(None)
    return dt_index


def maybe_backfill_aapl_options(
    *,
    start_date: str,
    end_date: str,
    symbol: str = "AAPL",
    source: str = "arctic-fmp",
    overwrite: bool = False,
    skip_existing: bool = True,
) -> dict[str, Any]:
    summary = backfill_thetadata_options(
        symbols=[symbol],
        source=source,  # explicit symbol list takes precedence
        start_date=start_date,
        end_date=end_date,
        overwrite=overwrite,
        skip_existing=skip_existing,
        request_sleep=0.0,
        us_only=True,
    )
    return summary


def maybe_backfill_options(
    *,
    symbols: Sequence[str],
    start_date: str,
    end_date: str,
    source: str = "arctic-fmp",
    overwrite: bool = False,
    skip_existing: bool = True,
) -> dict[str, Any]:
    return backfill_thetadata_options(
        symbols=list(symbols),
        source=source,
        start_date=start_date,
        end_date=end_date,
        overwrite=overwrite,
        skip_existing=skip_existing,
        request_sleep=0.0,
        us_only=True,
    )


def load_price_history(symbol: str, *, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    prices = load_ohlcv(symbol, provider="fmp", start=start, end=end).copy()
    prices.index = _normalize_datetime_index(prices.index)
    return prices.sort_index()


def load_fundamental_panel(
    symbol: str,
    *,
    sections: Sequence[str] = ("ratios", "metrics"),
    start: str | None = None,
    end: str | None = None,
    provider: str = "fmp",
) -> pd.DataFrame:
    wh = Warehouse()
    frames: list[pd.DataFrame] = []
    for section in sections:
        raw = wh.read_fundamentals(symbol, section=section, provider=provider, start=start, end=end)
        if raw is None or raw.empty:
            continue
        frame = raw.copy()
        frame.index = _normalize_datetime_index(frame.index)
        frame = frame.loc[frame.index.notna()].sort_index()
        frame.columns = [f"{section}__{str(col).strip().lower()}" for col in frame.columns]
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    panel = pd.concat(frames, axis=1, join="outer").sort_index()
    panel = panel.loc[:, ~panel.columns.duplicated()]
    return panel


def build_daily_feature_frame(
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame | None = None,
) -> pd.DataFrame:
    close = pd.to_numeric(prices["close"], errors="coerce")
    volume = pd.to_numeric(prices.get("volume"), errors="coerce")
    frame = pd.DataFrame(index=prices.index)
    frame["close"] = close
    frame["ret_1d"] = close.pct_change(1)
    frame["ret_5d"] = close.pct_change(5)
    frame["ret_21d"] = close.pct_change(21)
    frame["ret_63d"] = close.pct_change(63)
    frame["sma_20_gap"] = close / close.rolling(20).mean() - 1.0
    frame["sma_50_gap"] = close / close.rolling(50).mean() - 1.0
    frame["vol_21d"] = close.pct_change().rolling(21).std()
    frame["vol_63d"] = close.pct_change().rolling(63).std()
    if volume is not None:
        frame["volume_z_21d"] = (volume - volume.rolling(21).mean()) / volume.rolling(21).std()
    else:
        frame["volume_z_21d"] = np.nan

    if fundamentals is not None and not fundamentals.empty:
        fund = fundamentals.copy()
        fund.index = _normalize_datetime_index(fund.index)
        fund = fund.loc[fund.index.notna()].sort_index()
        aligned = fund.reindex(prices.index, method="ffill")
        frame = frame.join(aligned, how="left")
    return frame.replace([np.inf, -np.inf], np.nan)


def build_trade_candidates(
    prices: pd.DataFrame,
    *,
    symbol: str = "AAPL",
    start: str | None = None,
    end: str | None = None,
    k_params: dict[str, list[int]] | None = None,
    min_profit_pct: float = 0.0,
) -> pd.DataFrame:
    spec = LabelBuildSpec(
        k_params=k_params or {"M": [1], "QE": [1], "YE": [1]},
        min_profit_pct=float(min_profit_pct),
        buy_execution="high",
        sell_execution="low",
        short_execution="low",
        cover_execution="high",
        start_date=start,
        end_date=end,
    )
    symbol = str(symbol).upper()
    result = build_trade_results([symbol], spec=spec, price_frames={symbol: prices})
    trades = pd.DataFrame(result.completed_trades)
    if trades.empty:
        return trades
    trades["entry_date"] = pd.to_datetime(trades["entry_date"], errors="coerce")
    trades["exit_date"] = pd.to_datetime(trades["exit_date"], errors="coerce")
    if "trade_id" not in trades.columns:
        trades["trade_id"] = trades.apply(
            lambda row: "|".join(
                [
                    str(row.get("symbol", "")).upper(),
                    str(row.get("side", "")),
                    str(row.get("freq", "")),
                    str(row.get("k", "")),
                    pd.Timestamp(row.get("entry_date")).date().isoformat() if pd.notna(row.get("entry_date")) else "",
                    pd.Timestamp(row.get("exit_date")).date().isoformat() if pd.notna(row.get("exit_date")) else "",
                ]
            ),
            axis=1,
        )
    trades["entry_px"] = pd.to_numeric(trades["entry_px"], errors="coerce")
    trades["exit_px"] = pd.to_numeric(trades["exit_px"], errors="coerce")
    trades["ret_dec"] = pd.to_numeric(trades["ret_dec"], errors="coerce")
    trades["hold_days"] = pd.to_numeric(trades["hold_days"], errors="coerce")
    return trades.sort_values(["entry_date", "exit_date", "freq", "k"]).reset_index(drop=True)


def build_trade_feature_panel(
    trades: pd.DataFrame,
    daily_features: pd.DataFrame,
) -> pd.DataFrame:
    if trades is None or trades.empty:
        return pd.DataFrame()
    features = daily_features.copy()
    features.index = _normalize_datetime_index(features.index)
    features = features.loc[features.index.notna()].sort_index()
    rows = trades.copy()
    rows["entry_date"] = pd.to_datetime(rows["entry_date"], errors="coerce").dt.normalize()
    if "trade_id" not in rows.columns:
        rows["trade_id"] = rows.apply(
            lambda row: "|".join(
                [
                    str(row.get("symbol", "")).upper(),
                    str(row.get("side", "")),
                    str(row.get("freq", "")),
                    str(row.get("k", "")),
                    pd.Timestamp(row.get("entry_date")).date().isoformat() if pd.notna(row.get("entry_date")) else "",
                    pd.Timestamp(row.get("exit_date")).date().isoformat() if pd.notna(row.get("exit_date")) else "",
                ]
            ),
            axis=1,
        )
    merged = rows.merge(
        features.reset_index().rename(columns={"index": "entry_date", "date": "entry_date"}),
        on="entry_date",
        how="left",
    )
    merged["target_return"] = pd.to_numeric(merged.get("ret_dec"), errors="coerce")
    merged["target_up"] = (merged["target_return"] > 0.0).astype(int)
    return merged


def run_symbol_options_research(
    symbol: str,
    *,
    trade_start: str = "2018-01-01",
    trade_end: str | None = None,
    train_cutoff: str | pd.Timestamp = "2025-01-01",
    download_missing: bool = False,
    k_params: dict[str, list[int]] | None = None,
    max_dte: int = 90,
    strike_range: int = 12,
    target_tenor_days: int = 60,
) -> dict[str, Any]:
    train_cutoff_ts = pd.Timestamp(train_cutoff)
    prices = load_price_history(symbol, start=trade_start, end=trade_end)
    fundamentals = load_fundamental_panel(symbol, sections=("ratios", "metrics"), start=trade_start, end=trade_end)
    daily_features = build_daily_feature_frame(prices, fundamentals)
    trades = build_trade_candidates(prices, symbol=symbol, start=trade_start, end=trade_end, k_params=k_params)
    trade_panel = build_trade_feature_panel(trades, daily_features)

    trade_train = trade_panel.loc[trade_panel["entry_date"] < train_cutoff_ts].copy()
    trade_model = train_trade_selector_model(trade_train, target_col="target_return", model_kind="regressor")
    trade_panel = trade_panel.copy()
    trade_panel["trade_selector_score"] = score_trade_candidates(trade_model, trade_panel)
    selected_trades = select_best_trade_per_date(trade_panel.assign(score=trade_panel["trade_selector_score"]))
    selected_train = selected_trades.loc[selected_trades["entry_date"] < train_cutoff_ts].copy()
    selected_eval = selected_trades.loc[selected_trades["entry_date"] >= train_cutoff_ts].copy()

    option_train_panel = build_option_training_panel(
        selected_train,
        download_missing=download_missing,
        max_dte=max_dte,
        strike_range=strike_range,
    )
    option_eval_panel = build_option_training_panel(
        selected_eval,
        download_missing=download_missing,
        max_dte=max_dte,
        strike_range=strike_range,
    )

    rank_train_panel = option_train_panel.loc[option_train_panel["label_method"].eq("rank")].copy()
    hybrid_train_panel = option_train_panel.loc[option_train_panel["label_method"].eq("hybrid")].copy()
    mean_var_train_panel = option_train_panel.loc[option_train_panel["label_method"].eq("mean_variance")].copy()

    ranker = train_pairwise_option_ranker(rank_train_panel)
    hybrid_model = train_trade_selector_model(
        hybrid_train_panel,
        target_col="target_value",
        model_kind="regressor",
        extra_exclude=(
            "contract_symbol",
            "trade_id",
            "trade_entry_date",
            "trade_exit_date",
            "entry_snapshot_date",
            "exit_snapshot_date",
            "option_return_pct",
            "rank_y",
            "mv_weight",
            "label",
            "target_value",
            "target_col",
            "label_method",
            "task_name",
        ),
    )

    variant_summary, variant_curves = backtest_trade_variants(
        selected_eval,
        option_panel=option_eval_panel,
        option_ranker=ranker,
        mv_option_model=hybrid_model,
        target_tenor_days=target_tenor_days,
    )
    curve_summary = summarize_variant_curves(variant_curves)
    return {
        "symbol": symbol,
        "prices": prices,
        "fundamentals": fundamentals,
        "daily_features": daily_features,
        "trades": trades,
        "trade_panel": trade_panel,
        "trade_model": trade_model,
        "selected_trades": selected_trades,
        "selected_train": selected_train,
        "selected_eval": selected_eval,
        "option_train_panel": option_train_panel,
        "option_eval_panel": option_eval_panel,
        "rank_train_panel": rank_train_panel,
        "hybrid_train_panel": hybrid_train_panel,
        "mean_var_train_panel": mean_var_train_panel,
        "option_ranker": ranker,
        "hybrid_option_model": hybrid_model,
        "variant_summary": variant_summary,
        "curve_summary": curve_summary,
        "variant_curves": variant_curves,
        "summary_row": {
            "symbol": symbol,
            "trade_rows": int(len(trade_panel)),
            "selected_rows": int(len(selected_trades)),
            "selected_eval_rows": int(len(selected_eval)),
            "option_train_rows": int(len(option_train_panel)),
            "option_eval_rows": int(len(option_eval_panel)),
            "trade_model_r2": trade_model.metrics.get("r2"),
            "trade_model_mae": trade_model.metrics.get("mae"),
            "pairwise_auc": ranker.metrics.get("pairwise_auc_in_sample"),
            "hybrid_r2": hybrid_model.metrics.get("r2"),
        },
    }


def run_mag7_options_research(
    symbols: Sequence[str] = MAG7_SYMBOLS,
    *,
    trade_start: str = "2018-01-01",
    trade_end: str | None = None,
    train_cutoff: str | pd.Timestamp = "2025-01-01",
    download_missing: bool = False,
    k_params: dict[str, list[int]] | None = None,
    max_dte: int = 90,
    strike_range: int = 12,
    target_tenor_days: int = 60,
) -> dict[str, Any]:
    symbol_results: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for symbol in symbols:
        try:
            result = run_symbol_options_research(
                symbol,
                trade_start=trade_start,
                trade_end=trade_end,
                train_cutoff=train_cutoff,
                download_missing=download_missing,
                k_params=k_params,
                max_dte=max_dte,
                strike_range=strike_range,
                target_tenor_days=target_tenor_days,
            )
            symbol_results.append(result)
            summaries.append(result["summary_row"])
        except Exception as exc:
            symbol_results.append({"symbol": symbol, "error": str(exc)})
            summaries.append({"symbol": symbol, "error": str(exc)})
    summary_frame = pd.DataFrame(summaries)
    return {
        "symbols": list(symbols),
        "symbol_results": symbol_results,
        "summary_frame": summary_frame,
        "trade_start": trade_start,
        "trade_end": trade_end,
        "train_cutoff": str(pd.Timestamp(train_cutoff).date()),
        "download_missing": download_missing,
        "k_params": k_params or {"M": [1], "QE": [1], "YE": [1]},
        "max_dte": max_dte,
        "strike_range": strike_range,
        "target_tenor_days": target_tenor_days,
    }


def _split_feature_columns(
    frame: pd.DataFrame,
    *,
    exclude: Sequence[str],
) -> tuple[list[str], list[str]]:
    cols = [col for col in frame.columns if col not in set(exclude)]
    categorical = [
        col
        for col in cols
        if (
            col in {"side", "freq", "symbol"}
            or frame[col].dtype == object
            or pd.api.types.is_categorical_dtype(frame[col])
        )
    ]
    numeric = [
        col
        for col in cols
        if col not in categorical and not pd.api.types.is_datetime64_any_dtype(frame[col])
    ]
    return numeric, categorical


def train_trade_selector_model(
    train_frame: pd.DataFrame,
    *,
    target_col: str = "target_return",
    model_kind: str = "regressor",
    extra_exclude: Sequence[str] = (),
) -> TradeModelBundle:
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score, average_precision_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    if train_frame.empty:
        raise ValueError("train_frame is empty")

    exclude = {
        "entry_date",
        "exit_date",
        "target_return",
        "target_up",
        "entry_px",
        "exit_px",
        "ret_dec",
        "ret_pct",
        "trade_id",
        "symbol",
    }
    exclude.update(extra_exclude)
    numeric_cols, categorical_cols = _split_feature_columns(train_frame, exclude=exclude)
    X = train_frame[numeric_cols + categorical_cols].copy()
    usable_numeric_cols = [col for col in numeric_cols if X[col].notna().any()]
    usable_categorical_cols = [col for col in categorical_cols if X[col].notna().any()]
    X = train_frame[usable_numeric_cols + usable_categorical_cols].copy()
    y = pd.to_numeric(train_frame[target_col], errors="coerce")
    valid = y.notna()
    X = X.loc[valid].copy()
    y = y.loc[valid].copy()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), usable_numeric_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                usable_categorical_cols,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    if model_kind == "classifier":
        model = RandomForestClassifier(
            n_estimators=400,
            max_depth=10,
            min_samples_leaf=4,
            random_state=1337,
            n_jobs=-1,
            class_weight="balanced",
        )
    else:
        model = RandomForestRegressor(
            n_estimators=500,
            max_depth=10,
            min_samples_leaf=4,
            random_state=1337,
            n_jobs=-1,
        )

    pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])
    pipeline.fit(X, y)
    pred = pipeline.predict(X)

    metrics: dict[str, Any] = {
        "rows": int(len(X)),
        "features": int(X.shape[1]),
        "numeric_features": len(usable_numeric_cols),
        "categorical_features": len(usable_categorical_cols),
        "target_col": target_col,
        "model_kind": model_kind,
    }
    if model_kind == "classifier":
        try:
            proba = pipeline.predict_proba(X)[:, 1]
            metrics["roc_auc"] = float(roc_auc_score(y, proba))
            metrics["pr_auc"] = float(average_precision_score(y, proba))
        except Exception:
            pass
    else:
        metrics["mae"] = float(mean_absolute_error(y, pred))
        metrics["r2"] = float(r2_score(y, pred)) if len(X) > 1 else None

    return TradeModelBundle(
        pipeline=pipeline,
        feature_columns=list(X.columns),
        categorical_columns=usable_categorical_cols,
        numeric_columns=usable_numeric_cols,
        metrics=metrics,
    )


def score_trade_candidates(bundle: TradeModelBundle, frame: pd.DataFrame) -> pd.Series:
    X = frame.loc[:, bundle.feature_columns].copy()
    if hasattr(bundle.pipeline, "predict_proba"):
        try:
            return pd.Series(bundle.pipeline.predict_proba(X)[:, 1], index=frame.index, name="score")
        except Exception:
            pass
    return pd.Series(bundle.pipeline.predict(X), index=frame.index, name="score")


def select_best_trade_per_date(
    scored_trades: pd.DataFrame,
    *,
    score_col: str = "score",
) -> pd.DataFrame:
    if scored_trades.empty:
        return scored_trades
    work = scored_trades.copy()
    work["entry_date"] = pd.to_datetime(work["entry_date"], errors="coerce").dt.normalize()
    work = work.loc[work["entry_date"].notna()].copy()
    picked = (
        work.sort_values(["entry_date", score_col, "ret_dec"], ascending=[True, False, False])
        .groupby("entry_date", as_index=False, sort=False)
        .head(1)
        .copy()
    )
    return picked.sort_values(["entry_date", "exit_date"]).reset_index(drop=True)


def build_option_training_panel(
    trades: pd.DataFrame,
    *,
    download_missing: bool = True,
    max_dte: int = 90,
    strike_range: int = 12,
) -> pd.DataFrame:
    spec = OptionMlDatasetSpec(
        rank_spec=OptionLabelSpec(label_method="rank", include_equity=False),
        mv_spec=OptionLabelSpec.diversified_mean_variance(include_equity=False),
        hybrid_spec=OptionLabelSpec.diversified_hybrid(include_equity=False),
        thetadata=ThetaDataDownloadSpec(max_dte=max_dte, strike_range=strike_range),
        download_missing=download_missing,
    )
    result = build_option_ml_dataset(trades, dataset_spec=spec)
    panel = pd.DataFrame(result.rows)
    if panel.empty:
        return panel
    panel = enrich_option_feature_panel(panel, trades=trades)
    panel["trade_entry_date"] = pd.to_datetime(panel["trade_entry_date"], errors="coerce")
    panel["trade_exit_date"] = pd.to_datetime(panel["trade_exit_date"], errors="coerce")
    panel["entry_snapshot_date"] = pd.to_datetime(panel["entry_snapshot_date"], errors="coerce")
    panel["exit_snapshot_date"] = pd.to_datetime(panel["exit_snapshot_date"], errors="coerce")
    panel["option_return_pct"] = pd.to_numeric(panel["option_return_pct"], errors="coerce")
    panel["rank_y"] = pd.to_numeric(panel["rank_y"], errors="coerce")
    panel["mv_weight"] = pd.to_numeric(panel["mv_weight"], errors="coerce")
    return panel


def enrich_option_feature_panel(
    panel: pd.DataFrame,
    *,
    trades: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Add option-aware features such as spot, spread, and moneyness."""

    if panel is None or panel.empty:
        return pd.DataFrame()

    work = panel.copy()
    numeric_candidates = [
        "strike_entry",
        "strike_exit",
        "bid_entry",
        "ask_entry",
        "mid_entry",
        "bid_exit",
        "ask_exit",
        "mid_exit",
        "volume_entry",
        "volume_exit",
        "count_entry",
        "count_exit",
        "bid_size_entry",
        "ask_size_entry",
        "bid_size_exit",
        "ask_size_exit",
        "trade_duration_days",
        "option_return_pct",
        "mv_weight",
        "rank_y",
        "underlying_return_pct",
    ]
    for col in numeric_candidates:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    if trades is not None and not trades.empty and "trade_id" in work.columns:
        trade_rows = trades.copy()
        if "trade_id" not in trade_rows.columns:
            trade_rows["trade_id"] = trade_rows.apply(
                lambda row: "|".join(
                    [
                        str(row.get("symbol", "")).upper(),
                        str(row.get("side", "")),
                        str(row.get("freq", "")),
                        str(row.get("k", "")),
                        pd.Timestamp(row.get("entry_date")).date().isoformat() if pd.notna(row.get("entry_date")) else "",
                        pd.Timestamp(row.get("exit_date")).date().isoformat() if pd.notna(row.get("exit_date")) else "",
                    ]
                ),
                axis=1,
            )
        trade_rows["trade_id"] = trade_rows["trade_id"].astype(str)
        trade_rows["entry_px"] = pd.to_numeric(trade_rows.get("entry_px"), errors="coerce")
        trade_rows["exit_px"] = pd.to_numeric(trade_rows.get("exit_px"), errors="coerce")
        lookup = trade_rows.loc[:, [c for c in ("trade_id", "entry_px", "exit_px", "entry_date", "exit_date", "side") if c in trade_rows.columns]].copy()
        work = work.merge(lookup, on="trade_id", how="left", suffixes=("", "_trade"))

    if "underlying_spot_entry" not in work.columns:
        if "entry_px" in work.columns:
            work["underlying_spot_entry"] = pd.to_numeric(work["entry_px"], errors="coerce")
        elif "entry_px_trade" in work.columns:
            work["underlying_spot_entry"] = pd.to_numeric(work["entry_px_trade"], errors="coerce")
        else:
            work["underlying_spot_entry"] = np.nan
    if "underlying_spot_exit" not in work.columns:
        if "exit_px" in work.columns:
            work["underlying_spot_exit"] = pd.to_numeric(work["exit_px"], errors="coerce")
        elif "exit_px_trade" in work.columns:
            work["underlying_spot_exit"] = pd.to_numeric(work["exit_px_trade"], errors="coerce")
        else:
            work["underlying_spot_exit"] = np.nan

    if "strike_entry" in work.columns:
        strike = pd.to_numeric(work["strike_entry"], errors="coerce")
    else:
        strike = pd.Series(np.nan, index=work.index, dtype=float)
    spot = pd.to_numeric(work["underlying_spot_entry"], errors="coerce")
    dte = None
    if {"trade_exit_date", "trade_entry_date"}.issubset(work.columns):
        entry = pd.to_datetime(work["trade_entry_date"], errors="coerce")
        exit_ = pd.to_datetime(work["trade_exit_date"], errors="coerce")
        dte = (exit_ - entry).dt.days
        work["days_to_expiry"] = dte
    elif "trade_duration_days" in work.columns:
        work["days_to_expiry"] = pd.to_numeric(work["trade_duration_days"], errors="coerce")
        dte = work["days_to_expiry"]
    else:
        work["days_to_expiry"] = np.nan
        dte = work["days_to_expiry"]

    work["entry_spread"] = pd.to_numeric(work.get("ask_entry"), errors="coerce") - pd.to_numeric(work.get("bid_entry"), errors="coerce")
    work["entry_mid"] = pd.to_numeric(work.get("mid_entry"), errors="coerce")
    if "entry_mid" not in work.columns or work["entry_mid"].isna().all():
        work["entry_mid"] = (pd.to_numeric(work.get("ask_entry"), errors="coerce") + pd.to_numeric(work.get("bid_entry"), errors="coerce")) / 2.0
    work["entry_spread_pct"] = work["entry_spread"] / work["entry_mid"].replace(0, np.nan)
    work["exit_spread"] = pd.to_numeric(work.get("ask_exit"), errors="coerce") - pd.to_numeric(work.get("bid_exit"), errors="coerce")
    work["exit_mid"] = pd.to_numeric(work.get("mid_exit"), errors="coerce")
    if "exit_mid" not in work.columns or work["exit_mid"].isna().all():
        work["exit_mid"] = (pd.to_numeric(work.get("ask_exit"), errors="coerce") + pd.to_numeric(work.get("bid_exit"), errors="coerce")) / 2.0
    work["exit_spread_pct"] = work["exit_spread"] / work["exit_mid"].replace(0, np.nan)

    work["moneyness_ratio_entry"] = strike / spot.replace(0, np.nan)
    work["moneyness_pct_entry"] = work["moneyness_ratio_entry"] - 1.0
    work["abs_moneyness_pct_entry"] = work["moneyness_pct_entry"].abs()
    work["log_moneyness_entry"] = np.log(work["moneyness_ratio_entry"].replace(0, np.nan))

    if "option_type_entry" in work.columns:
        is_call = work["option_type_entry"].astype(str).str.lower().str.startswith("c")
        work["is_call_entry"] = is_call.astype(int)
        work["is_put_entry"] = (~is_call).astype(int)

    greek_cols = [col for col in work.columns if any(token in col.lower() for token in ("delta", "gamma", "theta", "vega", "rho"))]
    for col in greek_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    if "days_to_expiry" in work.columns:
        work["days_to_expiry_norm"] = work["days_to_expiry"] / 365.0
    if "volume_entry" in work.columns:
        vol = pd.to_numeric(work["volume_entry"], errors="coerce")
        work["log_volume_entry"] = np.log1p(vol.clip(lower=0))
    if "volume_exit" in work.columns:
        vol = pd.to_numeric(work["volume_exit"], errors="coerce")
        work["log_volume_exit"] = np.log1p(vol.clip(lower=0))

    return work.replace([np.inf, -np.inf], np.nan)


def _pairwise_feature_frame(frame: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
    work = frame.copy()
    work = work.loc[:, list(feature_cols)].copy()
    for col in work.columns:
        if not pd.api.types.is_numeric_dtype(work[col]):
            work[col] = pd.to_numeric(work[col], errors="coerce")
    return work.replace([np.inf, -np.inf], np.nan)


def train_pairwise_option_ranker(
    panel: pd.DataFrame,
    *,
    target_col: str = "rank_y",
    max_pairs_per_trade: int | None = 2500,
) -> PairwiseRankerBundle:
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    if panel.empty:
        raise ValueError("panel is empty")

    exclude = {
        "trade_id",
        "contract_symbol",
        "trade_entry_date",
        "trade_exit_date",
        "entry_snapshot_date",
        "exit_snapshot_date",
        "option_return_pct",
        "rank_y",
        "mv_weight",
        "label",
        "target_value",
        "target_col",
        "label_method",
        "task_name",
    }
    numeric_cols, categorical_cols = _split_feature_columns(panel, exclude=exclude)
    feature_cols = numeric_cols + categorical_cols

    train_panel = panel.loc[panel[target_col].notna()].copy()
    X = train_panel.loc[:, feature_cols].copy()
    usable_numeric_cols = [col for col in numeric_cols if X[col].notna().any()]
    usable_categorical_cols = [col for col in categorical_cols if X[col].notna().any()]
    feature_cols = usable_numeric_cols + usable_categorical_cols
    X = train_panel.loc[:, feature_cols].copy()
    y = pd.to_numeric(train_panel[target_col], errors="coerce")
    valid = y.notna()
    X = X.loc[valid].copy()
    y = y.loc[valid].copy()
    groups = train_panel.loc[valid, "trade_id"].astype(str)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), usable_numeric_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                usable_categorical_cols,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    transformed = preprocessor.fit_transform(X)
    if not isinstance(transformed, np.ndarray):
        transformed = transformed.toarray()

    pairwise_rows: list[np.ndarray] = []
    pairwise_targets: list[int] = []
    rng = np.random.default_rng(1337)
    row_positions = pd.Series(np.arange(len(X), dtype=int), index=X.index)

    for _, group in train_panel.loc[valid].groupby(groups, sort=False):
        idx = list(group.index)
        if len(idx) < 2:
            continue
        positions = row_positions.reindex(idx).dropna().astype(int).to_numpy()
        group_y = y.loc[idx].to_numpy(dtype=float)
        group_x = transformed[positions, :]
        pairs: list[tuple[int, int]] = []
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                pairs.append((i, j))
        if max_pairs_per_trade is not None and len(pairs) > int(max_pairs_per_trade):
            chosen = rng.choice(len(pairs), size=int(max_pairs_per_trade), replace=False)
            pairs = [pairs[pos] for pos in chosen]
        for i, j in pairs:
            diff_ij = group_x[i] - group_x[j]
            diff_ji = group_x[j] - group_x[i]
            label_ij = int(group_y[i] > group_y[j])
            label_ji = int(group_y[j] > group_y[i])
            pairwise_rows.append(diff_ij)
            pairwise_targets.append(label_ij)
            pairwise_rows.append(diff_ji)
            pairwise_targets.append(label_ji)

    if not pairwise_rows:
        raise ValueError("No pairwise rows could be built")

    pairwise_X = np.vstack(pairwise_rows)
    pairwise_y = np.asarray(pairwise_targets, dtype=int)
    clf = LogisticRegression(max_iter=2000, solver="liblinear", class_weight="balanced")
    if np.unique(pairwise_y).size < 2:
        from sklearn.dummy import DummyClassifier

        clf = DummyClassifier(strategy="most_frequent")
    clf.fit(pairwise_X, pairwise_y)

    try:
        pairwise_auc = float(roc_auc_score(pairwise_y, clf.predict_proba(pairwise_X)[:, 1]))
    except Exception:
        pairwise_auc = None

    metrics = {
        "trade_rows": int(len(train_panel)),
        "pairwise_rows": int(len(pairwise_y)),
        "pairwise_auc_in_sample": pairwise_auc,
        "numeric_features": len(usable_numeric_cols),
        "categorical_features": len(usable_categorical_cols),
    }

    return PairwiseRankerBundle(
        preprocessor=preprocessor,
        model=clf,
        feature_columns=feature_cols,
        categorical_columns=usable_categorical_cols,
        numeric_columns=usable_numeric_cols,
        metrics=metrics,
    )


def score_option_panel(bundle: PairwiseRankerBundle, panel: pd.DataFrame) -> pd.Series:
    X = panel.loc[:, bundle.feature_columns].copy()
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = pd.to_numeric(X[col], errors="coerce")
    try:
        transformed = bundle.preprocessor.transform(X)
        if not isinstance(transformed, np.ndarray):
            transformed = transformed.toarray()
        scores = bundle.model.decision_function(transformed)
    except Exception:
        transformed = bundle.preprocessor.transform(X)
        if not isinstance(transformed, np.ndarray):
            transformed = transformed.toarray()
        scores = bundle.model.predict_proba(transformed)[:, 1]
    return pd.Series(scores, index=panel.index, name="pairwise_score")


def choose_fixed_bucket_contract(
    trade_panel: pd.DataFrame,
    *,
    bucket: str,
    target_tenor_days: int = 60,
) -> pd.Series | None:
    if trade_panel.empty:
        return None
    work = trade_panel.copy()
    if "trade_duration_days" in work.columns:
        work["trade_duration_days"] = pd.to_numeric(work["trade_duration_days"], errors="coerce")
    else:
        work["trade_duration_days"] = np.nan
    if "strike_entry" in work.columns:
        work["strike_entry"] = pd.to_numeric(work["strike_entry"], errors="coerce")
    if "entry_px" in work.columns:
        work["entry_px"] = pd.to_numeric(work["entry_px"], errors="coerce")
    if "option_type_entry" not in work.columns:
        return None

    spot = float(work["entry_px"].iloc[0]) if work["entry_px"].notna().any() else np.nan
    if not np.isfinite(spot) or spot <= 0:
        return None

    option_type = str(work["option_type_entry"].iloc[0]).lower()
    if option_type.startswith("c"):
        bucket_targets = {
            "otm": 1.05,
            "aitm": 0.95,
            "ditm": 0.90,
        }
    else:
        bucket_targets = {
            "otm": 0.95,
            "aitm": 1.05,
            "ditm": 1.10,
        }
    if bucket not in bucket_targets:
        raise ValueError(f"Unknown bucket {bucket}")

    target_strike = spot * bucket_targets[bucket]
    dte_col = None
    for candidate in ("days_to_expiry", "trade_duration_days"):
        if candidate in work.columns:
            dte_col = candidate
            break
    if dte_col is None:
        work["days_to_expiry"] = np.nan
        dte_col = "days_to_expiry"

    work["bucket_distance"] = (pd.to_numeric(work["strike_entry"], errors="coerce") - target_strike).abs()
    work["tenor_distance"] = (pd.to_numeric(work[dte_col], errors="coerce") - float(target_tenor_days)).abs()
    if "entry_spread_pct" not in work.columns:
        work["entry_spread_pct"] = np.nan
    work = work.sort_values(["tenor_distance", "bucket_distance", "entry_spread_pct"], ascending=[True, True, True])
    return work.iloc[0]


def choose_mv_option_basket(
    trade_group: pd.DataFrame,
    *,
    mv_model: TradeModelBundle,
    max_legs: int = 3,
    risk_aversion: float = 3.0,
) -> tuple[pd.DataFrame, float]:
    """Select a long-only multi-leg basket using MV weights over scored contracts."""

    if trade_group.empty:
        return pd.DataFrame(), 0.0

    work = trade_group.copy()
    if "contract_symbol" in work.columns:
        work = work.sort_values(["contract_symbol", "label_method" if "label_method" in work.columns else "contract_symbol"]).drop_duplicates(
            subset=["contract_symbol"], keep="first"
        )
    work["mv_selector_score"] = score_trade_candidates(mv_model, work)
    work = work.replace([np.inf, -np.inf], np.nan)
    work = work.sort_values(["mv_selector_score", "option_return_pct"], ascending=[False, False]).copy()

    if max_legs <= 0:
        max_legs = 1

    eligible = work["mv_selector_score"].notna() & (work["mv_selector_score"] > 0.0)
    if eligible.sum() == 0:
        eligible = work["mv_selector_score"].notna()

    if eligible.sum() > max_legs:
        cutoff_index = work.loc[eligible, "mv_selector_score"].nlargest(max_legs).index
        eligible = work.index.isin(cutoff_index)

    spread = pd.to_numeric(work.get("entry_spread_pct"), errors="coerce").fillna(0.25).clip(lower=0.0)
    moneyness = pd.to_numeric(work.get("abs_moneyness_pct_entry"), errors="coerce").fillna(0.0).clip(lower=0.0)
    tenor = pd.to_numeric(work.get("days_to_expiry_norm"), errors="coerce").fillna(
        pd.to_numeric(work.get("trade_duration_days"), errors="coerce").fillna(60.0) / 365.0
    )
    liquidity = pd.to_numeric(work.get("log_volume_entry"), errors="coerce").fillna(0.0)
    variances = (
        0.45 * spread.to_numpy(dtype=float)
        + 0.30 * moneyness.to_numpy(dtype=float)
        + 0.15 * (1.0 / np.maximum(pd.to_numeric(tenor, errors="coerce").to_numpy(dtype=float) * 365.0 + 1.0, 1.0))
        + 0.10 * (1.0 / np.maximum(np.exp(np.clip(liquidity.to_numpy(dtype=float), -10.0, 10.0)) + 1.0, 1.0))
    )
    variances = np.maximum(variances, 1e-6)
    expected = pd.to_numeric(work["mv_selector_score"], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    from quant_warehouse.target_engineering.option_labels import solve_long_only_mean_variance_weights

    weights = solve_long_only_mean_variance_weights(
        expected,
        variances,
        eligible=np.asarray(eligible, dtype=bool),
        risk_aversion=float(risk_aversion),
    )
    weights = np.asarray(weights, dtype=float)
    work["mv_weight_pred"] = weights
    portfolio = work.loc[work["mv_weight_pred"] > 1e-6].copy()
    if portfolio.empty:
        return pd.DataFrame(), 0.0

    return_decimal = pd.to_numeric(portfolio["option_return_pct"], errors="coerce").fillna(0.0).to_numpy(dtype=float) / 100.0
    basket_return = float(np.dot(portfolio["mv_weight_pred"].to_numpy(dtype=float), return_decimal))
    portfolio = portfolio.sort_values(["mv_weight_pred", "mv_selector_score"], ascending=[False, False]).reset_index(drop=True)
    return portfolio, basket_return


def backtest_trade_variants(
    selected_trades: pd.DataFrame,
    *,
    option_panel: pd.DataFrame,
    option_ranker: PairwiseRankerBundle,
    mv_option_model: TradeModelBundle | None = None,
    target_tenor_days: int = 60,
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    if selected_trades.empty:
        return pd.DataFrame(), {}

    variant_rows: list[dict[str, Any]] = []
    variant_names = ["equity", "otm", "aitm", "ditm", "ml_option_selector"]
    if mv_option_model is not None:
        variant_names.append("mv_option_selector")
    curves: dict[str, list[float]] = {name: [] for name in variant_names}
    equity_state: dict[str, float] = {name: 1.0 for name in variant_names}
    curve_index: list[pd.Timestamp] = []

    option_panel = option_panel.copy()
    if "trade_entry_date" in option_panel.columns:
        option_panel["trade_entry_date"] = pd.to_datetime(option_panel["trade_entry_date"], errors="coerce").dt.normalize()
    if "trade_exit_date" in option_panel.columns:
        option_panel["trade_exit_date"] = pd.to_datetime(option_panel["trade_exit_date"], errors="coerce").dt.normalize()

    def _return_decimal(value: Any) -> float:
        numeric = pd.to_numeric(value, errors="coerce")
        if pd.isna(numeric):
            return 0.0
        return float(numeric) / 100.0

    for _, trade in selected_trades.sort_values("entry_date").iterrows():
        trade_id = str(trade["trade_id"])
        entry_date = pd.Timestamp(trade["entry_date"]).normalize()
        if "trade_id" in option_panel.columns:
            trade_group = option_panel.loc[option_panel["trade_id"].astype(str).eq(trade_id)].copy()
        else:
            trade_group = pd.DataFrame()

        if not trade_group.empty and "contract_symbol" in trade_group.columns:
            trade_group = trade_group.sort_values(
                ["contract_symbol", "label_method"] if "label_method" in trade_group.columns else ["contract_symbol"]
            ).drop_duplicates(subset=["contract_symbol"], keep="first")

        trade_group["entry_px"] = pd.to_numeric(trade.get("entry_px"), errors="coerce")
        equity_ret = float(pd.to_numeric(trade.get("ret_dec"), errors="coerce") or 0.0)

        if trade_group.empty:
            variant_rows.append(
                {
                    "trade_id": trade_id,
                    "entry_date": entry_date,
                    "exit_date": pd.Timestamp(trade["exit_date"]).normalize(),
                    "equity_return": equity_ret,
                    "otm_return": np.nan,
                    "aitm_return": np.nan,
                    "ditm_return": np.nan,
                    "ml_option_selector_return": np.nan,
                    "mv_option_selector_return": np.nan,
                    "ml_pairwise_score": np.nan,
                    "selected_contract_ml": "",
                    "selected_contract_mv": "",
                    "mv_leg_count": 0,
                }
            )
            for name in variant_names:
                if name == "equity":
                    trade_ret = equity_ret
                else:
                    trade_ret = 0.0
                if pd.isna(trade_ret):
                    trade_ret = 0.0
                equity_state[name] *= 1.0 + float(trade_ret)
                curves[name].append(equity_state[name])
            curve_index.append(entry_date)
            continue

        trade_group["pairwise_score"] = score_option_panel(option_ranker, trade_group)
        mv_portfolio = pd.DataFrame()
        mv_return = np.nan
        if mv_option_model is not None:
            mv_portfolio, mv_return = choose_mv_option_basket(
                trade_group,
                mv_model=mv_option_model,
                max_legs=3,
                risk_aversion=3.0,
            )

        fixed_otm = choose_fixed_bucket_contract(trade_group, bucket="otm", target_tenor_days=target_tenor_days)
        fixed_aitm = choose_fixed_bucket_contract(trade_group, bucket="aitm", target_tenor_days=target_tenor_days)
        fixed_ditm = choose_fixed_bucket_contract(trade_group, bucket="ditm", target_tenor_days=target_tenor_days)
        ml_sel = trade_group.sort_values(["pairwise_score", "option_return_pct"], ascending=[False, False]).iloc[0]

        variant_rows.append(
            {
                "trade_id": trade_id,
                "entry_date": entry_date,
                "exit_date": pd.Timestamp(trade["exit_date"]).normalize(),
                "equity_return": equity_ret,
                "otm_return": _return_decimal(fixed_otm.get("option_return_pct")) if fixed_otm is not None else np.nan,
                "aitm_return": _return_decimal(fixed_aitm.get("option_return_pct")) if fixed_aitm is not None else np.nan,
                "ditm_return": _return_decimal(fixed_ditm.get("option_return_pct")) if fixed_ditm is not None else np.nan,
                "ml_option_selector_return": _return_decimal(ml_sel.get("option_return_pct")),
                "mv_option_selector_return": mv_return,
                "ml_pairwise_score": float(pd.to_numeric(ml_sel.get("pairwise_score"), errors="coerce")),
                "selected_contract_ml": str(ml_sel.get("contract_symbol", "")),
                "selected_contract_mv": ",".join(mv_portfolio["contract_symbol"].astype(str).tolist()) if mv_option_model is not None and not mv_portfolio.empty else "",
                "mv_leg_count": int(len(mv_portfolio)) if mv_option_model is not None else 0,
            }
        )

        for name, trade_ret in [
            ("equity", equity_ret),
            ("otm", variant_rows[-1]["otm_return"]),
            ("aitm", variant_rows[-1]["aitm_return"]),
            ("ditm", variant_rows[-1]["ditm_return"]),
            ("ml_option_selector", variant_rows[-1]["ml_option_selector_return"]),
        ]:
            if pd.isna(trade_ret):
                trade_ret = 0.0
            equity_state[name] *= 1.0 + float(trade_ret)
            curves[name].append(equity_state[name])
        if mv_option_model is not None:
            mv_ret = variant_rows[-1]["mv_option_selector_return"]
            if pd.isna(mv_ret):
                mv_ret = 0.0
            equity_state["mv_option_selector"] *= 1.0 + float(mv_ret)
            curves["mv_option_selector"].append(equity_state["mv_option_selector"])
        curve_index.append(entry_date)

    summary = pd.DataFrame(variant_rows)
    equity_curves = {name: pd.Series(values, index=curve_index, name=name) for name, values in curves.items()}
    return summary, equity_curves


def summarize_variant_curves(curves: dict[str, pd.Series]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, curve in curves.items():
        if curve is None or curve.empty:
            continue
        start = float(curve.iloc[0])
        total_return = float(curve.iloc[-1] / max(start, 1e-12) - 1.0)
        drawdown = curve / curve.cummax() - 1.0
        returns = curve.pct_change().fillna(0.0)
        rows.append(
            {
                "variant": name,
                "start": str(curve.index[0].date()),
                "end": str(curve.index[-1].date()),
                "trades": int(len(curve)),
                "final_equity": float(curve.iloc[-1]),
                "total_return": float(total_return),
                "max_drawdown": float(drawdown.min()),
                "daily_vol": float(returns.std()),
            }
        )
    return pd.DataFrame(rows).sort_values("total_return", ascending=False)
