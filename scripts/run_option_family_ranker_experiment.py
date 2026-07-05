from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
QUANT_WAREHOUSE_ROOT = REPO_ROOT.parent / "quant-warehouse"
for path in (REPO_ROOT, QUANT_WAREHOUSE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from quant_warehouse import Warehouse  # noqa: E402
from quant_warehouse.research_tools.feature_family_eval import (  # noqa: E402
    FamilyEvaluationConfig,
    build_fundamental_feature_panel,
    cap_features_by_quality,
)


DEFAULT_OPTION_PANEL = (
    Path("artifacts")
    / "clean_oracle_1t_options"
    / "options"
    / "option_full_chain_actions"
    / "individual__fmp.fmp_daily_mcap_yield"
    / "option_candidate_panel.parquet"
)


OPTION_FEATURES = (
    "dte",
    "dte_gap",
    "moneyness",
    "abs_moneyness",
    "spread_pct",
    "volume",
    "open_interest",
    "liquidity_score",
    "delta",
    "abs_delta",
    "gamma",
    "abs_gamma",
    "theta",
    "abs_theta",
    "vega",
    "abs_vega",
    "rho",
    "abs_rho",
    "theta_to_mid",
    "vega_to_mid",
    "iv",
    "iv_expiration_z",
    "iv_times_sqrt_dte",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--option-panel", default=str(DEFAULT_OPTION_PANEL))
    parser.add_argument("--feature-family", default="fmp.fmp_daily_mcap_yield")
    parser.add_argument("--output-dir", default="artifacts/option_family_ranker/fmp_daily_mcap_yield")
    parser.add_argument("--train-end", default="2020-12-31")
    parser.add_argument("--eval-start", default="2021-01-01")
    parser.add_argument("--max-family-features", type=int, default=25)
    parser.add_argument("--max-trades", type=int, default=0)
    parser.add_argument("--min-market-cap", type=int, default=1_000_000_000_000)
    parser.add_argument("--start-date", default="1900-01-01")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--random-seed", type=int, default=20260704)
    args = parser.parse_args()

    started = perf_counter()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    option_panel = _load_option_panel(Path(args.option_panel), max_trades=int(args.max_trades))
    source, family = _parse_feature_family(args.feature_family)
    symbols = tuple(sorted(option_panel["symbol"].dropna().astype(str).str.upper().unique()))
    feature_panel, selected_features, feature_metadata, feature_quality = _build_single_family_panel(
        symbols,
        source=source,
        family=family,
        start_date=str(args.start_date),
        end_date=str(args.end_date) or None,
        min_market_cap=int(args.min_market_cap),
        max_features=int(args.max_family_features),
    )
    joined = _join_family_features(option_panel, feature_panel, selected_features)
    train, eval_ = _split_by_entry_date(joined, train_end=str(args.train_end), eval_start=str(args.eval_start))

    option_features = [col for col in OPTION_FEATURES if col in joined.columns and pd.to_numeric(joined[col], errors="coerce").notna().any()]
    family_features = [col for col in selected_features if col in joined.columns and pd.to_numeric(joined[col], errors="coerce").notna().any()]
    baseline = _fit_score_and_evaluate(
        train,
        eval_,
        numeric_features=option_features,
        random_seed=int(args.random_seed),
        n_estimators=int(args.n_estimators),
        score_col="pred_option_only_rank",
    )
    family_aware = _fit_score_and_evaluate(
        train,
        eval_,
        numeric_features=[*option_features, *family_features],
        random_seed=int(args.random_seed) + 17,
        n_estimators=int(args.n_estimators),
        score_col="pred_option_plus_family_rank",
    )
    fixed = _evaluate_selector(eval_, score_col="fixed_near_atm_score", selector_name="fixed_near_atm")
    summary = {
        "option_panel": str(Path(args.option_panel).expanduser().resolve()),
        "feature_family": f"{source}.{family}",
        "symbols": int(len(symbols)),
        "option_rows": int(len(option_panel)),
        "joined_rows": int(len(joined)),
        "train_rows": int(len(train)),
        "eval_rows": int(len(eval_)),
        "train_trades": int(train["trade_id"].nunique()) if not train.empty else 0,
        "eval_trades": int(eval_["trade_id"].nunique()) if not eval_.empty else 0,
        "option_feature_count": int(len(option_features)),
        "family_feature_count": int(len(family_features)),
        "option_features": option_features,
        "family_features": family_features,
        "fixed_near_atm": fixed,
        "option_only_ranker": baseline["metrics"],
        "option_plus_family_ranker": family_aware["metrics"],
        "elapsed_seconds": float(perf_counter() - started),
    }
    scored_eval = eval_.copy()
    for scored in (baseline["eval_scored"], family_aware["eval_scored"]):
        for col in scored.columns:
            if col.startswith("pred_"):
                scored_eval[col] = scored[col]
    selector_rows = [
        {"selector": "fixed_near_atm", **fixed},
        {"selector": "option_only_ranker", **baseline["metrics"]["selector"]},
        {"selector": "option_plus_family_ranker", **family_aware["metrics"]["selector"]},
    ]
    pd.DataFrame(selector_rows).to_csv(out_dir / "selector_summary.csv", index=False)
    scored_eval.to_parquet(out_dir / "eval_scored.parquet", index=False)
    feature_metadata.to_csv(out_dir / "feature_metadata.csv", index=False)
    feature_quality.to_csv(out_dir / "feature_quality.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))


def _load_option_panel(path: Path, *, max_trades: int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing option candidate panel: {path}")
    frame = pd.read_parquet(path)
    required = {"trade_id", "symbol", "entry_date", "option_return"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"option panel missing required columns: {sorted(missing)}")
    out = frame.copy()
    out["symbol"] = out["symbol"].astype(str).str.upper()
    out["entry_date"] = pd.to_datetime(out["entry_date"], errors="coerce").dt.normalize()
    out["option_return"] = pd.to_numeric(out["option_return"], errors="coerce")
    out = out.dropna(subset=["trade_id", "symbol", "entry_date", "option_return"])
    if "rank_y" not in out.columns:
        out["rank_y"] = out.groupby("trade_id")["option_return"].rank(method="average", pct=True, ascending=True)
    else:
        out["rank_y"] = pd.to_numeric(out["rank_y"], errors="coerce")
    if "fixed_near_atm_score" not in out.columns:
        if "dte_gap" not in out.columns and "dte" in out.columns:
            out["dte_gap"] = (pd.to_numeric(out["dte"], errors="coerce") - 90.0).abs()
        score_terms = []
        for col in ("dte_gap", "abs_moneyness", "spread_pct"):
            if col in out.columns:
                values = pd.to_numeric(out[col], errors="coerce")
                scale = values.abs().median()
                if pd.notna(scale) and float(scale) > 0:
                    values = values / float(scale)
                score_terms.append(values.fillna(values.max()))
        out["fixed_near_atm_score"] = -sum(score_terms) if score_terms else np.nan
    if max_trades > 0:
        trade_ids = out[["trade_id", "entry_date"]].drop_duplicates().sort_values(["entry_date", "trade_id"]).head(max_trades)["trade_id"]
        out = out.loc[out["trade_id"].isin(set(trade_ids))].copy()
    return out.reset_index(drop=True)


def _parse_feature_family(value: str) -> tuple[str, str]:
    text = str(value).strip()
    if "." not in text:
        raise ValueError("--feature-family must be in '<source>.<family>' format")
    source, family = text.split(".", 1)
    return source.strip(), family.strip()


def _build_single_family_panel(
    symbols: tuple[str, ...],
    *,
    source: str,
    family: str,
    start_date: str,
    end_date: str | None,
    min_market_cap: int,
    max_features: int,
) -> tuple[pd.DataFrame, list[str], pd.DataFrame, pd.DataFrame]:
    warehouse = Warehouse()
    config = FamilyEvaluationConfig(
        market_cap_min=int(min_market_cap),
        start_date=start_date,
        end_date=end_date,
        max_features_per_family=max_features,
    )
    panel, metadata, _diagnostics, _timings = build_fundamental_feature_panel(symbols, config, warehouse=warehouse)
    family_meta = metadata.loc[
        metadata["source"].astype(str).eq(source) & metadata["family"].astype(str).eq(family)
    ].copy()
    if family_meta.empty:
        available = metadata[["source", "family"]].drop_duplicates().sort_values(["source", "family"])
        raise ValueError(f"feature family {source}.{family} not found; available examples: {available.head(20).to_dict('records')}")
    selected, selected_meta, quality = cap_features_by_quality(
        panel,
        family_meta,
        max_features=max_features,
    )
    keep = ["symbol", "date", *selected]
    out = panel[keep].copy()
    out["symbol"] = out["symbol"].astype(str).str.upper()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    return out, selected, selected_meta, quality


def _join_family_features(option_panel: pd.DataFrame, feature_panel: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    left = option_panel.copy()
    left["_row_id"] = np.arange(len(left))
    left = left.sort_values(["symbol", "entry_date", "_row_id"])
    right = feature_panel.rename(columns={"date": "feature_date"}).sort_values(["symbol", "feature_date"])
    merged_parts: list[pd.DataFrame] = []
    for symbol, group in left.groupby("symbol", sort=False):
        family = right.loc[right["symbol"].eq(symbol)]
        if family.empty:
            merged_parts.append(group)
            continue
        merged_parts.append(
            pd.merge_asof(
                group.sort_values("entry_date"),
                family[["symbol", "feature_date", *feature_cols]].sort_values("feature_date"),
                by="symbol",
                left_on="entry_date",
                right_on="feature_date",
                direction="backward",
            )
        )
    out = pd.concat(merged_parts, ignore_index=True, sort=False).sort_values("_row_id").drop(columns=["_row_id"])
    return out.reset_index(drop=True)


def _split_by_entry_date(frame: pd.DataFrame, *, train_end: str, eval_start: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.to_datetime(frame["entry_date"], errors="coerce").dt.normalize()
    train = frame.loc[dates.le(pd.Timestamp(train_end))].copy()
    eval_ = frame.loc[dates.ge(pd.Timestamp(eval_start))].copy()
    return train.reset_index(drop=True), eval_.reset_index(drop=True)


def _fit_score_and_evaluate(
    train: pd.DataFrame,
    eval_: pd.DataFrame,
    *,
    numeric_features: list[str],
    random_seed: int,
    n_estimators: int,
    score_col: str,
) -> dict[str, Any]:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.pipeline import Pipeline

    if train.empty or eval_.empty or not numeric_features:
        scored = eval_.copy()
        scored[score_col] = np.nan
        return {"model": None, "eval_scored": scored, "metrics": {"selector": _evaluate_selector(scored, score_col=score_col, selector_name=score_col)}}
    features = [col for col in numeric_features if col in train.columns and pd.to_numeric(train[col], errors="coerce").notna().any()]
    target = pd.to_numeric(train["rank_y"], errors="coerce")
    valid = target.notna()
    model = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            (
                "rf",
                RandomForestRegressor(
                    n_estimators=max(50, int(n_estimators)),
                    max_depth=10,
                    min_samples_leaf=3,
                    n_jobs=-1,
                    random_state=int(random_seed),
                ),
            ),
        ]
    )
    model.fit(train.loc[valid, features], target.loc[valid])
    scored = eval_.copy()
    scored[score_col] = np.clip(model.predict(scored[features]), 0.0, 1.0)
    train_pred = model.predict(train.loc[valid, features])
    eval_target = pd.to_numeric(scored["rank_y"], errors="coerce")
    eval_pred = pd.to_numeric(scored[score_col], errors="coerce")
    eval_valid = eval_target.notna() & eval_pred.notna()
    metrics = {
        "feature_count": int(len(features)),
        "train_rows": int(valid.sum()),
        "eval_rows": int(len(scored)),
        "train_mae": float(mean_absolute_error(target.loc[valid], train_pred)),
        "train_r2": float(r2_score(target.loc[valid], train_pred)) if int(valid.sum()) > 1 else np.nan,
        "eval_mae": float(mean_absolute_error(eval_target.loc[eval_valid], eval_pred.loc[eval_valid])) if eval_valid.any() else np.nan,
        "eval_r2": float(r2_score(eval_target.loc[eval_valid], eval_pred.loc[eval_valid])) if int(eval_valid.sum()) > 1 else np.nan,
        "selector": _evaluate_selector(scored, score_col=score_col, selector_name=score_col),
    }
    return {"model": model, "eval_scored": scored, "metrics": metrics}


def _evaluate_selector(frame: pd.DataFrame, *, score_col: str, selector_name: str) -> dict[str, Any]:
    if frame.empty or score_col not in frame.columns:
        return {"selector": selector_name, "trades": 0}
    work = frame.copy()
    work[score_col] = pd.to_numeric(work[score_col], errors="coerce")
    work["option_return"] = pd.to_numeric(work["option_return"], errors="coerce")
    work["rank_y"] = pd.to_numeric(work["rank_y"], errors="coerce")
    selected = (
        work.dropna(subset=[score_col])
        .sort_values(["trade_id", score_col], ascending=[True, False], kind="stable")
        .groupby("trade_id", as_index=False, sort=False)
        .head(1)
        .copy()
    )
    if selected.empty:
        return {"selector": selector_name, "trades": 0}
    return {
        "selector": selector_name,
        "trades": int(len(selected)),
        "mean_option_return": float(selected["option_return"].mean()),
        "median_option_return": float(selected["option_return"].median()),
        "win_rate": float(selected["option_return"].gt(0).mean()),
        "mean_rank_y": float(selected["rank_y"].mean()),
        "median_rank_y": float(selected["rank_y"].median()),
    }


if __name__ == "__main__":
    main()
