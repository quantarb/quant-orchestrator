from __future__ import annotations

import argparse
import json
import pickle
import re
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
    parser.add_argument("--symbols", default="", help="Optional comma-separated symbol filter, e.g. NVDA,GOOG")
    parser.add_argument(
        "--feature-family",
        action="append",
        default=None,
        help="Feature family in '<source>.<family>' format. Can be passed multiple times or comma-separated.",
    )
    parser.add_argument("--all-feature-families", action="store_true")
    parser.add_argument("--output-dir", default="artifacts/option_family_ranker/per_family")
    parser.add_argument("--train-end", default="2020-12-31")
    parser.add_argument("--eval-start", default="2021-01-01")
    parser.add_argument("--max-family-features", type=int, default=25)
    parser.add_argument("--max-trades", type=int, default=0)
    parser.add_argument("--min-market-cap", type=int, default=1_000_000_000_000)
    parser.add_argument("--start-date", default="1900-01-01")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--random-seed", type=int, default=20260704)
    parser.add_argument("--target-col", default="rank_y", help="Numeric option label column to predict.")
    parser.add_argument("--basket-k", type=int, default=4, help="Max option legs per basket selector.")
    parser.add_argument("--basket-min-weight", type=float, default=0.0, help="Minimum positive score for weighted basket legs.")
    parser.add_argument("--disable-pairwise-ranker", action="store_true")
    parser.add_argument("--pairwise-pairs-per-trade", type=int, default=20)
    args = parser.parse_args()

    started = perf_counter()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    requested_symbols = _parse_symbols(args.symbols)
    option_panel = _load_option_panel(
        Path(args.option_panel),
        max_trades=int(args.max_trades),
        symbols=requested_symbols,
    )
    target_col = str(args.target_col)
    if target_col not in option_panel.columns:
        raise ValueError(f"option panel missing target column {target_col!r}")
    option_panel[target_col] = pd.to_numeric(option_panel[target_col], errors="coerce")
    if option_panel[target_col].notna().sum() == 0:
        raise ValueError(f"option panel target column {target_col!r} has no numeric values")
    symbols = tuple(sorted(option_panel["symbol"].dropna().astype(str).str.upper().unique()))
    feature_panel, metadata = _build_feature_panel(
        symbols,
        start_date=str(args.start_date),
        end_date=str(args.end_date) or None,
        min_market_cap=int(args.min_market_cap),
    )
    requested_families = _requested_feature_families(
        metadata,
        raw_values=args.feature_family,
        include_all=bool(args.all_feature_families),
    )

    train_base, eval_base = _split_by_entry_date(option_panel, train_end=str(args.train_end), eval_start=str(args.eval_start))
    option_features = [
        col
        for col in OPTION_FEATURES
        if col in option_panel.columns and pd.to_numeric(option_panel[col], errors="coerce").notna().any()
    ]
    fixed = _evaluate_selector(eval_base, score_col="fixed_near_atm_score", selector_name="fixed_near_atm", target_col=target_col)
    oracle_baskets = _oracle_basket_metrics(
        eval_base,
        target_col=target_col,
        basket_k=int(args.basket_k),
    )
    target_suffix = _safe_score_suffix(target_col)
    option_only_score_col = "pred_option_only_rank" if target_col == "rank_y" else f"pred_option_only_{target_suffix}"
    baseline = _fit_score_and_evaluate(
        train_base,
        eval_base,
        numeric_features=option_features,
        random_seed=int(args.random_seed),
        n_estimators=int(args.n_estimators),
        score_col=option_only_score_col,
        target_col=target_col,
        basket_k=int(args.basket_k),
        basket_min_weight=float(args.basket_min_weight),
        enable_pairwise=not bool(args.disable_pairwise_ranker),
        pairwise_pairs_per_trade=int(args.pairwise_pairs_per_trade),
    )
    if baseline["model"] is not None:
        _write_pickle(out_dir / "option_only_ranker.pkl", baseline["model"])
    if baseline["pairwise_model"] is not None:
        _write_pickle(out_dir / "option_only_pairwise_ranker.pkl", baseline["pairwise_model"])

    family_summaries = []
    selector_rows = [
        {"feature_family": "fixed_near_atm", "selector": "fixed_near_atm", **fixed},
        {"feature_family": "option_only", "selector": "option_only_ranker", **baseline["metrics"]["selector"]},
    ]
    if baseline["metrics"].get("pairwise_selector"):
        selector_rows.append(
            {"feature_family": "option_only", "selector": "option_only_pairwise_ranker", **baseline["metrics"]["pairwise_selector"]}
        )
    basket_rows = [
        {"feature_family": "oracle", "selector": name, **metrics}
        for name, metrics in oracle_baskets.items()
    ]
    basket_rows.extend(_basket_rows("option_only", baseline["metrics"]))
    print(
        f"[option-family-ranker] target={target_col} option_rows={len(option_panel)} "
        f"train_rows={len(train_base)} eval_rows={len(eval_base)} families={len(requested_families)} "
        f"option_only_elapsed={perf_counter() - started:.2f}s",
        flush=True,
    )
    for family_index, (source, family) in enumerate(requested_families):
        family_started = perf_counter()
        family_dir = out_dir / _safe_family_dir(source, family)
        family_dir.mkdir(parents=True, exist_ok=True)
        selected_features, feature_metadata, feature_quality = _select_family_features(
            feature_panel,
            metadata,
            source=source,
            family=family,
            max_features=int(args.max_family_features),
        )
        joined = _join_family_features(option_panel, feature_panel, selected_features)
        train, eval_ = _split_by_entry_date(joined, train_end=str(args.train_end), eval_start=str(args.eval_start))
        family_features = [
            col
            for col in selected_features
            if col in joined.columns and pd.to_numeric(joined[col], errors="coerce").notna().any()
        ]
        score_col = f"pred_{_safe_family_dir(source, family)}_rank" if target_col == "rank_y" else f"pred_{_safe_family_dir(source, family)}_{target_suffix}"
        family_aware = _fit_score_and_evaluate(
            train,
            eval_,
            numeric_features=[*option_features, *family_features],
            random_seed=int(args.random_seed) + 17 + family_index,
            n_estimators=int(args.n_estimators),
            score_col=score_col,
            target_col=target_col,
            basket_k=int(args.basket_k),
            basket_min_weight=float(args.basket_min_weight),
            enable_pairwise=not bool(args.disable_pairwise_ranker),
            pairwise_pairs_per_trade=int(args.pairwise_pairs_per_trade),
        )
        if family_aware["model"] is not None:
            _write_pickle(family_dir / "ranker.pkl", family_aware["model"])
        if family_aware["pairwise_model"] is not None:
            _write_pickle(family_dir / "pairwise_ranker.pkl", family_aware["pairwise_model"])
        scored_eval = family_aware["eval_scored"].copy()
        if not baseline["eval_scored"].empty and option_only_score_col in baseline["eval_scored"].columns:
            scored_eval[option_only_score_col] = baseline["eval_scored"][option_only_score_col].to_numpy()
        baseline_pairwise_score_col = baseline["metrics"].get("pairwise_score_col")
        if (
            not baseline["eval_scored"].empty
            and isinstance(baseline_pairwise_score_col, str)
            and baseline_pairwise_score_col in baseline["eval_scored"].columns
        ):
            scored_eval[baseline_pairwise_score_col] = baseline["eval_scored"][baseline_pairwise_score_col].to_numpy()
        summary = {
            "option_panel": str(Path(args.option_panel).expanduser().resolve()),
            "target_col": target_col,
            "requested_symbols": list(requested_symbols),
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
            "oracle_baskets": oracle_baskets,
            "option_only_ranker": baseline["metrics"],
            "family_ranker": family_aware["metrics"],
            "elapsed_seconds": float(perf_counter() - family_started),
        }
        selector_rows.append(
            {
                "feature_family": f"{source}.{family}",
                "selector": "option_plus_family_ranker",
                **family_aware["metrics"]["selector"],
            }
        )
        if family_aware["metrics"].get("pairwise_selector"):
            selector_rows.append(
                {
                    "feature_family": f"{source}.{family}",
                    "selector": "option_plus_family_pairwise_ranker",
                    **family_aware["metrics"]["pairwise_selector"],
                }
            )
        basket_rows.extend(_basket_rows(f"{source}.{family}", family_aware["metrics"]))
        family_summaries.append(_flatten_family_summary(summary))
        family_selector_rows = [
            {"selector": "fixed_near_atm", **fixed},
            {"selector": "option_only_ranker", **baseline["metrics"]["selector"]},
            {"selector": "option_plus_family_ranker", **family_aware["metrics"]["selector"]},
        ]
        if baseline["metrics"].get("pairwise_selector"):
            family_selector_rows.append({"selector": "option_only_pairwise_ranker", **baseline["metrics"]["pairwise_selector"]})
        if family_aware["metrics"].get("pairwise_selector"):
            family_selector_rows.append(
                {"selector": "option_plus_family_pairwise_ranker", **family_aware["metrics"]["pairwise_selector"]}
            )
        pd.DataFrame(family_selector_rows).to_csv(family_dir / "selector_summary.csv", index=False)
        pd.DataFrame(
            [
                {"selector": name, **metrics}
                for name, metrics in {
                    **oracle_baskets,
                    **_model_basket_metrics("option_only_ranker", baseline["metrics"]),
                    **_model_basket_metrics("option_plus_family_ranker", family_aware["metrics"]),
                }.items()
            ]
        ).to_csv(family_dir / "basket_summary.csv", index=False)
        scored_eval.to_parquet(family_dir / "eval_scored.parquet", index=False)
        feature_metadata.to_csv(family_dir / "feature_metadata.csv", index=False)
        feature_quality.to_csv(family_dir / "feature_quality.csv", index=False)
        (family_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        print(
            f"[option-family-ranker] target={target_col} "
            f"family={family_index + 1}/{len(requested_families)} {source}.{family} "
            f"features={len(family_features)} elapsed={summary['elapsed_seconds']:.2f}s "
            f"top_k_mean={family_aware['metrics']['top_k_equal_weight_basket'].get('mean_return')} "
            f"pairwise_top_k_mean={family_aware['metrics'].get('pairwise_top_k_equal_weight_basket', {}).get('mean_return')}",
            flush=True,
        )
    summary = {
        "option_panel": str(Path(args.option_panel).expanduser().resolve()),
        "target_col": target_col,
        "requested_symbols": list(requested_symbols),
        "feature_families": [f"{source}.{family}" for source, family in requested_families],
        "symbols": int(len(symbols)),
        "option_rows": int(len(option_panel)),
        "train_rows": int(len(train_base)),
        "eval_rows": int(len(eval_base)),
        "train_trades": int(train_base["trade_id"].nunique()) if not train_base.empty else 0,
        "eval_trades": int(eval_base["trade_id"].nunique()) if not eval_base.empty else 0,
        "option_feature_count": int(len(option_features)),
        "option_features": option_features,
        "fixed_near_atm": fixed,
        "oracle_baskets": oracle_baskets,
        "option_only_ranker": baseline["metrics"],
        "family_rankers": family_summaries,
        "elapsed_seconds": float(perf_counter() - started),
    }
    pd.DataFrame(selector_rows).to_csv(out_dir / "selector_summary.csv", index=False)
    pd.DataFrame(basket_rows).to_csv(out_dir / "basket_summary.csv", index=False)
    pd.DataFrame(family_summaries).to_csv(out_dir / "family_summary.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))


def _load_option_panel(path: Path, *, max_trades: int, symbols: tuple[str, ...] = ()) -> pd.DataFrame:
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
    if symbols:
        wanted = set(symbols)
        out = out.loc[out["symbol"].isin(wanted)].copy()
        if out.empty:
            raise ValueError(f"option panel has no rows for requested symbols: {sorted(wanted)}")
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


def _parse_symbols(value: str) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            symbol.strip().upper()
            for symbol in str(value or "").split(",")
            if symbol.strip()
        )
    )


def _parse_feature_family(value: str) -> tuple[str, str]:
    text = str(value).strip()
    if "." not in text:
        raise ValueError("--feature-family must be in '<source>.<family>' format")
    source, family = text.split(".", 1)
    return source.strip(), family.strip()


def _requested_feature_families(
    metadata: pd.DataFrame,
    *,
    raw_values: list[str] | None,
    include_all: bool,
) -> list[tuple[str, str]]:
    available = (
        metadata[["source", "family"]]
        .drop_duplicates()
        .sort_values(["source", "family"])
        .itertuples(index=False, name=None)
    )
    available_families = [(str(source), str(family)) for source, family in available]
    if include_all:
        return available_families
    values = raw_values or ["fmp.fmp_daily_mcap_yield"]
    requested: list[tuple[str, str]] = []
    for value in values:
        for part in str(value).split(","):
            requested.append(_parse_feature_family(part))
    known = set(available_families)
    missing = [f"{source}.{family}" for source, family in requested if (source, family) not in known]
    if missing:
        examples = [f"{source}.{family}" for source, family in available_families[:20]]
        raise ValueError(f"Unknown feature families: {missing}; available examples: {examples}")
    return list(dict.fromkeys(requested))


def _build_feature_panel(
    symbols: tuple[str, ...],
    *,
    start_date: str,
    end_date: str | None,
    min_market_cap: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    warehouse = Warehouse()
    config = FamilyEvaluationConfig(
        market_cap_min=int(min_market_cap),
        start_date=start_date,
        end_date=end_date,
    )
    panel, metadata, _diagnostics, _timings = build_fundamental_feature_panel(symbols, config, warehouse=warehouse)
    out = panel.copy()
    out["symbol"] = out["symbol"].astype(str).str.upper()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    return out, metadata


def _select_family_features(
    panel: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    source: str,
    family: str,
    max_features: int,
) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
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
    return selected, selected_meta, quality


def _safe_family_dir(source: str, family: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{source}.{family}").strip("._") or "feature_family"


def _safe_score_suffix(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(value)).strip("_") or "target"


def _write_pickle(path: Path, value: Any) -> None:
    with path.open("wb") as handle:
        pickle.dump(value, handle)


def _flatten_family_summary(summary: dict[str, Any]) -> dict[str, Any]:
    selector = dict(summary["family_ranker"]["selector"])
    top_k = dict(summary["family_ranker"].get("top_k_equal_weight_basket") or {})
    weighted = dict(summary["family_ranker"].get("score_weighted_basket") or {})
    pairwise_selector = dict(summary["family_ranker"].get("pairwise_selector") or {})
    pairwise_top_k = dict(summary["family_ranker"].get("pairwise_top_k_equal_weight_basket") or {})
    pairwise_weighted = dict(summary["family_ranker"].get("pairwise_score_weighted_basket") or {})
    return {
        "feature_family": summary["feature_family"],
        "family_feature_count": summary["family_feature_count"],
        "train_rows": summary["train_rows"],
        "eval_rows": summary["eval_rows"],
        "train_trades": summary["train_trades"],
        "eval_trades": summary["eval_trades"],
        "train_mae": summary["family_ranker"].get("train_mae"),
        "train_r2": summary["family_ranker"].get("train_r2"),
        "eval_mae": summary["family_ranker"].get("eval_mae"),
        "eval_r2": summary["family_ranker"].get("eval_r2"),
        "selected_mean_option_return": selector.get("mean_option_return"),
        "selected_median_option_return": selector.get("median_option_return"),
        "selected_win_rate": selector.get("win_rate"),
        "selected_mean_rank_y": selector.get("mean_rank_y"),
        "selected_median_rank_y": selector.get("median_rank_y"),
        "selected_mean_target": selector.get("mean_target"),
        "selected_median_target": selector.get("median_target"),
        "top_k_basket_mean_return": top_k.get("mean_return"),
        "top_k_basket_median_return": top_k.get("median_return"),
        "top_k_basket_win_rate": top_k.get("win_rate"),
        "top_k_basket_avg_legs": top_k.get("avg_legs_per_trade"),
        "weighted_basket_mean_return": weighted.get("mean_return"),
        "weighted_basket_median_return": weighted.get("median_return"),
        "weighted_basket_win_rate": weighted.get("win_rate"),
        "weighted_basket_avg_legs": weighted.get("avg_legs_per_trade"),
        "pairwise_train_pairs": summary["family_ranker"].get("pairwise_train_pairs"),
        "pairwise_train_accuracy": summary["family_ranker"].get("pairwise_train_accuracy"),
        "pairwise_selected_mean_option_return": pairwise_selector.get("mean_option_return"),
        "pairwise_selected_median_option_return": pairwise_selector.get("median_option_return"),
        "pairwise_selected_win_rate": pairwise_selector.get("win_rate"),
        "pairwise_top_k_basket_mean_return": pairwise_top_k.get("mean_return"),
        "pairwise_top_k_basket_median_return": pairwise_top_k.get("median_return"),
        "pairwise_top_k_basket_win_rate": pairwise_top_k.get("win_rate"),
        "pairwise_top_k_basket_avg_legs": pairwise_top_k.get("avg_legs_per_trade"),
        "pairwise_weighted_basket_mean_return": pairwise_weighted.get("mean_return"),
        "pairwise_weighted_basket_median_return": pairwise_weighted.get("median_return"),
        "pairwise_weighted_basket_win_rate": pairwise_weighted.get("win_rate"),
        "pairwise_weighted_basket_avg_legs": pairwise_weighted.get("avg_legs_per_trade"),
        "elapsed_seconds": summary["elapsed_seconds"],
    }


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
    target_col: str,
    basket_k: int,
    basket_min_weight: float,
    enable_pairwise: bool,
    pairwise_pairs_per_trade: int,
) -> dict[str, Any]:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.pipeline import Pipeline

    if train.empty or eval_.empty or not numeric_features:
        scored = eval_.copy()
        scored[score_col] = np.nan
        return {
            "model": None,
            "pairwise_model": None,
            "eval_scored": scored,
            "metrics": _empty_model_metrics(scored, score_col=score_col, target_col=target_col),
        }
    features = [col for col in numeric_features if col in train.columns and pd.to_numeric(train[col], errors="coerce").notna().any()]
    target = pd.to_numeric(train[target_col], errors="coerce")
    valid = target.notna()
    if not features or int(valid.sum()) == 0:
        scored = eval_.copy()
        scored[score_col] = np.nan
        return {
            "model": None,
            "pairwise_model": None,
            "eval_scored": scored,
            "metrics": _empty_model_metrics(scored, score_col=score_col, target_col=target_col),
        }
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
    predictions = model.predict(scored[features])
    scored[score_col] = np.clip(predictions, 0.0, 1.0) if target_col == "rank_y" else np.clip(predictions, 0.0, None)
    train_pred = model.predict(train.loc[valid, features])
    eval_target = pd.to_numeric(scored[target_col], errors="coerce")
    eval_pred = pd.to_numeric(scored[score_col], errors="coerce")
    eval_valid = eval_target.notna() & eval_pred.notna()
    metrics = {
        "feature_count": int(len(features)),
        "target_col": target_col,
        "train_rows": int(valid.sum()),
        "eval_rows": int(len(scored)),
        "train_target_mean": float(target.loc[valid].mean()),
        "eval_target_mean": float(eval_target.mean()),
        "train_mae": float(mean_absolute_error(target.loc[valid], train_pred)),
        "train_r2": float(r2_score(target.loc[valid], train_pred)) if int(valid.sum()) > 1 else np.nan,
        "eval_mae": float(mean_absolute_error(eval_target.loc[eval_valid], eval_pred.loc[eval_valid])) if eval_valid.any() else np.nan,
        "eval_r2": float(r2_score(eval_target.loc[eval_valid], eval_pred.loc[eval_valid])) if int(eval_valid.sum()) > 1 else np.nan,
        "selector": _evaluate_selector(scored, score_col=score_col, selector_name=score_col, target_col=target_col),
        "top_k_equal_weight_basket": _evaluate_top_k_basket(
            scored,
            score_col=score_col,
            selector_name=f"{score_col}_top_{int(basket_k)}_equal_weight",
            target_col=target_col,
            basket_k=int(basket_k),
        ),
        "score_weighted_basket": _evaluate_weighted_basket(
            scored,
            weight_col=score_col,
            selector_name=f"{score_col}_weighted_basket",
            target_col=target_col,
            basket_k=int(basket_k),
            min_weight=float(basket_min_weight),
        ),
    }
    pairwise_model = None
    if enable_pairwise:
        pairwise_score_col = f"{score_col}_pairwise"
        pairwise = _fit_pairwise_ranker(
            train,
            scored,
            numeric_features=features,
            target_col=target_col,
            score_col=pairwise_score_col,
            random_seed=int(random_seed) + 1009,
            n_estimators=int(n_estimators),
            pairs_per_trade=int(pairwise_pairs_per_trade),
            basket_k=int(basket_k),
            basket_min_weight=float(basket_min_weight),
        )
        pairwise_model = pairwise["model"]
        scored = pairwise["eval_scored"]
        metrics.update(pairwise["metrics"])
    return {"model": model, "pairwise_model": pairwise_model, "eval_scored": scored, "metrics": metrics}


def _empty_model_metrics(frame: pd.DataFrame, *, score_col: str, target_col: str) -> dict[str, Any]:
    return {
        "selector": _evaluate_selector(frame, score_col=score_col, selector_name=score_col, target_col=target_col),
        "top_k_equal_weight_basket": _evaluate_top_k_basket(
            frame,
            score_col=score_col,
            selector_name=f"{score_col}_top_k_equal_weight",
            target_col=target_col,
            basket_k=0,
        ),
        "score_weighted_basket": _evaluate_weighted_basket(
            frame,
            weight_col=score_col,
            selector_name=f"{score_col}_weighted_basket",
            target_col=target_col,
            basket_k=0,
        ),
    }


def _fit_pairwise_ranker(
    train: pd.DataFrame,
    eval_: pd.DataFrame,
    *,
    numeric_features: list[str],
    target_col: str,
    score_col: str,
    random_seed: int,
    n_estimators: int,
    pairs_per_trade: int,
    basket_k: int,
    basket_min_weight: float,
) -> dict[str, Any]:
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import accuracy_score
    from sklearn.pipeline import Pipeline

    scored = eval_.copy()
    scored[score_col] = np.nan
    pair_frame = _build_pairwise_training_frame(
        train,
        numeric_features=numeric_features,
        target_col=target_col,
        pairs_per_trade=int(pairs_per_trade),
        random_seed=int(random_seed),
    )
    if pair_frame.empty:
        return {
            "model": None,
            "eval_scored": scored,
            "metrics": {
                "pairwise_score_col": score_col,
                "pairwise_train_pairs": 0,
                "pairwise_selector": _evaluate_selector(scored, score_col=score_col, selector_name=score_col, target_col=target_col),
                "pairwise_top_k_equal_weight_basket": _evaluate_top_k_basket(
                    scored,
                    score_col=score_col,
                    selector_name=f"{score_col}_top_{int(basket_k)}_equal_weight",
                    target_col=target_col,
                    basket_k=int(basket_k),
                ),
                "pairwise_score_weighted_basket": _evaluate_weighted_basket(
                    scored,
                    weight_col=score_col,
                    selector_name=f"{score_col}_weighted_basket",
                    target_col=target_col,
                    basket_k=int(basket_k),
                    min_weight=float(basket_min_weight),
                ),
            },
        }
    model = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            (
                "clf",
                ExtraTreesClassifier(
                    n_estimators=max(50, min(200, int(n_estimators))),
                    max_depth=10,
                    min_samples_leaf=5,
                    n_jobs=-1,
                    random_state=int(random_seed),
                ),
            ),
        ]
    )
    feature_cols = [col for col in pair_frame.columns if col.startswith("diff__")]
    target = pair_frame["pairwise_label"].astype(int)
    model.fit(pair_frame[feature_cols], target)
    train_pred = model.predict(pair_frame[feature_cols])
    scored[score_col] = _score_pairwise_eval(
        model,
        scored,
        numeric_features=numeric_features,
    )
    metrics = {
        "pairwise_score_col": score_col,
        "pairwise_train_pairs": int(len(pair_frame)),
        "pairwise_train_accuracy": float(accuracy_score(target, train_pred)),
        "pairwise_selector": _evaluate_selector(scored, score_col=score_col, selector_name=score_col, target_col=target_col),
        "pairwise_top_k_equal_weight_basket": _evaluate_top_k_basket(
            scored,
            score_col=score_col,
            selector_name=f"{score_col}_top_{int(basket_k)}_equal_weight",
            target_col=target_col,
            basket_k=int(basket_k),
        ),
        "pairwise_score_weighted_basket": _evaluate_weighted_basket(
            scored,
            weight_col=score_col,
            selector_name=f"{score_col}_weighted_basket",
            target_col=target_col,
            basket_k=int(basket_k),
            min_weight=float(basket_min_weight),
        ),
    }
    return {"model": model, "eval_scored": scored, "metrics": metrics}


def _build_pairwise_training_frame(
    train: pd.DataFrame,
    *,
    numeric_features: list[str],
    target_col: str,
    pairs_per_trade: int,
    random_seed: int,
) -> pd.DataFrame:
    if train.empty or not numeric_features or int(pairs_per_trade) <= 0:
        return pd.DataFrame()
    rng = np.random.default_rng(int(random_seed))
    rows = []
    feature_cols = [col for col in numeric_features if col in train.columns]
    for _trade_id, group in train.groupby("trade_id", sort=False):
        work = group.dropna(subset=[target_col]).copy()
        if len(work) < 2:
            continue
        target = pd.to_numeric(work[target_col], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(target)
        if int(valid.sum()) < 2 or np.nanmax(target[valid]) <= np.nanmin(target[valid]):
            continue
        work = work.loc[valid].reset_index(drop=True)
        target = target[valid]
        features = work[feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        max_pairs = int(min(max(1, pairs_per_trade), len(work) * (len(work) - 1) // 2))
        made = 0
        attempts = 0
        while made < max_pairs and attempts < max_pairs * 20:
            attempts += 1
            left, right = rng.choice(len(work), size=2, replace=False)
            if target[left] == target[right]:
                continue
            label = int(target[left] > target[right])
            diff = features[left] - features[right]
            rows.append({**{f"diff__{col}": diff[i] for i, col in enumerate(feature_cols)}, "pairwise_label": label})
            rows.append({**{f"diff__{col}": -diff[i] for i, col in enumerate(feature_cols)}, "pairwise_label": 1 - label})
            made += 1
    return pd.DataFrame(rows)


def _score_pairwise_eval(model: Any, frame: pd.DataFrame, *, numeric_features: list[str]) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float)
    feature_cols = [col for col in numeric_features if col in frame.columns]
    if not feature_cols:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    diff_cols = [f"diff__{col}" for col in feature_cols]
    work = frame[feature_cols].apply(pd.to_numeric, errors="coerce")
    reference = work.groupby(frame["trade_id"], sort=False).transform("median")
    diffs = work.subtract(reference, axis="columns")
    diffs.columns = diff_cols
    probabilities = model.predict_proba(diffs)
    score = probabilities[:, 0] if probabilities.shape[1] == 1 else probabilities[:, 1]
    return pd.Series(score, index=frame.index, dtype=float)


def _evaluate_selector(frame: pd.DataFrame, *, score_col: str, selector_name: str, target_col: str = "rank_y") -> dict[str, Any]:
    if frame.empty or score_col not in frame.columns:
        return {"selector": selector_name, "trades": 0}
    work = frame.copy()
    work[score_col] = pd.to_numeric(work[score_col], errors="coerce")
    work["option_return"] = pd.to_numeric(work["option_return"], errors="coerce")
    if "rank_y" in work.columns:
        work["rank_y"] = pd.to_numeric(work["rank_y"], errors="coerce")
    if target_col in work.columns:
        work[target_col] = pd.to_numeric(work[target_col], errors="coerce")
    selected = (
        work.dropna(subset=[score_col])
        .sort_values(["trade_id", score_col], ascending=[True, False], kind="stable")
        .groupby("trade_id", as_index=False, sort=False)
        .head(1)
        .copy()
    )
    if selected.empty:
        return {"selector": selector_name, "trades": 0}
    result = {
        "selector": selector_name,
        "trades": int(len(selected)),
        "mean_option_return": float(selected["option_return"].mean()),
        "median_option_return": float(selected["option_return"].median()),
        "win_rate": float(selected["option_return"].gt(0).mean()),
    }
    if "rank_y" in selected.columns:
        result["mean_rank_y"] = float(selected["rank_y"].mean())
        result["median_rank_y"] = float(selected["rank_y"].median())
    if target_col in selected.columns:
        result["target_col"] = target_col
        result["mean_target"] = float(selected[target_col].mean())
        result["median_target"] = float(selected[target_col].median())
    return result


def _oracle_basket_metrics(frame: pd.DataFrame, *, target_col: str, basket_k: int) -> dict[str, dict[str, Any]]:
    metrics = {
        f"oracle_top_{int(basket_k)}_by_option_return": _evaluate_top_k_basket(
            frame,
            score_col="option_return",
            selector_name=f"oracle_top_{int(basket_k)}_by_option_return",
            target_col=target_col,
            basket_k=int(basket_k),
        ),
    }
    if "mv_weight" in frame.columns:
        metrics["oracle_mv_weighted_basket"] = _evaluate_weighted_basket(
            frame,
            weight_col="mv_weight",
            selector_name="oracle_mv_weighted_basket",
            target_col=target_col,
            basket_k=int(basket_k),
            min_weight=0.0,
        )
    return metrics


def _model_basket_metrics(prefix: str, metrics: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if "top_k_equal_weight_basket" in metrics:
        out[f"{prefix}_top_k_equal_weight_basket"] = metrics["top_k_equal_weight_basket"]
    if "score_weighted_basket" in metrics:
        out[f"{prefix}_score_weighted_basket"] = metrics["score_weighted_basket"]
    if "pairwise_top_k_equal_weight_basket" in metrics:
        out[f"{prefix}_pairwise_top_k_equal_weight_basket"] = metrics["pairwise_top_k_equal_weight_basket"]
    if "pairwise_score_weighted_basket" in metrics:
        out[f"{prefix}_pairwise_score_weighted_basket"] = metrics["pairwise_score_weighted_basket"]
    return out


def _basket_rows(feature_family: str, metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for selector, values in _model_basket_metrics("model", metrics).items():
        rows.append({"feature_family": feature_family, "selector": selector, **values})
    return rows


def _evaluate_top_k_basket(
    frame: pd.DataFrame,
    *,
    score_col: str,
    selector_name: str,
    target_col: str,
    basket_k: int,
) -> dict[str, Any]:
    if frame.empty or score_col not in frame.columns or int(basket_k) <= 0:
        return {"selector": selector_name, "trades": 0}
    work = frame.copy()
    work[score_col] = pd.to_numeric(work[score_col], errors="coerce")
    work = work.loc[work[score_col].notna()].copy()
    if work.empty:
        return {"selector": selector_name, "trades": 0}
    selected = (
        work.sort_values(["trade_id", score_col, "option_return"], ascending=[True, False, False], kind="stable")
        .groupby("trade_id", as_index=False, sort=False)
        .head(int(basket_k))
        .copy()
    )
    selected["basket_weight"] = selected.groupby("trade_id")["trade_id"].transform(lambda values: 1.0 / float(len(values)))
    return _summarize_basket(selected, selector_name=selector_name, target_col=target_col)


def _evaluate_weighted_basket(
    frame: pd.DataFrame,
    *,
    weight_col: str,
    selector_name: str,
    target_col: str,
    basket_k: int,
    min_weight: float = 0.0,
) -> dict[str, Any]:
    if frame.empty or weight_col not in frame.columns or int(basket_k) <= 0:
        return {"selector": selector_name, "trades": 0}
    work = frame.copy()
    work["_raw_basket_weight"] = pd.to_numeric(work[weight_col], errors="coerce").fillna(0.0).clip(lower=0.0)
    work = work.loc[work["_raw_basket_weight"].gt(float(min_weight))].copy()
    if work.empty:
        return {"selector": selector_name, "trades": 0}
    selected = (
        work.sort_values(["trade_id", "_raw_basket_weight", "option_return"], ascending=[True, False, False], kind="stable")
        .groupby("trade_id", as_index=False, sort=False)
        .head(int(basket_k))
        .copy()
    )
    totals = selected.groupby("trade_id")["_raw_basket_weight"].transform("sum")
    selected = selected.loc[totals.gt(0)].copy()
    if selected.empty:
        return {"selector": selector_name, "trades": 0}
    selected["basket_weight"] = selected["_raw_basket_weight"] / totals.loc[selected.index]
    return _summarize_basket(selected.drop(columns=["_raw_basket_weight"]), selector_name=selector_name, target_col=target_col)


def _summarize_basket(frame: pd.DataFrame, *, selector_name: str, target_col: str) -> dict[str, Any]:
    if frame.empty:
        return {"selector": selector_name, "trades": 0}
    work = frame.copy()
    work["basket_weight"] = pd.to_numeric(work["basket_weight"], errors="coerce").fillna(0.0)
    work["option_return"] = pd.to_numeric(work["option_return"], errors="coerce").fillna(0.0)
    work["_weighted_return"] = work["basket_weight"] * work["option_return"]
    trade_returns = work.groupby("trade_id", dropna=False)["_weighted_return"].sum()
    result = {
        "selector": selector_name,
        "trades": int(trade_returns.shape[0]),
        "legs": int(len(work)),
        "avg_legs_per_trade": float(work.groupby("trade_id", dropna=False).size().mean()),
        "mean_return": float(trade_returns.mean()),
        "median_return": float(trade_returns.median()),
        "win_rate": float(trade_returns.gt(0).mean()),
    }
    if "rank_y" in work.columns:
        work["rank_y"] = pd.to_numeric(work["rank_y"], errors="coerce")
        weighted_rank = (work["basket_weight"] * work["rank_y"]).groupby(work["trade_id"], dropna=False).sum()
        result["mean_weighted_rank_y"] = float(weighted_rank.mean())
        result["median_weighted_rank_y"] = float(weighted_rank.median())
    if target_col in work.columns:
        work[target_col] = pd.to_numeric(work[target_col], errors="coerce")
        weighted_target = (work["basket_weight"] * work[target_col]).groupby(work["trade_id"], dropna=False).sum()
        result["target_col"] = target_col
        result["mean_weighted_target"] = float(weighted_target.mean())
        result["median_weighted_target"] = float(weighted_target.median())
    return result


if __name__ == "__main__":
    main()
