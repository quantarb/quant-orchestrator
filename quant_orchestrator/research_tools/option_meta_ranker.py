from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from quant_orchestrator.research_tools.family_score_pipeline import FamilyScoreStore
from quant_orchestrator.research_tools.option_family_ranker import OPTION_FEATURES, _filter_oracle_entry_options, _load_option_panel


@dataclass(frozen=True)
class OptionMetaRankerConfig:
    option_panel: Path
    equity_score_store: Path
    output_dir: Path
    symbols: tuple[str, ...] = ()
    max_candidates_per_trade: int = 128
    max_trades: int = 0
    n_estimators: int = 300
    random_seed: int = 20260704
    target_col: str = "rank_y"
    model_backend: str = "rapids_random_forest"


@dataclass(frozen=True)
class OptionMetaRankerResult:
    output_dir: Path
    model_path: Path
    summary_path: Path
    training_rows: int
    training_trades: int
    features: tuple[str, ...]


def train_option_meta_ranker(config: OptionMetaRankerConfig) -> OptionMetaRankerResult:
    """Train one option ranker from option features plus reusable equity-family scores."""

    started = perf_counter()
    options = _load_option_panel(
        Path(config.option_panel),
        max_trades=int(config.max_trades),
        symbols=tuple(config.symbols),
        max_candidates_per_trade=int(config.max_candidates_per_trade),
    )
    options = _filter_oracle_entry_options(options, target_col=config.target_col)
    scores = FamilyScoreStore(Path(config.equity_score_store)).read_scores(model_ids=None)
    wide, score_features = wide_equity_family_scores(scores)
    stack = options.merge(
        wide.rename(columns={"date": "entry_date"}),
        on=["symbol", "entry_date"],
        how="inner",
        validate="many_to_one",
    )
    if stack.empty:
        raise RuntimeError("No option rows matched reusable equity-family scores on symbol/entry_date")
    option_features = [
        column
        for column in OPTION_FEATURES
        if column in stack.columns and pd.to_numeric(stack[column], errors="coerce").notna().any()
    ]
    features = [*option_features, *score_features]
    target = pd.to_numeric(stack[config.target_col], errors="coerce")
    valid = target.notna()
    if not features or not valid.any():
        raise RuntimeError("Option meta-ranker has no usable features or targets")
    model, medians = _fit_option_ranker(
        stack.loc[valid, features], target.loc[valid], config
    )
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "meta_stack_ranker.pkl"
    with model_path.open("wb") as handle:
        pickle.dump(
            {
                "schema_version": 3,
                "equity_score_contract": "family_long_probability_only",
                "model_backend": config.model_backend,
                "model": model,
                "medians": medians,
                "features": features,
                "option_features": option_features,
                "equity_score_cols": score_features,
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    summary = {
        "schema_version": 1,
        "design": "one option meta-ranker over option features and reusable equity-family scores",
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
        "option_rows_after_bounding": int(len(options)),
        "training_rows": int(valid.sum()),
        "training_trades": int(stack.loc[valid, "trade_id"].nunique()),
        "symbols": int(stack.loc[valid, "symbol"].nunique()),
        "option_features": option_features,
        "equity_score_features": score_features,
        "features": features,
        "elapsed_seconds": float(perf_counter() - started),
    }
    summary_path = output_dir / "meta_stack_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return OptionMetaRankerResult(
        output_dir=output_dir,
        model_path=model_path,
        summary_path=summary_path,
        training_rows=int(valid.sum()),
        training_trades=int(stack.loc[valid, "trade_id"].nunique()),
        features=tuple(features),
    )


def wide_equity_family_scores(scores: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    if scores is None or scores.empty:
        return pd.DataFrame(columns=["symbol", "date"]), []
    work = scores.copy()
    work["symbol"] = work["symbol"].astype(str).str.upper()
    work["date"] = pd.to_datetime(work["date"], errors="coerce").dt.normalize()
    frames: list[pd.DataFrame] = []
    features: list[str] = []
    for value_col in ("long_score",):
        wide = work.pivot_table(index=["symbol", "date"], columns="model_id", values=value_col, aggfunc="mean")
        wide.columns = [f"{value_col}__{str(column).replace('.', '_')}" for column in wide.columns]
        features.extend(wide.columns.astype(str).tolist())
        frames.append(wide)
    out = pd.concat(frames, axis=1).reset_index()
    for feature in features:
        out[feature] = pd.to_numeric(out[feature], errors="coerce").astype("float32")
    return out, features


def score_option_meta_ranker(
    model_path: Path,
    option_candidates: pd.DataFrame,
    equity_scores: pd.DataFrame,
) -> pd.DataFrame:
    """Score already-selected score-date option candidates with a persisted meta-ranker."""

    with Path(model_path).open("rb") as handle:
        bundle = pickle.load(handle)
    if bundle.get("schema_version") != 3 or bundle.get("equity_score_contract") != "family_long_probability_only":
        raise ValueError(
            "Option meta-ranker artifact is incompatible; retrain with family long-probability-only features"
        )
    model = bundle["model"]
    features = list(bundle["features"])
    candidates = option_candidates.copy()
    candidates["symbol"] = candidates["symbol"].astype(str).str.upper()
    candidates["entry_date"] = pd.to_datetime(candidates["entry_date"], errors="coerce").dt.normalize()
    wide, _ = wide_equity_family_scores(equity_scores)
    scored = candidates.merge(
        wide.rename(columns={"date": "entry_date"}),
        on=["symbol", "entry_date"],
        how="left",
        validate="many_to_one",
    )
    for feature in features:
        if feature not in scored.columns:
            scored[feature] = np.nan
    scored["pred_meta_stack_rank"] = np.clip(
        _predict_option_ranker(
            model,
            scored[features],
            medians=bundle["medians"],
            backend=str(bundle["model_backend"]),
        ),
        0.0,
        1.0,
    )
    scored = scored.sort_values(
        ["trade_id", "pred_meta_stack_rank"],
        ascending=[True, False],
        kind="stable",
    )
    scored["option_ensemble_rank"] = scored.groupby("trade_id", sort=False).cumcount() + 1
    scored["selected_by_option_ensemble"] = scored["option_ensemble_rank"].eq(1)
    return scored.reset_index(drop=True)


def _fit_option_ranker(
    features: pd.DataFrame,
    target: pd.Series,
    config: OptionMetaRankerConfig,
) -> tuple[Any, dict[str, float]]:
    medians = (
        features.apply(pd.to_numeric, errors="coerce").median(axis=0).fillna(0.0).astype(float)
    )
    training = features.apply(pd.to_numeric, errors="coerce").fillna(medians).astype("float32")
    y = pd.to_numeric(target, errors="coerce").astype("float32")
    if config.model_backend == "rapids_random_forest":
        import cudf
        from cuml.ensemble import RandomForestRegressor as CuRandomForestRegressor

        model = CuRandomForestRegressor(
            n_estimators=max(50, int(config.n_estimators)),
            max_depth=10,
            min_samples_leaf=3,
            max_features="sqrt",
            n_bins=128,
            n_streams=8,
            random_state=int(config.random_seed),
        )
        model.fit(cudf.from_pandas(training), cudf.Series(y.to_numpy()))
        return model, medians.to_dict()
    if config.model_backend == "sklearn_random_forest":
        model = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "rf",
                    RandomForestRegressor(
                        n_estimators=max(50, int(config.n_estimators)),
                        max_depth=10,
                        min_samples_leaf=3,
                        n_jobs=-1,
                        random_state=int(config.random_seed),
                    ),
                ),
            ]
        )
        model.fit(training, y)
        return model, medians.to_dict()
    raise ValueError(f"Unsupported option ranker backend: {config.model_backend}")


def _predict_option_ranker(
    model: Any,
    features: pd.DataFrame,
    *,
    medians: dict[str, float],
    backend: str,
) -> np.ndarray:
    work = features.apply(pd.to_numeric, errors="coerce").fillna(pd.Series(medians)).astype("float32")
    if backend == "rapids_random_forest":
        import cupy as cp
        import cudf

        prediction = model.predict(cudf.from_pandas(work))
        return prediction.to_numpy() if hasattr(prediction, "to_numpy") else cp.asnumpy(prediction)
    if backend == "sklearn_random_forest":
        return np.asarray(model.predict(work))
    raise ValueError(f"Unsupported option ranker backend: {backend}")
