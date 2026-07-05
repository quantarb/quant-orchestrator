from __future__ import annotations

import json
import pickle
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import numpy as np
import pandas as pd

from quant_orchestrator.artifact_contracts import (
    StrategyArtifactBundle,
    write_strategy_artifacts,
)


@dataclass(frozen=True)
class OptimalTraderArtifactReplayConfig:
    artifact_dir: Path
    output_dir: Path | None = None
    feature_start: str = "2020-01-01"
    backtest_start: str = "2022-01-01"
    end_date: str = "2026-06-23"
    top_k: int = 20
    component_threshold: float = 0.50
    price_provider: str = "fmp"
    max_symbols: int = 0
    initial_balance: float = 100_000.0
    reuse_feature_panel: bool = False
    reuse_scored_panel: bool = False
    run_fmp_synthetic_options: bool = False
    option_workers: int = 1


@dataclass(frozen=True)
class TradingAppRuleReplay:
    equity: pd.Series
    returns: pd.Series
    cash: pd.Series
    action_tape: pd.DataFrame
    executions: pd.DataFrame
    positions: pd.DataFrame
    trade_list: pd.DataFrame
    meta: dict[str, Any]


@dataclass(frozen=True)
class OptimalTraderArtifactReplayResult:
    feature_panel: pd.DataFrame
    scored_panel: pd.DataFrame
    rule_replay: TradingAppRuleReplay
    summary: dict[str, Any]
    option_execution: Any | None = None
    option_portfolio: Any | None = None


@dataclass(frozen=True)
class OptionPortfolioReplay:
    equity: pd.Series
    returns: pd.Series
    cash: pd.Series
    active_value: pd.Series
    capital_deployed: pd.Series
    trade_ledger: pd.DataFrame
    summary: dict[str, Any]


def install_pickle_compat_modules() -> None:
    """Provide minimal classes for old saved artifacts without importing optimal_trader."""

    import torch
    import torch.nn as nn
    import torch.nn.init as init

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
        "ml.frameworks.sklearn.classifier",
        "ml.frameworks.sklearn.regressor",
        "ml.autoencoder",
        "ml.autoencoder.adapter",
        "ml.autoencoder.config",
        "ml.autoencoder.trainer",
        "ml.autoencoder.model",
    ):
        module(name)

    class SklearnRFClassifier:
        pass

    class SklearnRFRegressor:
        pass

    class AutoEncoderConfig:
        pass

    class AutoEncoderArtifact:
        pass

    def build_halving_dims(in_dim: int, n_layers: int, min_layer_dim: int) -> list[int]:
        dims: list[int] = []
        prev = int(in_dim)
        for _ in range(int(n_layers)):
            nxt = max(int(min_layer_dim), prev // 2)
            dims.append(nxt)
            prev = nxt
        return dims

    def build_encoder(in_dim: int, dims: list[int]) -> nn.Sequential:
        layers: list[nn.Module] = []
        prev = int(in_dim)
        for i, dim in enumerate(dims):
            layers.append(nn.Linear(prev, int(dim)))
            if i < len(dims) - 1:
                layers.append(nn.BatchNorm1d(int(dim)))
                layers.append(nn.ReLU())
            prev = int(dim)
        return nn.Sequential(*layers)

    def build_decoder(out_dim: int, dims: list[int]) -> nn.Sequential:
        layers: list[nn.Module] = []
        prev = int(dims[-1])
        for dim in reversed(dims[:-1]):
            layers.append(nn.Linear(prev, int(dim)))
            layers.append(nn.BatchNorm1d(int(dim)))
            layers.append(nn.ReLU())
            prev = int(dim)
        layers.append(nn.Linear(prev, int(out_dim)))
        return nn.Sequential(*layers)

    class NumericAutoEncoder(nn.Module):
        def __init__(self, in_dim: int, n_layers: int, min_layer_dim: int = 2) -> None:
            super().__init__()
            self.in_dim = int(in_dim)
            self.n_layers = int(n_layers)
            self.layer_dims = build_halving_dims(self.in_dim, self.n_layers, int(min_layer_dim))
            self.bottleneck_dim = int(self.layer_dims[-1])
            self.encoder = build_encoder(self.in_dim, self.layer_dims)
            self.decoder = build_decoder(self.in_dim, self.layer_dims)
            self.apply(self._init_weights)

        @staticmethod
        def _init_weights(layer: nn.Module) -> None:
            if isinstance(layer, nn.Linear):
                init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
                if layer.bias is not None:
                    init.constant_(layer.bias, 0)

        def forward(self, x_num: torch.Tensor) -> torch.Tensor:
            return self.decoder(self.encoder(x_num))

    class DynamicHybridAutoEncoder(nn.Module):
        def __init__(
            self,
            in_dim: int,
            cat_cardinalities: list[int],
            embed_dim: int,
            n_layers: int,
            min_layer_dim: int = 2,
        ) -> None:
            super().__init__()
            self.in_dim = int(in_dim)
            self.embed_dim = int(embed_dim)
            self.n_layers = int(n_layers)
            self.cat_cardinalities = [int(x) for x in cat_cardinalities]
            self.cat_embeddings = nn.ModuleList([nn.Embedding(cardinality, self.embed_dim) for cardinality in self.cat_cardinalities])
            self.cat_feature_dim = len(self.cat_cardinalities) * self.embed_dim
            self.cat_norm = nn.LayerNorm(self.cat_feature_dim) if self.cat_feature_dim > 0 else nn.Identity()
            total_in = self.in_dim + self.cat_feature_dim
            self.layer_dims = build_halving_dims(total_in, self.n_layers, int(min_layer_dim))
            self.bottleneck_dim = int(self.layer_dims[-1])
            self.encoder = build_encoder(total_in, self.layer_dims)
            self.decoder = build_decoder(self.in_dim, self.layer_dims)
            self.apply(NumericAutoEncoder._init_weights)

        def encode(self, x_num: torch.Tensor, x_cats: torch.Tensor) -> torch.Tensor:
            embed_vecs = [emb(x_cats[:, i]) for i, emb in enumerate(self.cat_embeddings)]
            cat_features = torch.cat(embed_vecs, dim=1) if embed_vecs else torch.empty((x_num.shape[0], 0), device=x_num.device)
            return self.encoder(torch.cat([x_num, self.cat_norm(cat_features)], dim=1))

        def forward(self, x_num: torch.Tensor, x_cats: torch.Tensor) -> torch.Tensor:
            return self.decoder(self.encode(x_num, x_cats))

    class TorchAutoEncoder:
        pass

    bindings = {
        "ml.frameworks.sklearn.classifier": {"SklearnRFClassifier": SklearnRFClassifier},
        "ml.frameworks.sklearn.regressor": {"SklearnRFRegressor": SklearnRFRegressor},
        "ml.autoencoder.config": {"AutoEncoderConfig": AutoEncoderConfig},
        "ml.autoencoder.trainer": {"AutoEncoderArtifact": AutoEncoderArtifact},
        "ml.autoencoder.model": {
            "NumericAutoEncoder": NumericAutoEncoder,
            "DynamicHybridAutoEncoder": DynamicHybridAutoEncoder,
        },
        "ml.autoencoder.adapter": {"TorchAutoEncoder": TorchAutoEncoder},
    }
    for mod_name, names in bindings.items():
        mod = module(mod_name)
        for attr, cls in names.items():
            cls.__module__ = mod_name
            setattr(mod, attr, cls)


def read_pickle(path: Path) -> Any:
    install_pickle_compat_modules()
    with path.open("rb") as handle:
        return pickle.load(handle)


def clean_price_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    out = frame.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        if "date" in out.columns:
            out["date"] = pd.to_datetime(out["date"], errors="coerce")
            out = out.set_index("date")
        else:
            out.index = pd.to_datetime(out.index, errors="coerce")
    out = out.loc[out.index.notna()].sort_index()
    for src, dst in (("adj_open", "open"), ("adj_high", "high"), ("adj_low", "low"), ("adj_close", "close")):
        if dst not in out.columns and src in out.columns:
            out[dst] = out[src]
    keep = [
        col
        for col in ("open", "high", "low", "close", "volume", "adj_open", "adj_high", "adj_low", "adj_close")
        if col in out.columns
    ]
    out = out[keep].copy()
    for col in keep:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out.index.name = "date"
    return out.dropna(subset=["close"])


def read_price_frame(warehouse: Any, symbol: str, *, start: str, end: str, provider: str) -> tuple[pd.DataFrame, str]:
    providers = [provider, "fmp", "yfinance"]
    for candidate in dict.fromkeys(providers):
        raw = warehouse.read_prices(symbol, provider=candidate, start=start, end=end)
        frame = clean_price_frame(raw)
        if not frame.empty:
            return frame, candidate
    return pd.DataFrame(), ""


def build_technical_panel(symbols: Sequence[str], *, start: str, end: str, provider: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    from quant_warehouse import Warehouse
    from quant_warehouse.platforms.data_providers.fmp.feature_engineering import build_price_technical_features

    warehouse = Warehouse()
    frames: list[pd.DataFrame] = []
    provider_counts: dict[str, int] = {}
    skipped: list[str] = []
    started = perf_counter()
    for idx, symbol in enumerate(symbols, start=1):
        prices, used_provider = read_price_frame(warehouse, symbol, start=start, end=end, provider=provider)
        if prices.empty:
            skipped.append(symbol)
            continue
        built = build_price_technical_features(symbol, prices)
        if built.df.empty:
            skipped.append(symbol)
            continue
        px = prices[["open", "high", "low", "close", "volume"]].copy()
        for src, dst in (
            ("adj_open", "px__adj_open"),
            ("adj_high", "px__adj_high"),
            ("adj_low", "px__adj_low"),
            ("adj_close", "px__adj_close"),
        ):
            px[dst] = pd.to_numeric(prices[src] if src in prices.columns else prices[src.removeprefix("adj_")], errors="coerce")
        px["symbol"] = symbol
        px = px.reset_index().set_index(["date", "symbol"]).sort_index()
        frames.append(px.join(built.df, how="left"))
        provider_counts[used_provider] = provider_counts.get(used_provider, 0) + 1
        if idx % 50 == 0:
            print(f"[technical] {idx:,}/{len(symbols):,} symbols | built={len(frames):,} | elapsed={perf_counter() - started:.1f}s", flush=True)
    if not frames:
        raise RuntimeError("No technical frames were built from quant-warehouse prices.")
    panel = pd.concat(frames, axis=0).sort_index()
    if panel.index.has_duplicates:
        panel = panel[~panel.index.duplicated(keep="last")]
    return panel, {
        "technical_rows": int(len(panel)),
        "technical_symbols": int(panel.index.get_level_values("symbol").nunique()),
        "price_provider_counts": provider_counts,
        "technical_skipped_symbols": skipped,
        "technical_seconds": perf_counter() - started,
    }


def snake_to_pascal(value: str) -> str:
    raw = str(value)
    overrides = {
        "ev_to_sales": "EVToSales",
        "ev_to_operating_cash_flow": "EVToOperatingCashFlow",
        "ev_to_free_cash_flow": "EVToFreeCashFlow",
        "ev_to_ebitda": "EVToEBITDA",
        "net_debt_to_ebitda": "NetDebtToEBITDA",
        "ebit_margin": "EBITMargin",
        "ebitda_margin": "EBITDAMargin",
        "net_income_per_ebt": "NetIncomePerEBT",
        "ebt_per_ebit": "EBTPerEBIT",
        "days_since_last_fundamental_update": "Dayssincelastfundamentalupdate",
        "is_new_fundamental_update": "Isnewfundamentalupdate",
    }
    if raw in overrides:
        return overrides[raw]
    return "".join(part[:1].upper() + part[1:] for part in raw.split("_") if part)


def present_fundamental_columns(frame: pd.DataFrame) -> pd.DataFrame:
    columns: dict[str, pd.Series] = {}
    for col in frame.columns:
        text = str(col)
        if text.startswith(("km__", "rt__")):
            target = snake_to_pascal(text.split("__", 1)[1])
        else:
            continue
        if target in columns:
            continue
        columns[target] = pd.to_numeric(frame[col], errors="coerce")
    return pd.DataFrame(columns, index=frame.index)


def build_fundamental_panel(symbols: Sequence[str], target_index: pd.Index, *, start: str, end: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    from quant_warehouse.platforms.data_providers.fmp.feature_engineering import (
        broadcast_fundamentals_to_daily,
        fetch_fundamentals_data,
    )

    started = perf_counter()
    sparse = fetch_fundamentals_data(
        list(symbols),
        period="quarter",
        limit=160,
        verbose=False,
        use_filing_lag=True,
        filing_lag_days=45,
    )
    if sparse.empty:
        return pd.DataFrame(index=target_index), {"fundamental_rows": 0, "fundamental_columns": 0, "fundamental_seconds": perf_counter() - started}
    sparse = sparse.loc[
        (sparse.index.get_level_values("date") >= pd.Timestamp(start))
        & (sparse.index.get_level_values("date") <= pd.Timestamp(end))
    ].copy()
    dense = broadcast_fundamentals_to_daily(sparse, target_index)
    presented = present_fundamental_columns(dense)
    event_index = pd.MultiIndex.from_tuples(
        [(pd.Timestamp(date).normalize(), str(symbol).upper()) for date, symbol in sparse.index.to_list()],
        names=["date", "symbol"],
    )
    event = pd.Series(0.0, index=target_index, dtype="float64")
    aligned_events = target_index.intersection(event_index)
    if len(aligned_events):
        event.loc[aligned_events] = 1.0
    timing = pd.DataFrame(
        {
            "_date": pd.to_datetime(target_index.get_level_values("date")),
            "_symbol": target_index.get_level_values("symbol").astype(str).str.upper(),
            "_event": event.to_numpy(dtype=float),
        },
        index=target_index,
    ).sort_values(["_symbol", "_date"])
    timing["_last_update_date"] = timing["_date"].where(timing["_event"].gt(0.0))
    timing["_last_update_date"] = timing.groupby("_symbol", sort=False)["_last_update_date"].ffill()
    presented = pd.concat(
        [
            presented,
            pd.DataFrame(
                {
                    "Dayssincelastfundamentalupdate": (timing["_date"] - timing["_last_update_date"]).dt.days.astype("float64").reindex(target_index),
                    "Isnewfundamentalupdate": event,
                },
                index=target_index,
            ),
        ],
        axis=1,
    )
    return presented, {
        "fundamental_sparse_rows": int(len(sparse)),
        "fundamental_rows": int(len(presented)),
        "fundamental_columns": int(len(presented.columns)),
        "fundamental_seconds": perf_counter() - started,
    }


def build_macro_panel(target_index: pd.Index, *, start: str, end: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    from quant_warehouse import Warehouse

    artifact_macro_names = {
        "GDP": "GDP",
        "CPI": "CPI",
        "UnemploymentRate": "unemploymentRate",
        "InflationRate": "inflationRate",
        "FederalFunds": "federalFunds",
        "USTMonth1": "macro__ust_month_1",
        "USTMonth2": "macro__ust_month_2",
        "USTMonth3": "macro__ust_month_3",
        "USTMonth6": "macro__ust_month_6",
        "USTYear1": "macro__ust_year_1",
        "USTYear2": "macro__ust_year_2",
        "USTYear3": "macro__ust_year_3",
        "USTYear5": "macro__ust_year_5",
        "USTYear7": "macro__ust_year_7",
        "USTYear10": "macro__ust_year_10",
        "USTYear20": "macro__ust_year_20",
        "USTYear30": "macro__ust_year_30",
    }
    started = perf_counter()
    warehouse = Warehouse()
    panel = warehouse.read_macro_panel(list(artifact_macro_names.values()), start=start, end=end)
    out = pd.DataFrame(index=target_index)
    if panel.empty:
        return out, {"macro_rows": 0, "macro_columns": 0, "macro_seconds": perf_counter() - started}
    panel = panel.sort_index().ffill()
    target_dates = pd.DatetimeIndex(pd.to_datetime(target_index.get_level_values("date"))).normalize()
    aligned = panel.reindex(sorted(target_dates.unique())).ffill().reindex(target_dates)
    aligned.index = target_index
    for artifact_name, source_name in artifact_macro_names.items():
        if source_name in aligned.columns:
            out[artifact_name] = pd.to_numeric(aligned[source_name], errors="coerce")
    return out, {
        "macro_rows": int(len(out)),
        "macro_columns": int(len(out.columns)),
        "macro_seconds": perf_counter() - started,
    }


def build_feature_panel(
    symbols: Sequence[str],
    *,
    feature_cols: list[str],
    start: str,
    end: str,
    price_provider: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    technical, tech_meta = build_technical_panel(symbols, start=start, end=end, provider=price_provider)
    fundamentals, fund_meta = build_fundamental_panel(symbols, technical.index, start=start, end=end)
    macro, macro_meta = build_macro_panel(technical.index, start=start, end=end)
    panel = pd.concat([technical, fundamentals, macro], axis=1).sort_index()
    panel = panel.loc[:, ~panel.columns.duplicated(keep="first")]
    present = [col for col in feature_cols if col in panel.columns]
    missing = [col for col in feature_cols if col not in panel.columns]
    for col in missing:
        panel[col] = 0.0
    panel.loc[:, feature_cols] = panel[feature_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return panel, {
        **tech_meta,
        **fund_meta,
        **macro_meta,
        "feature_rows": int(len(panel)),
        "feature_columns": int(len(panel.columns)),
        "artifact_feature_count": int(len(feature_cols)),
        "present_artifact_feature_count": int(len(present)),
        "missing_artifact_features": missing,
    }


def standardize_apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray, lower: np.ndarray | None, upper: np.ndarray | None) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if lower is not None and upper is not None and len(lower) and len(upper):
        arr = np.clip(arr, np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64))
    arr = np.where(np.isfinite(arr), arr, mean)
    arr = (arr - mean) / std
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(arr, -50.0, 50.0).astype(np.float32, copy=False)


def autoencoder_familiarity(ae: Any, frame: pd.DataFrame, numeric_cols: list[str], *, batch_size: int = 8192) -> np.ndarray:
    import torch
    from sklearn.neighbors import NearestNeighbors

    artifact = ae._artifact
    work = frame.copy()
    for col in numeric_cols:
        if col not in work.columns:
            work[col] = 0.0
    x_num = work[numeric_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    xs = standardize_apply(x_num, artifact.mean_, artifact.std_, artifact.lower_, artifact.upper_)
    device = torch.device("cpu")
    artifact.model.to(device)
    artifact.model.eval()
    latent_dim = artifact.model.encoder[-1].out_features
    z = np.empty((xs.shape[0], latent_dim), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, xs.shape[0], batch_size):
            end = min(start + batch_size, xs.shape[0])
            tensor = torch.tensor(xs[start:end], dtype=torch.float32, device=device)
            z[start:end] = artifact.model.encoder(tensor).detach().cpu().numpy().astype(np.float32, copy=False)
    ref = np.asarray(getattr(artifact, "latent_ref", np.empty((0, latent_dim))), dtype=np.float32)
    if ref.size == 0:
        return np.ones(xs.shape[0], dtype=float)
    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(ref)
    distances, _ = nn.kneighbors(z, return_distance=True)
    d = distances[:, 0].astype(float)
    d_ref = np.asarray(getattr(artifact, "latent_dist_train_sorted", d), dtype=np.float64)
    cutoff = float(np.percentile(d_ref, 99.9)) if len(d_ref) else float(np.percentile(d, 99.9))
    cutoff = max(cutoff, 1e-12)
    return (1.0 / (1.0 + (d / cutoff))).astype(float, copy=False)


def probability_frame(model: Any, x: pd.DataFrame, prefix: str) -> pd.DataFrame:
    estimator = getattr(model, "model", model)
    proba = np.asarray(estimator.predict_proba(x), dtype=float)
    classes = list(getattr(estimator, "classes_", [])) or list(range(proba.shape[1]))
    mapping = getattr(model, "_class_mapping", {}) or {}
    out = pd.DataFrame(index=x.index)
    for idx, cls in enumerate(classes):
        label = str(mapping.get(cls, cls)).strip().lower().replace(" ", "_")
        out[f"{prefix}__prob_{label}"] = proba[:, idx]
    return out


def score_panel(panel: pd.DataFrame, *, artifact_dir: Path, feature_cols: list[str], ae_numeric_cols: list[str]) -> pd.DataFrame:
    clf = read_pickle(artifact_dir / "clf_raw.pkl")
    reg = read_pickle(artifact_dir / "reg_trade_return_raw.pkl")
    ae = read_pickle(artifact_dir / "ae_raw.pkl")
    work = panel.copy()
    for col in feature_cols:
        if col not in work.columns:
            work[col] = 0.0
    x = work[feature_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    scored = work[["close"]].copy()
    probs = probability_frame(clf, x, "clf")
    scored = scored.join(probs)
    if "clf__prob_1" not in scored.columns:
        prob_cols = [col for col in scored.columns if col.startswith("clf__prob_")]
        scored["clf__prob_1"] = scored[prob_cols[-1]] if prob_cols else 0.0
    scored["ranking"] = getattr(reg, "model", reg).predict(x).astype(float)
    scored["ae_familiarity"] = autoencoder_familiarity(ae, work, ae_numeric_cols)
    return enrich_scored_panel(scored)


def enrich_scored_panel(scored: pd.DataFrame) -> pd.DataFrame:
    out = scored.copy()
    out["prob_buy"] = pd.to_numeric(out["clf__prob_1"], errors="coerce").fillna(0.0)
    out["prob_short"] = (1.0 - out["prob_buy"]).clip(0.0, 1.0)
    out["pred_rf_reg"] = pd.to_numeric(out["ranking"], errors="coerce").fillna(0.0)
    out["ae_familiarity"] = pd.to_numeric(out["ae_familiarity"], errors="coerce").fillna(1.0)

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
    out["short_score_mean_raw3"] = out[["prob_short", "pred_rf_reg", "ae_familiarity"]].mean(axis=1, skipna=True)
    out["buy_score_mean_raw_pct6"] = out[
        ["prob_buy", "pred_rf_reg", "ae_familiarity", "prob_buy_pct", "pred_rf_reg_pct", "ae_familiarity_pct"]
    ].mean(axis=1, skipna=True)
    out["short_score_mean_raw_pct6"] = out[
        ["prob_short", "pred_rf_reg", "ae_familiarity", "prob_short_pct", "pred_rf_reg_pct", "ae_familiarity_pct"]
    ].mean(axis=1, skipna=True)
    out["buy_score"] = out["buy_score_raw"]
    out["short_score"] = out["short_score_raw"]
    return out


def pivot(panel: pd.DataFrame, column: str, symbols: list[str]) -> pd.DataFrame:
    work = panel[[column]].reset_index()
    if work.duplicated(["date", "symbol"]).any():
        work = work.sort_values(["date", "symbol"]).groupby(["date", "symbol"], as_index=False, sort=False).last()
    return work.pivot(index="date", columns="symbol", values=column).reindex(columns=symbols).sort_index()


def _nan_float(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def run_top_k_rule(
    panel: pd.DataFrame,
    *,
    score_col: str,
    top_k: int,
    component_threshold: float,
    initial_balance: float,
    fee_bps: float,
    slippage_bps: float,
) -> tuple[pd.Series, pd.Series, pd.Series, dict[str, Any]]:
    replay = replay_trading_app_top_k_rule(
        panel,
        score_col=score_col,
        top_k=top_k,
        component_threshold=component_threshold,
        initial_balance=initial_balance,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
    )
    return replay.equity, replay.returns, replay.cash, replay.meta


def replay_trading_app_top_k_rule(
    panel: pd.DataFrame,
    *,
    score_col: str,
    top_k: int,
    component_threshold: float,
    initial_balance: float,
    fee_bps: float,
    slippage_bps: float,
) -> TradingAppRuleReplay:
    component_cols = [
        "prob_buy",
        "pred_rf_reg",
        "ae_familiarity",
        "prob_buy_pct",
        "pred_rf_reg_pct",
        "ae_familiarity_pct",
    ]
    symbols = sorted(panel.index.get_level_values("symbol").unique())
    score = pivot(panel, score_col, symbols).shift(1)
    close = pivot(panel, "close", symbols).replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    prob_buy = pivot(panel, "prob_buy", symbols).shift(1)
    prob_short = pivot(panel, "prob_short", symbols).shift(1)
    common_dates = score.index.intersection(close.index).intersection(prob_buy.index).intersection(prob_short.index)
    score = score.loc[common_dates].replace([np.inf, -np.inf], np.nan)
    close = close.loc[common_dates]
    prob_buy = prob_buy.loc[common_dates].replace([np.inf, -np.inf], np.nan)
    prob_short = prob_short.loc[common_dates].replace([np.inf, -np.inf], np.nan)
    entry_ok = (score.notna() & np.isfinite(score) & close.gt(0.0)).fillna(False)
    for col in component_cols:
        component = pivot(panel, col, symbols).shift(1).reindex(index=common_dates, columns=symbols)
        entry_ok &= (component.gt(float(component_threshold)) & component.notna() & np.isfinite(component)).fillna(False)

    action_type = np.zeros((len(common_dates), len(symbols)), dtype=int)
    held: set[int] = set()
    symbol_to_idx = {symbol: idx for idx, symbol in enumerate(symbols)}
    positions = pd.DataFrame(0, index=common_dates, columns=symbols, dtype=int)
    for row_idx, dt in enumerate(common_dates):
        price_ok = close.loc[dt].gt(0.0)
        classifier_short = (prob_short.loc[dt] > prob_buy.loc[dt]).fillna(False)
        exits = sorted(
            idx
            for idx in held
            if (not bool(price_ok.iloc[idx]))
            or (not np.isfinite(prob_buy.loc[dt].iloc[idx]))
            or (not np.isfinite(prob_short.loc[dt].iloc[idx]))
            or bool(classifier_short.iloc[idx])
        )
        if exits:
            action_type[row_idx, exits] = 2
            held -= set(exits)
        slots_left = max(0, int(top_k) - len(held))
        if slots_left:
            candidates = entry_ok.loc[dt] & price_ok & (~classifier_short)
            ranked = score.loc[dt][candidates].sort_values(ascending=False, kind="stable")
            entries: list[int] = []
            for symbol in ranked.index:
                idx = symbol_to_idx[str(symbol)]
                if idx in held:
                    continue
                entries.append(idx)
                if len(entries) >= slots_left:
                    break
            if entries:
                action_type[row_idx, entries] = 1
                held |= set(entries)
        if held:
            positions.iloc[row_idx, sorted(held)] = 1

    action_tape = _build_action_tape(
        action_type=action_type,
        close=close,
        score=score,
        prob_buy=prob_buy,
        prob_short=prob_short,
        symbols=symbols,
        top_k=top_k,
    )
    equity, returns, cash, details, executions = discrete_backtest_with_executions(
        action_type=action_type,
        close=close,
        symbol_order=symbols,
        initial_balance=initial_balance,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
    )
    action_tape = _merge_action_tape_executions(action_tape, executions)
    details.update(
        {
            "top_k": int(top_k),
            "component_threshold": float(component_threshold),
            "avg_positions": float((positions > 0).sum(axis=1).mean()) if len(positions) else np.nan,
            "median_positions": float((positions > 0).sum(axis=1).median()) if len(positions) else np.nan,
        }
    )
    trade_windows = action_tape_to_trade_windows(action_tape, prices=close)
    return TradingAppRuleReplay(
        equity=equity,
        returns=returns,
        cash=cash,
        action_tape=action_tape,
        executions=executions,
        positions=positions,
        trade_list=trade_windows,
        meta=details,
    )


def _build_action_tape(
    *,
    action_type: np.ndarray,
    close: pd.DataFrame,
    score: pd.DataFrame,
    prob_buy: pd.DataFrame,
    prob_short: pd.DataFrame,
    symbols: list[str],
    top_k: int,
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
                    "prob_buy": _nan_float(prob_buy.iloc[row_idx, col_idx]),
                    "prob_short": _nan_float(prob_short.iloc[row_idx, col_idx]),
                    "top_k": int(top_k),
                    "reason": "entry_top_k" if action == "buy" else "exit_classifier_or_invalid",
                }
            )
    return pd.DataFrame(rows)


def _merge_action_tape_executions(action_tape: pd.DataFrame, executions: pd.DataFrame) -> pd.DataFrame:
    if action_tape.empty:
        return action_tape.copy()
    out = action_tape.copy()
    if executions.empty:
        out["gross_notional"] = 0.0
        out["fee"] = 0.0
        out["net_cash_flow"] = 0.0
        out["shares_delta"] = 0.0
        return out
    keys = ["date", "symbol", "action"]
    for frame in (out, executions):
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
        frame["symbol"] = frame["symbol"].astype(str).str.upper()
        frame["action"] = frame["action"].astype(str)
    merged = out.merge(
        executions[
            [
                "date",
                "symbol",
                "action",
                "gross_notional",
                "fee",
                "net_cash_flow",
                "shares_delta",
                "shares_after",
                "cash_after",
                "equity_after",
            ]
        ],
        on=keys,
        how="left",
    )
    for col in ("gross_notional", "fee", "net_cash_flow", "shares_delta", "shares_after", "cash_after", "equity_after"):
        merged[col] = pd.to_numeric(merged.get(col), errors="coerce").fillna(0.0)
    return merged


def action_tape_to_trade_windows(action_tape: pd.DataFrame, *, prices: pd.DataFrame | None = None) -> pd.DataFrame:
    if action_tape.empty:
        return pd.DataFrame(
            columns=[
                "trade_id",
                "symbol",
                "side",
                "entry_date",
                "exit_date",
                "entry_price",
                "exit_price",
                "ret_dec",
                "top_k",
                "entry_score",
                "exit_reason",
            ]
        )
    actions = action_tape.copy()
    actions["date"] = pd.to_datetime(actions["date"], errors="coerce").dt.normalize()
    actions["symbol"] = actions["symbol"].astype(str).str.upper()
    actions = actions.dropna(subset=["date", "symbol"]).sort_values(["symbol", "date", "action"]).reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    open_trades: dict[str, dict[str, Any]] = {}
    for row in actions.to_dict("records"):
        symbol = str(row["symbol"])
        action = str(row.get("action", ""))
        if action == "buy":
            if "gross_notional" in actions.columns and _nan_float(row.get("gross_notional")) <= 0.0:
                continue
            if symbol in open_trades:
                open_trades[symbol]["ignored_duplicate_entry"] = True
                continue
            open_trades[symbol] = row
            continue
        if action != "sell" or symbol not in open_trades:
            continue
        entry = open_trades.pop(symbol)
        entry_price = _nan_float(entry.get("price"))
        exit_price = _nan_float(row.get("price"))
        ret_dec = (exit_price / entry_price) - 1.0 if np.isfinite(entry_price) and entry_price > 0 and np.isfinite(exit_price) else np.nan
        rows.append(
            {
                "trade_id": f"{symbol}_{pd.Timestamp(entry['date']).strftime('%Y%m%d')}_{len(rows) + 1}",
                "symbol": symbol,
                "side": "long",
                "entry_date": pd.Timestamp(entry["date"]).normalize(),
                "exit_date": pd.Timestamp(row["date"]).normalize(),
                "entry_price": entry_price,
                "exit_price": exit_price,
                "ret_dec": float(ret_dec) if np.isfinite(ret_dec) else np.nan,
                "equity_entry_notional": _nan_float(entry.get("gross_notional")),
                "equity_entry_fee": _nan_float(entry.get("fee")),
                "equity_entry_shares": _nan_float(entry.get("shares_delta")),
                "equity_exit_notional": _nan_float(row.get("gross_notional")),
                "equity_exit_fee": _nan_float(row.get("fee")),
                "top_k": int(entry.get("top_k", 0)) if pd.notna(entry.get("top_k")) else 0,
                "entry_score": _nan_float(entry.get("score")),
                "exit_reason": row.get("reason", ""),
            }
        )
    if prices is not None and not prices.empty:
        final_date = pd.Timestamp(prices.index.max()).normalize()
        for symbol, entry in sorted(open_trades.items()):
            if symbol not in prices.columns:
                continue
            entry_price = _nan_float(entry.get("price"))
            exit_price = _nan_float(prices.loc[final_date, symbol])
            ret_dec = (exit_price / entry_price) - 1.0 if np.isfinite(entry_price) and entry_price > 0 and np.isfinite(exit_price) else np.nan
            rows.append(
                {
                    "trade_id": f"{symbol}_{pd.Timestamp(entry['date']).strftime('%Y%m%d')}_{len(rows) + 1}",
                    "symbol": symbol,
                    "side": "long",
                    "entry_date": pd.Timestamp(entry["date"]).normalize(),
                    "exit_date": final_date,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "ret_dec": float(ret_dec) if np.isfinite(ret_dec) else np.nan,
                    "equity_entry_notional": _nan_float(entry.get("gross_notional")),
                    "equity_entry_fee": _nan_float(entry.get("fee")),
                    "equity_entry_shares": _nan_float(entry.get("shares_delta")),
                    "equity_exit_notional": np.nan,
                    "equity_exit_fee": np.nan,
                    "top_k": int(entry.get("top_k", 0)) if pd.notna(entry.get("top_k")) else 0,
                    "entry_score": _nan_float(entry.get("score")),
                    "exit_reason": "end_of_backtest",
                }
            )
    return pd.DataFrame(rows).sort_values(["entry_date", "symbol"]).reset_index(drop=True) if rows else pd.DataFrame()


def trade_cost(gross: float, fee_bps: float, slippage_bps: float) -> float:
    return float(gross) * ((float(fee_bps) + float(slippage_bps)) / 10000.0)


def discrete_backtest(
    *,
    action_type: np.ndarray,
    close: pd.DataFrame,
    initial_balance: float,
    fee_bps: float,
    slippage_bps: float,
) -> tuple[pd.Series, pd.Series, pd.Series, dict[str, Any]]:
    equity, returns, cash, details, _executions = discrete_backtest_with_executions(
        action_type=action_type,
        close=close,
        symbol_order=list(close.columns.astype(str)),
        initial_balance=initial_balance,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
    )
    return equity, returns, cash, details


def discrete_backtest_with_executions(
    *,
    action_type: np.ndarray,
    close: pd.DataFrame,
    symbol_order: list[str],
    initial_balance: float,
    fee_bps: float,
    slippage_bps: float,
) -> tuple[pd.Series, pd.Series, pd.Series, dict[str, Any], pd.DataFrame]:
    prices = close.to_numpy(dtype=float)
    balance = float(initial_balance)
    shares = np.zeros(prices.shape[1], dtype=float)
    cost_rate = (float(fee_bps) + float(slippage_bps)) / 10000.0
    equity = np.zeros(prices.shape[0], dtype=float)
    cash = np.zeros(prices.shape[0], dtype=float)
    buy_count = 0
    sell_count = 0
    execution_rows: list[dict[str, Any]] = []
    for t in range(prices.shape[0]):
        date = pd.Timestamp(close.index[t]).normalize()
        px = prices[t]
        actions = np.asarray(action_type[t], dtype=int)
        for j in np.where(actions == 2)[0]:
            if px[j] <= 0 or shares[j] <= 0:
                continue
            shares_delta = -float(shares[j])
            gross = shares[j] * px[j]
            fee = trade_cost(gross, fee_bps, slippage_bps)
            balance += gross - fee
            shares[j] = 0.0
            sell_count += 1
            execution_rows.append(
                {
                    "date": date,
                    "symbol": symbol_order[j],
                    "action": "sell",
                    "price": float(px[j]),
                    "gross_notional": float(gross),
                    "fee": float(fee),
                    "net_cash_flow": float(gross - fee),
                    "shares_delta": shares_delta,
                    "shares_after": float(shares[j]),
                    "cash_after": float(balance),
                    "equity_after": float(balance + np.sum(shares * px)),
                }
            )
        net_worth = balance + float(np.sum(shares * px))
        buy_idx = np.where(actions == 1)[0]
        day_buy_budget = max(0.0, balance)
        target_notional = net_worth / float(len(buy_idx)) if len(buy_idx) else 0.0
        for j in buy_idx:
            if px[j] <= 0 or day_buy_budget <= 0:
                continue
            need_gross = max(0.0, target_notional - (shares[j] * px[j]))
            gross = min(need_gross, day_buy_budget / (1.0 + cost_rate))
            if gross <= 0:
                continue
            fee = trade_cost(gross, fee_bps, slippage_bps)
            shares_delta = gross / px[j]
            shares[j] += shares_delta
            day_buy_budget -= gross + fee
            buy_count += 1
            execution_rows.append(
                {
                    "date": date,
                    "symbol": symbol_order[j],
                    "action": "buy",
                    "price": float(px[j]),
                    "gross_notional": float(gross),
                    "fee": float(fee),
                    "net_cash_flow": float(-(gross + fee)),
                    "shares_delta": float(shares_delta),
                    "shares_after": float(shares[j]),
                    "cash_after": float(day_buy_budget),
                    "equity_after": float(day_buy_budget + np.sum(shares * px)),
                }
            )
        balance = day_buy_budget if day_buy_budget > 1e-9 else 0.0
        equity[t] = balance + float(np.sum(shares * px))
        cash[t] = balance
    eq = pd.Series(equity, index=close.index, name="equity")
    ret = eq.pct_change().fillna(0.0).rename("returns")
    cash_s = pd.Series(cash, index=close.index, name="cash")
    executions = pd.DataFrame(execution_rows)
    return (
        eq,
        ret,
        cash_s,
        {"executed_buy_count": buy_count, "executed_sell_count": sell_count, "cash_end": float(cash_s.iloc[-1]) if len(cash_s) else np.nan},
        executions,
    )


def summarize_returns(returns: pd.Series, initial_balance: float) -> dict[str, Any]:
    clean = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    equity = (1.0 + clean).cumprod()
    yearly = []
    for year in sorted({int(ts.year) for ts in clean.index}):
        part = clean.loc[(clean.index >= pd.Timestamp(f"{year}-01-01")) & (clean.index <= pd.Timestamp(f"{year}-12-31"))]
        yeq = (1.0 + part).cumprod()
        yearly.append(
            {
                "year": year,
                "total_return_pct": float((yeq.iloc[-1] - 1.0) * 100.0) if len(yeq) else np.nan,
                "sharpe": float((part.mean() / part.std(ddof=0)) * np.sqrt(252.0)) if len(part) and part.std(ddof=0) > 1e-12 else np.nan,
                "max_drawdown_pct": float((((yeq / yeq.cummax()) - 1.0).min()) * 100.0) if len(yeq) else np.nan,
                "trading_days": int(len(part)),
            }
        )
    return {
        "start_date": clean.index.min().date().isoformat() if len(clean) else None,
        "end_date": clean.index.max().date().isoformat() if len(clean) else None,
        "trading_days": int(len(clean)),
        "initial_balance": float(initial_balance),
        "final_equity": float(initial_balance * equity.iloc[-1]) if len(equity) else np.nan,
        "total_return_pct": float((equity.iloc[-1] - 1.0) * 100.0) if len(equity) else np.nan,
        "total_return_multiple": float(equity.iloc[-1]) if len(equity) else np.nan,
        "sharpe": float((clean.mean() / clean.std(ddof=0)) * np.sqrt(252.0)) if len(clean) and clean.std(ddof=0) > 1e-12 else np.nan,
        "max_drawdown_pct": float((((equity / equity.cummax()) - 1.0).min()) * 100.0) if len(equity) else np.nan,
        "yearly": yearly,
    }


def replay_option_portfolio_from_selected_paths(
    selected_option_trades: pd.DataFrame,
    selected_option_paths: pd.DataFrame,
    *,
    date_index: pd.Index,
    initial_balance: float,
) -> OptionPortfolioReplay:
    dates = pd.DatetimeIndex(pd.to_datetime(date_index, errors="coerce")).normalize()
    dates = pd.DatetimeIndex(sorted(dates[~dates.isna()].unique()))
    if len(dates) == 0:
        empty = pd.Series(dtype=float)
        return OptionPortfolioReplay(empty, empty, empty, empty, empty, pd.DataFrame(), {})
    cash_flows = pd.Series(0.0, index=dates)
    active_value = pd.Series(0.0, index=dates)
    capital_deployed = pd.Series(0.0, index=dates)
    ledger_rows: list[dict[str, Any]] = []
    if selected_option_trades is None or selected_option_trades.empty:
        equity = pd.Series(float(initial_balance), index=dates, name="option_equity")
        returns = equity.pct_change().fillna(0.0).rename("option_returns")
        cash = pd.Series(float(initial_balance), index=dates, name="option_cash")
        summary = summarize_returns(returns, float(initial_balance))
        return OptionPortfolioReplay(equity, returns, cash, active_value, capital_deployed, pd.DataFrame(), summary)

    paths = selected_option_paths.copy() if selected_option_paths is not None else pd.DataFrame()
    if not paths.empty:
        paths["snapshot_date"] = pd.to_datetime(paths["snapshot_date"], errors="coerce").dt.normalize()
        paths["trade_id"] = paths["trade_id"].astype(str)
        for col in ("mark_price", "bid", "mid"):
            if col in paths.columns:
                paths[col] = pd.to_numeric(paths[col], errors="coerce")

    trades = selected_option_trades.copy()
    trades["trade_id"] = trades["trade_id"].astype(str)
    trades["entry_date"] = pd.to_datetime(trades["entry_date"], errors="coerce").dt.normalize()
    trades["option_exit_date"] = pd.to_datetime(trades["option_exit_date"], errors="coerce").dt.normalize()
    for _, trade in trades.iterrows():
        trade_id = str(trade.get("trade_id"))
        budget = _nan_float(trade.get("equity_entry_notional"))
        entry_price = _nan_float(trade.get("entry_price"))
        exit_price = _nan_float(trade.get("exit_price"))
        entry_date = pd.Timestamp(trade.get("entry_date")).normalize()
        exit_date = pd.Timestamp(trade.get("option_exit_date")).normalize()
        if (
            not np.isfinite(budget)
            or budget <= 0.0
            or not np.isfinite(entry_price)
            or entry_price <= 0.0
            or not np.isfinite(exit_price)
            or pd.isna(entry_date)
            or pd.isna(exit_date)
        ):
            continue
        entry_dt = _align_to_calendar(entry_date, dates, direction="forward")
        exit_dt = _align_to_calendar(exit_date, dates, direction="backward")
        if entry_dt is None or exit_dt is None or exit_dt < entry_dt:
            continue
        units = float(budget) / float(entry_price)
        cash_flows.loc[entry_dt] -= float(budget)
        exit_value = float(units) * float(exit_price)
        cash_flows.loc[exit_dt] += exit_value
        trade_path = paths.loc[paths["trade_id"].eq(trade_id)].copy() if not paths.empty else pd.DataFrame()
        if trade_path.empty:
            mark_dates = pd.DatetimeIndex([entry_dt])
            marks = pd.Series(float(entry_price), index=mark_dates)
        else:
            mark_col = "mark_price" if "mark_price" in trade_path.columns else "bid" if "bid" in trade_path.columns else "mid"
            marks = (
                trade_path.dropna(subset=["snapshot_date"])
                .sort_values("snapshot_date")
                .drop_duplicates("snapshot_date", keep="last")
                .set_index("snapshot_date")[mark_col]
                .astype(float)
            )
        active_dates = dates[(dates >= entry_dt) & (dates < exit_dt)]
        if len(active_dates):
            aligned_marks = marks.reindex(active_dates).ffill()
            if aligned_marks.isna().all():
                aligned_marks = pd.Series(float(entry_price), index=active_dates)
            else:
                aligned_marks = aligned_marks.fillna(float(entry_price))
            active_value.loc[active_dates] += aligned_marks * float(units)
            capital_deployed.loc[active_dates] += float(budget)
        ledger_rows.append(
            {
                "trade_id": trade_id,
                "symbol": trade.get("symbol"),
                "entry_date": entry_dt,
                "option_exit_date": exit_dt,
                "equity_entry_notional": float(budget),
                "entry_price": float(entry_price),
                "exit_price": float(exit_price),
                "option_units": float(units),
                "exit_value": float(exit_value),
                "option_pnl_dollars": float(exit_value - budget),
                "option_return": float(exit_value / budget - 1.0),
                "expired_before_equity_exit": bool(trade.get("expired_before_equity_exit", False)),
            }
        )
    cash = (float(initial_balance) + cash_flows.cumsum()).rename("option_cash")
    equity = (cash + active_value).rename("option_equity")
    returns = equity.pct_change().fillna(0.0).rename("option_returns")
    ledger = pd.DataFrame(ledger_rows)
    summary = summarize_returns(returns, float(initial_balance))
    summary.update(
        {
            "funded_option_trades": int(len(ledger)),
            "total_premium_deployed": float(ledger["equity_entry_notional"].sum()) if not ledger.empty else 0.0,
            "total_option_pnl_dollars": float(ledger["option_pnl_dollars"].sum()) if not ledger.empty else 0.0,
        }
    )
    return OptionPortfolioReplay(
        equity=equity,
        returns=returns,
        cash=cash,
        active_value=active_value.rename("option_active_value"),
        capital_deployed=capital_deployed.rename("option_capital_deployed"),
        trade_ledger=ledger,
        summary=summary,
    )


def _align_to_calendar(target: pd.Timestamp, dates: pd.DatetimeIndex, *, direction: str) -> pd.Timestamp | None:
    target = pd.Timestamp(target).normalize()
    if direction == "forward":
        eligible = dates[dates >= target]
        return pd.Timestamp(eligible[0]) if len(eligible) else None
    eligible = dates[dates <= target]
    return pd.Timestamp(eligible[-1]) if len(eligible) else None


def compare_latest_scores(scored: pd.DataFrame, artifact_dir: Path, score_col: str) -> dict[str, Any]:
    stored_path = artifact_dir / "latest_scored_latest.pkl"
    if not stored_path.exists() or scored.empty:
        return {}
    stored = pd.read_pickle(stored_path)
    stored.index = pd.Index(stored.index.astype(str).str.upper(), name="symbol")
    latest_date = scored.index.get_level_values("date").max()
    latest = scored.xs(latest_date, level="date").copy()
    latest.index = pd.Index(latest.index.astype(str).str.upper(), name="symbol")
    common = sorted(set(latest.index).intersection(set(stored.index)))
    if not common:
        return {"latest_date": str(latest_date.date()), "common_symbols": 0}
    rows: dict[str, Any] = {"latest_date": str(latest_date.date()), "common_symbols": int(len(common))}
    for col in ("prob_buy", "pred_rf_reg", "ae_familiarity", score_col):
        if col not in latest.columns or col not in stored.columns:
            continue
        left = pd.to_numeric(latest.loc[common, col], errors="coerce")
        right = pd.to_numeric(stored.loc[common, col], errors="coerce")
        valid = left.notna() & right.notna()
        if valid.any():
            rows[f"{col}_corr"] = float(left[valid].corr(right[valid]))
            rows[f"{col}_mae"] = float((left[valid] - right[valid]).abs().mean())
    if score_col in latest.columns and score_col in stored.columns:
        rebuilt_top = set(latest[score_col].sort_values(ascending=False).head(20).index)
        stored_top = set(stored[score_col].sort_values(ascending=False).head(20).index)
        rows["top20_overlap"] = int(len(rebuilt_top.intersection(stored_top)))
    return rows


def run_optimal_trader_artifact_replay(config: OptimalTraderArtifactReplayConfig) -> OptimalTraderArtifactReplayResult:
    artifact_dir = Path(config.artifact_dir).expanduser().resolve()
    out_dir = Path(config.output_dir).expanduser().resolve() if config.output_dir is not None else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
    meta = json.loads((artifact_dir / "meta.json").read_text(encoding="utf-8"))
    leaderboard_meta = json.loads((artifact_dir / "leaderboard_latest_meta.json").read_text(encoding="utf-8"))
    feature_cols = list(map(str, meta["feature_list"]))
    ae_numeric_cols = list(map(str, meta.get("ae_numeric_cols", feature_cols)))
    latest_scored = pd.read_pickle(artifact_dir / "latest_scored_latest.pkl")
    symbols = sorted(str(symbol).strip().upper() for symbol in latest_scored.index if str(symbol).strip())
    if config.max_symbols > 0:
        symbols = symbols[: config.max_symbols]
    score_col = str(leaderboard_meta.get("config", {}).get("strategy", {}).get("score_col", "buy_score_mean_raw_pct6"))
    fee_bps = float(leaderboard_meta.get("config", {}).get("costs", {}).get("fee_bps", 5.0))
    slippage_bps = float(leaderboard_meta.get("config", {}).get("costs", {}).get("slippage_bps", 5.0))

    started = perf_counter()
    feature_path = out_dir / "feature_panel.parquet" if out_dir is not None else None
    scored_path = out_dir / "scored_panel.parquet" if out_dir is not None else None
    if config.reuse_feature_panel and feature_path is not None and feature_path.exists():
        feature_panel = pd.read_parquet(feature_path)
        feature_panel["date"] = pd.to_datetime(feature_panel["date"])
        feature_panel["symbol"] = feature_panel["symbol"].astype(str)
        feature_panel = feature_panel.set_index(["date", "symbol"]).sort_index()
        feature_meta = {"reused_feature_panel": True}
    else:
        feature_panel, feature_meta = build_feature_panel(
            symbols,
            feature_cols=feature_cols,
            start=str(config.feature_start),
            end=str(config.end_date),
            price_provider=str(config.price_provider),
        )
        if feature_path is not None:
            feature_panel.reset_index().to_parquet(feature_path, index=False)

    score_features = feature_panel.loc[
        (feature_panel.index.get_level_values("date") >= pd.Timestamp(config.backtest_start))
        & (feature_panel.index.get_level_values("date") <= pd.Timestamp(config.end_date))
    ].copy()
    if config.reuse_scored_panel and scored_path is not None and scored_path.exists():
        scored = pd.read_parquet(scored_path)
        scored["date"] = pd.to_datetime(scored["date"])
        scored["symbol"] = scored["symbol"].astype(str)
        scored = scored.set_index(["date", "symbol"]).sort_index()
    else:
        scored = score_panel(score_features, artifact_dir=artifact_dir, feature_cols=feature_cols, ae_numeric_cols=ae_numeric_cols)
        if scored_path is not None:
            scored.reset_index().to_parquet(scored_path, index=False)

    rule_replay = replay_trading_app_top_k_rule(
        scored,
        score_col=score_col,
        top_k=int(config.top_k),
        component_threshold=float(config.component_threshold),
        initial_balance=float(config.initial_balance),
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
    )
    option_execution = None
    option_portfolio = None
    option_summary: dict[str, Any] = {}
    if bool(config.run_fmp_synthetic_options):
        from quant_orchestrator.platforms.backtesting_frameworks.optimal_trader.synthetic_options import (
            FmpSyntheticOptionReplayConfig,
            run_fmp_synthetic_option_trade_replay,
        )

        option_execution = run_fmp_synthetic_option_trade_replay(
            rule_replay.trade_list,
            config=FmpSyntheticOptionReplayConfig(workers=max(1, int(config.option_workers))),
        )
        option_portfolio = replay_option_portfolio_from_selected_paths(
            option_execution.selected_option_trades,
            option_execution.selected_option_paths,
            date_index=rule_replay.equity.index,
            initial_balance=float(config.initial_balance),
        )
        option_return_summary = summarize_option_trade_returns(option_execution.selected_option_trades)
        option_summary = {
            "selected_option_trades": int(len(option_execution.selected_option_trades)),
            "selected_option_paths": int(len(option_execution.selected_option_paths)),
            "trade_status_counts": (
                option_execution.trade_status["status"].value_counts().to_dict()
                if not option_execution.trade_status.empty and "status" in option_execution.trade_status
                else {}
            ),
            "option_trade_returns": option_return_summary,
            "option_portfolio": option_portfolio.summary,
            "metrics": option_execution.metrics,
        }

    summary = {
        "artifact_dir": str(artifact_dir),
        "feature_source": "quant_warehouse",
        "model_source": "optimal_trader_saved_raw_stack_artifacts",
        "optimal_trader_imported": False,
        "symbols": int(len(symbols)),
        "feature_start": str(config.feature_start),
        "backtest_start": str(config.backtest_start),
        "end_date": str(config.end_date),
        "score_col": score_col,
        "top_k": int(config.top_k),
        "component_threshold": float(config.component_threshold),
        "fee_bps": fee_bps,
        "slippage_bps": slippage_bps,
        "elapsed_seconds": perf_counter() - started,
        "feature_meta": feature_meta,
        "scored_rows": int(len(scored)),
        "latest_score_validation": compare_latest_scores(scored, artifact_dir, score_col),
        "rule_meta": rule_replay.meta,
        "trade_list": int(len(rule_replay.trade_list)),
        "fmp_synthetic_options": option_summary,
        "performance": summarize_returns(rule_replay.returns, float(config.initial_balance)),
    }
    result = OptimalTraderArtifactReplayResult(
        feature_panel=feature_panel,
        scored_panel=scored,
        rule_replay=rule_replay,
        summary=summary,
        option_execution=option_execution,
        option_portfolio=option_portfolio,
    )
    if out_dir is not None:
        write_artifact_replay_outputs(result, out_dir)
    return result


def write_artifact_replay_outputs(result: OptimalTraderArtifactReplayResult, output_dir: str | Path) -> dict[str, Path]:
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "equity_curve": out_dir / "equity_curve.csv",
        "returns": out_dir / "returns.csv",
        "cash": out_dir / "cash.csv",
        "action_tape": out_dir / "action_tape.parquet",
        "positions": out_dir / "positions.parquet",
        "trade_list": out_dir / "trade_list.parquet",
        "summary": out_dir / "summary.json",
        "equity_executions": out_dir / "equity_executions.csv",
    }
    if result.option_execution is not None:
        paths.update(
            {
                "selected_option_trades": out_dir / "selected_option_trades.parquet",
                "selected_option_paths": out_dir / "selected_option_paths.parquet",
                "option_trade_status": out_dir / "option_trade_status.csv",
            }
        )
    if result.option_portfolio is not None:
        paths.update(
            {
                "option_equity_curve": out_dir / "option_equity_curve.csv",
                "option_cash": out_dir / "option_cash.csv",
                "option_active_value": out_dir / "option_active_value.csv",
                "option_capital_deployed": out_dir / "option_capital_deployed.csv",
                "option_portfolio_ledger": out_dir / "option_portfolio_ledger.parquet",
            }
        )
    result.rule_replay.equity.rename("equity").to_csv(paths["equity_curve"])
    result.rule_replay.returns.to_csv(paths["returns"])
    result.rule_replay.cash.to_csv(paths["cash"])
    result.rule_replay.action_tape.to_parquet(paths["action_tape"], index=False)
    result.rule_replay.executions.to_csv(paths["equity_executions"], index=False)
    result.rule_replay.positions.reset_index(names="date").to_parquet(paths["positions"], index=False)
    result.rule_replay.trade_list.to_parquet(paths["trade_list"], index=False)
    standard_paths = write_strategy_artifacts(
        StrategyArtifactBundle(
            feature_panel=result.feature_panel,
            scored_panel=result.scored_panel,
            action_tape=result.rule_replay.action_tape,
            trade_list=result.rule_replay.trade_list,
            summary=result.summary,
            strategy_name="optimal_trader.trading_app",
        ),
        out_dir,
        extra_paths=paths,
    )
    paths.update({f"standard_{key}": value for key, value in standard_paths.items()})
    if result.option_execution is not None:
        result.option_execution.selected_option_trades.to_parquet(paths["selected_option_trades"], index=False)
        result.option_execution.selected_option_paths.to_parquet(paths["selected_option_paths"], index=False)
        result.option_execution.trade_status.to_csv(paths["option_trade_status"], index=False)
    if result.option_portfolio is not None:
        result.option_portfolio.equity.to_csv(paths["option_equity_curve"])
        result.option_portfolio.cash.to_csv(paths["option_cash"])
        result.option_portfolio.active_value.to_csv(paths["option_active_value"])
        result.option_portfolio.capital_deployed.to_csv(paths["option_capital_deployed"])
        result.option_portfolio.trade_ledger.to_parquet(paths["option_portfolio_ledger"], index=False)
    paths["summary"].write_text(json.dumps(result.summary, indent=2, default=str), encoding="utf-8")
    return paths


def summarize_option_trade_returns(selected_option_trades: pd.DataFrame) -> dict[str, Any]:
    if selected_option_trades is None or selected_option_trades.empty or "option_return" not in selected_option_trades:
        return {}
    returns = pd.to_numeric(selected_option_trades["option_return"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if returns.empty:
        return {"count": 0}
    by_year: list[dict[str, Any]] = []
    if "entry_date" in selected_option_trades:
        frame = selected_option_trades.copy()
        frame["entry_date"] = pd.to_datetime(frame["entry_date"], errors="coerce")
        frame["option_return"] = pd.to_numeric(frame["option_return"], errors="coerce")
        for year, part in frame.dropna(subset=["entry_date", "option_return"]).groupby(frame["entry_date"].dt.year):
            values = part["option_return"].dropna()
            by_year.append(
                {
                    "year": int(year),
                    "count": int(len(values)),
                    "mean_return_pct": float(values.mean() * 100.0),
                    "median_return_pct": float(values.median() * 100.0),
                    "win_rate_pct": float(values.gt(0.0).mean() * 100.0),
                }
            )
    return {
        "count": int(len(returns)),
        "mean_return_pct": float(returns.mean() * 100.0),
        "median_return_pct": float(returns.median() * 100.0),
        "win_rate_pct": float(returns.gt(0.0).mean() * 100.0),
        "p10_return_pct": float(returns.quantile(0.10) * 100.0),
        "p90_return_pct": float(returns.quantile(0.90) * 100.0),
        "max_return_pct": float(returns.max() * 100.0),
        "min_return_pct": float(returns.min() * 100.0),
        "by_entry_year": by_year,
    }
