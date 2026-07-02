from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from quant_orchestrator.platforms.ml_frameworks.torch.runtime import configure_torch_runtime


@dataclass(frozen=True)
class LatentAutoencoderConfig:
    epochs: int = 20
    tuning_epochs: int = 8
    batch_size: int = 4096
    lr: float = 1e-3
    weight_decay: float = 1e-4
    denoise_std: float = 0.02
    familiarity_quantile: float = 99.9
    nn_metric: str = "euclidean"
    require_cuda: bool = False
    device: str | None = None
    tune_architecture: bool = True
    validation_fraction: float = 0.15
    min_validation_rows: int = 256
    min_hidden_dim: int = 2
    max_hidden_dim: int = 128
    min_bottleneck_dim: int = 1
    max_bottleneck_dim: int = 32
    max_architecture_candidates: int = 8
    hidden_multipliers: tuple[float, ...] = (0.75, 1.0, 1.5, 2.0)
    bottleneck_ratios: tuple[float, ...] = (0.125, 0.25, 0.5)
    architecture_selection_label_col: str = "collapsed_label"
    architecture_selection_metric: str = "latent_label_purity"
    latent_neighbor_count: int = 5
    latent_dim_penalty: float = 0.02


def create_family_autoencoder(in_dim: int, hidden_dim: int, bottleneck_dim: int):
    import torch.nn as nn

    class FamilyAutoEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, bottleneck_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(bottleneck_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, in_dim),
            )

        def forward(self, x):
            return self.decoder(self.encoder(x))

    return FamilyAutoEncoder()


@dataclass
class LatentAutoencoderIndex:
    """Denoising autoencoder plus an in-memory latent nearest-neighbor index."""

    model: object
    features: list[str]
    center: np.ndarray
    scale: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    nn_index: NearestNeighbors
    device: str
    config: LatentAutoencoderConfig
    latent_index_rows: int
    latent_distance_cutoff: float
    train_latent_distance_mean: float
    train_latent_distance_p95: float
    train_error_mean: float
    train_error_p95: float
    hidden_dim: int
    bottleneck_dim: int
    architecture_diagnostics: list[dict[str, float | int | str]]

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        *,
        features: Iterable[str],
        config: LatentAutoencoderConfig | None = None,
    ) -> "LatentAutoencoderIndex | None":
        import torch

        cfg = config or LatentAutoencoderConfig()
        feature_list = list(features)
        runtime = configure_torch_runtime(require_cuda=cfg.require_cuda, device=cfg.device)
        device = runtime.torch_device
        x_train, center, scale, lower, upper = _standardize_fit(frame, feature_list)
        if len(x_train) < 2:
            return None

        in_dim = x_train.shape[1]
        candidates = _candidate_architectures(in_dim, cfg)
        hidden_dim, bottleneck_dim, architecture_diagnostics = _select_architecture(
            x_train,
            labels=_architecture_labels(frame, cfg),
            candidates=candidates,
            cfg=cfg,
            device=device,
        )
        model = _train_autoencoder_model(
            x_train,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
            cfg=cfg,
            device=device,
            epochs=cfg.epochs,
        )
        if model is None:
            return None

        model.eval()
        train_error = _autoencoder_recon_error(model, x_train, device=device, batch_size=cfg.batch_size)
        train_latent = _autoencoder_latent(model, x_train, device=device, batch_size=cfg.batch_size)
        nn_index = NearestNeighbors(n_neighbors=1, metric=cfg.nn_metric)
        nn_index.fit(train_latent)
        calibration_neighbors = min(2, len(train_latent))
        calibration_index = NearestNeighbors(n_neighbors=calibration_neighbors, metric=cfg.nn_metric)
        calibration_index.fit(train_latent)
        calibration_distance, _ = calibration_index.kneighbors(train_latent, return_distance=True)
        train_latent_distance = calibration_distance[:, 1] if calibration_neighbors > 1 else calibration_distance[:, 0]
        latent_cutoff = max(float(np.nanpercentile(train_latent_distance, cfg.familiarity_quantile)), 1e-12)
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
        return cls(
            model=model,
            features=feature_list,
            center=center,
            scale=scale,
            lower=lower,
            upper=upper,
            nn_index=nn_index,
            device=device,
            config=cfg,
            latent_index_rows=int(len(train_latent)),
            latent_distance_cutoff=latent_cutoff,
            train_latent_distance_mean=float(np.nanmean(train_latent_distance)),
            train_latent_distance_p95=float(np.nanpercentile(train_latent_distance, 95.0)),
            train_error_mean=float(np.nanmean(train_error)),
            train_error_p95=float(np.nanpercentile(train_error, 95.0)),
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
            architecture_diagnostics=architecture_diagnostics,
        )

    def familiarity(self, frame: pd.DataFrame) -> pd.DataFrame:
        x = _standardize_apply(frame, self.features, self.center, self.scale, self.lower, self.upper)
        err = _autoencoder_recon_error(self.model, x, device=self.device, batch_size=self.config.batch_size)
        latent = _autoencoder_latent(self.model, x, device=self.device, batch_size=self.config.batch_size)
        latent_distance, _ = self.nn_index.kneighbors(latent, n_neighbors=1, return_distance=True)
        latent_distance = latent_distance[:, 0].astype("float64")
        familiarity = 1.0 / (1.0 + (latent_distance / self.latent_distance_cutoff))
        return pd.DataFrame(
            {
                "ae_familiarity": np.clip(familiarity, 0.0, 1.0).astype("float64"),
                "ae_recon_error": err.astype("float64"),
                "ae_latent_distance": latent_distance,
            },
            index=frame.index,
        )

    def metadata(self) -> dict[str, float | int | str]:
        selected = next((row for row in self.architecture_diagnostics if row.get("selected") == 1), {})
        return {
            "ae_latent_index_rows": self.latent_index_rows,
            "ae_latent_distance_cutoff": self.latent_distance_cutoff,
            "ae_train_latent_distance_mean": self.train_latent_distance_mean,
            "ae_train_latent_distance_p95": self.train_latent_distance_p95,
            "ae_train_error_mean": self.train_error_mean,
            "ae_train_error_p95": self.train_error_p95,
            "ae_bottleneck_dim": self.bottleneck_dim,
            "ae_hidden_dim": self.hidden_dim,
            "ae_device": self.device,
            "ae_nn_metric": self.config.nn_metric,
            "ae_familiarity_quantile": self.config.familiarity_quantile,
            "ae_architecture_tuned": int(bool(self.config.tune_architecture)),
            "ae_architecture_candidates": len(self.architecture_diagnostics),
            "ae_architecture_selection_metric": str(selected.get("selection_metric", "")),
            "ae_architecture_selection_score": float(selected.get("selection_score", np.nan)),
            "ae_validation_latent_label_purity": float(selected.get("validation_latent_label_purity", np.nan)),
            "ae_latent_dim_penalty": float(selected.get("latent_dim_penalty", np.nan)),
            "ae_validation_rows": int(selected.get("validation_rows", 0) or 0),
            "ae_validation_error_mean": float(selected.get("validation_error_mean", np.nan)),
            "ae_validation_error_p95": float(selected.get("validation_error_p95", np.nan)),
        }


def _train_autoencoder_model(
    x_train: np.ndarray,
    *,
    hidden_dim: int,
    bottleneck_dim: int,
    cfg: LatentAutoencoderConfig,
    device: str,
    epochs: int,
):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    if len(x_train) < 2:
        return None
    model = create_family_autoencoder(x_train.shape[1], hidden_dim, bottleneck_dim).to(device)
    loader = DataLoader(
        TensorDataset(torch.tensor(x_train, dtype=torch.float32)),
        batch_size=cfg.batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(epochs)))
    loss_fn = nn.MSELoss()
    model.train()
    for _epoch in range(max(1, int(epochs))):
        for (batch,) in loader:
            batch = batch.to(device)
            noisy = batch + torch.randn_like(batch) * cfg.denoise_std if cfg.denoise_std > 0 else batch
            optimizer.zero_grad(set_to_none=True)
            recon = model(noisy)
            loss = loss_fn(recon, batch)
            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        scheduler.step()
    return model


def _select_architecture(
    x_train: np.ndarray,
    *,
    labels: np.ndarray | None,
    candidates: list[tuple[int, int]],
    cfg: LatentAutoencoderConfig,
    device: str,
) -> tuple[int, int, list[dict[str, float | int | str]]]:
    if not candidates:
        raise ValueError("No autoencoder architecture candidates were generated.")
    if not cfg.tune_architecture or len(candidates) == 1:
        hidden_dim, bottleneck_dim = candidates[0]
        return hidden_dim, bottleneck_dim, [
            {
                "hidden_dim": int(hidden_dim),
                "bottleneck_dim": int(bottleneck_dim),
                "validation_rows": 0,
                "validation_error_mean": float("nan"),
                "validation_error_p95": float("nan"),
                "validation_latent_label_purity": float("nan"),
                "latent_dim_penalty": _latent_dimension_penalty(bottleneck_dim, x_train.shape[1], cfg),
                "selection_score": float("nan"),
                "selection_metric": "none",
                "selected": 1,
                "selection_reason": "single_candidate_or_tuning_disabled",
            }
        ]

    fit_x, valid_x, fit_labels, valid_labels = _tail_validation_split(x_train, labels, cfg)
    if len(valid_x) == 0 or len(fit_x) < 2:
        hidden_dim, bottleneck_dim = candidates[0]
        return hidden_dim, bottleneck_dim, [
            {
                "hidden_dim": int(hidden_dim),
                "bottleneck_dim": int(bottleneck_dim),
                "validation_rows": 0,
                "validation_error_mean": float("nan"),
                "validation_error_p95": float("nan"),
                "validation_latent_label_purity": float("nan"),
                "latent_dim_penalty": _latent_dimension_penalty(bottleneck_dim, x_train.shape[1], cfg),
                "selection_score": float("nan"),
                "selection_metric": "none",
                "selected": 1,
                "selection_reason": "insufficient_validation_rows",
            }
        ]

    diagnostics: list[dict[str, float | int | str]] = []
    for hidden_dim, bottleneck_dim in candidates:
        model = _train_autoencoder_model(
            fit_x,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
            cfg=cfg,
            device=device,
            epochs=cfg.tuning_epochs,
        )
        if model is None:
            continue
        valid_error = _autoencoder_recon_error(model, valid_x, device=device, batch_size=cfg.batch_size)
        latent_purity = _validation_latent_label_purity(
            model,
            fit_x,
            valid_x,
            fit_labels=fit_labels,
            valid_labels=valid_labels,
            cfg=cfg,
            device=device,
        )
        penalty = _latent_dimension_penalty(bottleneck_dim, x_train.shape[1], cfg)
        selection_metric, selection_score = _architecture_selection_score(
            validation_error_mean=float(np.nanmean(valid_error)),
            latent_label_purity=latent_purity,
            latent_dim_penalty=penalty,
            cfg=cfg,
        )
        diagnostics.append(
            {
                "hidden_dim": int(hidden_dim),
                "bottleneck_dim": int(bottleneck_dim),
                "validation_rows": int(len(valid_x)),
                "validation_error_mean": float(np.nanmean(valid_error)),
                "validation_error_p95": float(np.nanpercentile(valid_error, 95.0)),
                "validation_latent_label_purity": float(latent_purity),
                "latent_dim_penalty": float(penalty),
                "selection_score": float(selection_score),
                "selection_metric": selection_metric,
                "selected": 0,
                "selection_reason": "candidate",
            }
        )
    if not diagnostics:
        hidden_dim, bottleneck_dim = candidates[0]
        return hidden_dim, bottleneck_dim, [
            {
                "hidden_dim": int(hidden_dim),
                "bottleneck_dim": int(bottleneck_dim),
                "validation_rows": int(len(valid_x)),
                "validation_error_mean": float("nan"),
                "validation_error_p95": float("nan"),
                "validation_latent_label_purity": float("nan"),
                "latent_dim_penalty": _latent_dimension_penalty(bottleneck_dim, x_train.shape[1], cfg),
                "selection_score": float("nan"),
                "selection_metric": "none",
                "selected": 1,
                "selection_reason": "candidate_training_failed",
            }
        ]
    best_index = _best_architecture_index(diagnostics)
    diagnostics[best_index]["selected"] = 1
    diagnostics[best_index]["selection_reason"] = str(diagnostics[best_index]["selection_metric"])
    return (
        int(diagnostics[best_index]["hidden_dim"]),
        int(diagnostics[best_index]["bottleneck_dim"]),
        diagnostics,
    )


def _tail_validation_split(
    x_train: np.ndarray,
    labels: np.ndarray | None,
    cfg: LatentAutoencoderConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    rows = int(len(x_train))
    if rows < 4:
        return x_train, np.empty((0, x_train.shape[1]), dtype=x_train.dtype), labels, None
    requested = max(int(cfg.min_validation_rows), int(round(rows * float(cfg.validation_fraction))))
    validation_rows = min(max(1, requested), rows // 2)
    if validation_rows < 1:
        return x_train, np.empty((0, x_train.shape[1]), dtype=x_train.dtype), labels, None
    fit_labels = labels[:-validation_rows] if labels is not None else None
    valid_labels = labels[-validation_rows:] if labels is not None else None
    return x_train[:-validation_rows], x_train[-validation_rows:], fit_labels, valid_labels


def _architecture_labels(frame: pd.DataFrame, cfg: LatentAutoencoderConfig) -> np.ndarray | None:
    label_col = str(cfg.architecture_selection_label_col)
    if label_col not in frame.columns:
        return None
    labels = frame[label_col].astype("string").fillna("").to_numpy(dtype=object)
    if len(labels) < 2 or len(set(labels)) < 2:
        return None
    return labels


def _validation_latent_label_purity(
    model,
    fit_x: np.ndarray,
    valid_x: np.ndarray,
    *,
    fit_labels: np.ndarray | None,
    valid_labels: np.ndarray | None,
    cfg: LatentAutoencoderConfig,
    device: str,
) -> float:
    if fit_labels is None or valid_labels is None or len(set(fit_labels)) < 2:
        return float("nan")
    fit_latent = _autoencoder_latent(model, fit_x, device=device, batch_size=cfg.batch_size)
    valid_latent = _autoencoder_latent(model, valid_x, device=device, batch_size=cfg.batch_size)
    if len(fit_latent) == 0 or len(valid_latent) == 0:
        return float("nan")
    neighbors = max(1, min(int(cfg.latent_neighbor_count), len(fit_latent)))
    nn_index = NearestNeighbors(n_neighbors=neighbors, metric=cfg.nn_metric)
    nn_index.fit(fit_latent)
    _, indices = nn_index.kneighbors(valid_latent, return_distance=True)
    neighbor_labels = fit_labels[indices]
    purity = np.mean(neighbor_labels == valid_labels.reshape(-1, 1), axis=1)
    return float(np.nanmean(purity))


def _latent_dimension_penalty(bottleneck_dim: int, in_dim: int, cfg: LatentAutoencoderConfig) -> float:
    return float(cfg.latent_dim_penalty) * (float(bottleneck_dim) / max(float(in_dim), 1.0))


def _architecture_selection_score(
    *,
    validation_error_mean: float,
    latent_label_purity: float,
    latent_dim_penalty: float,
    cfg: LatentAutoencoderConfig,
) -> tuple[str, float]:
    metric = str(cfg.architecture_selection_metric).strip().lower()
    if metric == "latent_label_purity" and np.isfinite(latent_label_purity):
        return "latent_label_purity_minus_dim_penalty", float(latent_label_purity - latent_dim_penalty)
    if np.isfinite(validation_error_mean):
        return "negative_validation_reconstruction_error_minus_dim_penalty", float(-validation_error_mean - latent_dim_penalty)
    return "none", float("-inf")


def _best_architecture_index(diagnostics: list[dict[str, float | int | str]]) -> int:
    return max(
        range(len(diagnostics)),
        key=lambda idx: (
            float(diagnostics[idx]["selection_score"]),
            -int(diagnostics[idx]["bottleneck_dim"]),
            -int(diagnostics[idx]["hidden_dim"]),
        ),
    )


def _candidate_architectures(in_dim: int, cfg: LatentAutoencoderConfig) -> list[tuple[int, int]]:
    in_dim = int(in_dim)
    if in_dim <= 0:
        return []
    hidden_upper = max(1, min(int(cfg.max_hidden_dim), max(in_dim * 2, int(cfg.min_hidden_dim))))
    hidden_values = {
        int(np.ceil(in_dim * float(multiplier)))
        for multiplier in cfg.hidden_multipliers
        if np.isfinite(multiplier) and multiplier > 0
    }
    hidden_values.update({in_dim, in_dim * 2})
    hidden_values = {
        max(1, min(hidden_upper, max(int(cfg.min_hidden_dim), int(value))))
        for value in hidden_values
    }

    bottleneck_upper = max(1, min(int(cfg.max_bottleneck_dim), max(1, in_dim - 1), hidden_upper))
    bottleneck_values = {
        int(np.ceil(in_dim * float(ratio)))
        for ratio in cfg.bottleneck_ratios
        if np.isfinite(ratio) and ratio > 0
    }
    bottleneck_values.update({1, in_dim // 4, in_dim // 2})
    bottleneck_values = {
        max(1, min(bottleneck_upper, max(int(cfg.min_bottleneck_dim), int(value))))
        for value in bottleneck_values
    }

    candidates = sorted(
        {
            (hidden_dim, bottleneck_dim)
            for hidden_dim in hidden_values
            for bottleneck_dim in bottleneck_values
            if bottleneck_dim <= hidden_dim
        },
        key=lambda item: (item[0] * item[1], item[1], item[0]),
    )
    max_candidates = int(max(1, cfg.max_architecture_candidates))
    if len(candidates) <= max_candidates:
        return candidates
    pick_indexes = np.linspace(0, len(candidates) - 1, num=max_candidates, dtype=int)
    return [candidates[int(index)] for index in pick_indexes]


def _standardize_fit(frame: pd.DataFrame, features: list[str]):
    raw = frame[features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype="float64", copy=True)
    lower = np.nanpercentile(raw, 0.1, axis=0)
    upper = np.nanpercentile(raw, 99.9, axis=0)
    clipped = np.clip(raw, lower, upper)
    center = np.nanmedian(clipped, axis=0)
    q1 = np.nanpercentile(clipped, 25.0, axis=0)
    q3 = np.nanpercentile(clipped, 75.0, axis=0)
    scale = q3 - q1
    center = np.nan_to_num(center, nan=0.0, posinf=0.0, neginf=0.0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-9), scale, 1.0)
    lower = np.nan_to_num(lower, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
    upper = np.nan_to_num(upper, nan=np.inf, posinf=np.inf, neginf=-np.inf)
    filled = np.where(np.isfinite(clipped), clipped, center)
    x = (filled - center) / scale
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.clip(x, -50.0, 50.0).astype("float32")
    return x, center.astype("float32"), scale.astype("float32"), lower.astype("float32"), upper.astype("float32")


def _standardize_apply(
    frame: pd.DataFrame,
    features: list[str],
    center: np.ndarray,
    scale: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    raw = frame[features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype="float64", copy=True)
    clipped = np.clip(raw, lower, upper)
    filled = np.where(np.isfinite(clipped), clipped, center)
    x = (filled - center) / scale
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(x, -50.0, 50.0).astype("float32")


def _autoencoder_recon_error(model, x: np.ndarray, *, device: str, batch_size: int) -> np.ndarray:
    import torch

    out = np.empty((len(x),), dtype="float64")
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            end = min(start + batch_size, len(x))
            batch = torch.tensor(x[start:end], dtype=torch.float32, device=device)
            recon = model(batch).detach().cpu().numpy().astype("float32")
            diff = recon - x[start:end]
            out[start:end] = np.mean(diff * diff, axis=1)
    return out


def _autoencoder_latent(model, x: np.ndarray, *, device: str, batch_size: int) -> np.ndarray:
    import torch

    out = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            end = min(start + batch_size, len(x))
            batch = torch.tensor(x[start:end], dtype=torch.float32, device=device)
            latent = model.encoder(batch).detach().cpu().numpy().astype("float32")
            out.append(latent)
    return np.vstack(out) if out else np.empty((0, 0), dtype="float32")
