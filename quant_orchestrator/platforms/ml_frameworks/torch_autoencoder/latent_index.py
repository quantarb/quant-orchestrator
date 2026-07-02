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
    batch_size: int = 4096
    lr: float = 1e-3
    weight_decay: float = 1e-4
    denoise_std: float = 0.02
    familiarity_quantile: float = 99.9
    nn_metric: str = "euclidean"
    require_cuda: bool = False
    device: str | None = None


class _FamilyAutoEncoderModule:
    @staticmethod
    def create(in_dim: int, hidden_dim: int, bottleneck_dim: int):
        import torch.nn as nn

        class FamilyAutoEncoder(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(in_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, bottleneck_dim),
                )
                self.decoder = nn.Sequential(
                    nn.Linear(bottleneck_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
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

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        *,
        features: Iterable[str],
        config: LatentAutoencoderConfig | None = None,
    ) -> "LatentAutoencoderIndex | None":
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        cfg = config or LatentAutoencoderConfig()
        feature_list = list(features)
        runtime = configure_torch_runtime(require_cuda=cfg.require_cuda, device=cfg.device)
        device = runtime.torch_device
        x_train, center, scale, lower, upper = _standardize_fit(frame, feature_list)
        if len(x_train) < 2:
            return None

        in_dim = x_train.shape[1]
        hidden_dim = max(8, min(128, in_dim * 2))
        bottleneck_dim = max(2, min(32, hidden_dim // 4, max(2, in_dim // 2)))
        model = _FamilyAutoEncoderModule.create(in_dim, hidden_dim, bottleneck_dim).to(device)
        loader = DataLoader(
            TensorDataset(torch.tensor(x_train, dtype=torch.float32)),
            batch_size=cfg.batch_size,
            shuffle=True,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        loss_fn = nn.MSELoss()
        model.train()
        for _epoch in range(cfg.epochs):
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
        }


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
