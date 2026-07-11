from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Sequence

import numpy as np
import pandas as pd

from quant_orchestrator.platforms.ml_frameworks.torch.runtime import configure_torch_runtime


@dataclass(frozen=True)
class ReturnHorizonSpec:
    name: str
    periods: int | None
    kind: str = "trailing"


DEFAULT_RETURN_HORIZONS: tuple[ReturnHorizonSpec, ...] = (
    ReturnHorizonSpec("1d", 1),
    ReturnHorizonSpec("5d", 5),
    ReturnHorizonSpec("30d", 30),
    ReturnHorizonSpec("6m", 126),
    ReturnHorizonSpec("ytd", None, "ytd"),
    ReturnHorizonSpec("1y", 252),
    ReturnHorizonSpec("5y", 1260),
)


@dataclass(frozen=True)
class MultiHorizonReturnTensor:
    dates: pd.DatetimeIndex
    symbols: tuple[str, ...]
    horizons: tuple[str, ...]
    values: np.ndarray

    def panel(self, horizon: str) -> pd.DataFrame:
        idx = self.horizons.index(horizon)
        return pd.DataFrame(self.values[:, :, idx], index=self.dates, columns=self.symbols)

    def matrix(self, date: pd.Timestamp | str) -> pd.DataFrame:
        loc = self.dates.get_loc(pd.Timestamp(date).normalize())
        return pd.DataFrame(self.values[loc], index=self.symbols, columns=self.horizons)


@dataclass(frozen=True)
class MultiHorizonStandardizer:
    center: np.ndarray
    scale: np.ndarray
    lower: np.ndarray
    upper: np.ndarray


@dataclass(frozen=True)
class MatrixAutoencoderConfig:
    epochs: int = 80
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-4
    denoise_std: float = 0.01
    latent_dim_ratio: float = 0.20
    min_latent_dim: int = 2
    max_latent_dim: int = 32
    hidden_multiplier: float = 1.5
    max_hidden_dim: int = 512
    require_cuda: bool = False
    device: str | None = None
    random_seed: int | None = 7


@dataclass(frozen=True)
class MatrixAutoencoderFit:
    model: object
    standardizer: MultiHorizonStandardizer
    dates: pd.DatetimeIndex
    symbols: tuple[str, ...]
    horizons: tuple[str, ...]
    device: str
    hidden_dim: int
    latent_dim: int
    fit_seconds: float
    losses: tuple[float, ...]


def build_multi_horizon_return_tensor(
    wide_close: pd.DataFrame,
    *,
    horizons: Sequence[ReturnHorizonSpec] = DEFAULT_RETURN_HORIZONS,
    min_finite_horizons: int = 1,
) -> MultiHorizonReturnTensor:
    """Build a date x symbol x horizon tensor of trailing return features."""

    close = _normalize_wide_close(wide_close)
    panels: list[pd.DataFrame] = []
    names: list[str] = []
    for spec in horizons:
        panel = _horizon_returns(close, spec)
        panels.append(panel)
        names.append(str(spec.name))

    values = np.stack([panel.to_numpy(dtype="float64", copy=True) for panel in panels], axis=2)
    finite_counts = np.isfinite(values).sum(axis=(1, 2))
    keep = finite_counts >= int(min_finite_horizons)
    return MultiHorizonReturnTensor(
        dates=pd.DatetimeIndex(close.index[keep]),
        symbols=tuple(str(col) for col in close.columns),
        horizons=tuple(names),
        values=values[keep],
    )


def fit_multi_horizon_standardizer(
    values: np.ndarray,
    *,
    lower_quantile: float = 0.1,
    upper_quantile: float = 99.9,
) -> tuple[np.ndarray, MultiHorizonStandardizer]:
    raw = np.asarray(values, dtype="float64")
    lower = np.nanpercentile(raw, lower_quantile, axis=0)
    upper = np.nanpercentile(raw, upper_quantile, axis=0)
    clipped = np.clip(raw, lower, upper)
    center = np.nanmedian(clipped, axis=0)
    q1 = np.nanpercentile(clipped, 25.0, axis=0)
    q3 = np.nanpercentile(clipped, 75.0, axis=0)
    scale = q3 - q1
    center = np.nan_to_num(center, nan=0.0, posinf=0.0, neginf=0.0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-9), scale, 1.0)
    lower = np.nan_to_num(lower, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
    upper = np.nan_to_num(upper, nan=np.inf, posinf=np.inf, neginf=-np.inf)
    standardizer = MultiHorizonStandardizer(center=center, scale=scale, lower=lower, upper=upper)
    return apply_multi_horizon_standardizer(raw, standardizer), standardizer


def apply_multi_horizon_standardizer(
    values: np.ndarray,
    standardizer: MultiHorizonStandardizer,
) -> np.ndarray:
    raw = np.asarray(values, dtype="float64")
    clipped = np.clip(raw, standardizer.lower, standardizer.upper)
    filled = np.where(np.isfinite(clipped), clipped, standardizer.center)
    x = (filled - standardizer.center) / standardizer.scale
    return np.clip(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), -50.0, 50.0).astype("float32")


def train_matrix_autoencoder(
    tensor: MultiHorizonReturnTensor,
    *,
    train_end: pd.Timestamp | str,
    config: MatrixAutoencoderConfig | None = None,
) -> MatrixAutoencoderFit:
    """Train an autoencoder that reconstructs each date's symbol x horizon return matrix."""

    import torch
    import torch.nn as nn

    cfg = config or MatrixAutoencoderConfig()
    if cfg.random_seed is not None:
        np.random.seed(int(cfg.random_seed))
        torch.manual_seed(int(cfg.random_seed))

    train_mask = tensor.dates <= pd.Timestamp(train_end).normalize()
    x_train, standardizer = fit_multi_horizon_standardizer(tensor.values[train_mask])
    if len(x_train) < 2:
        raise ValueError("At least two train dates are required to train the matrix autoencoder")

    runtime = configure_torch_runtime(require_cuda=cfg.require_cuda, device=cfg.device)
    device = runtime.torch_device
    symbol_count, horizon_count = x_train.shape[1], x_train.shape[2]
    in_dim = int(symbol_count * horizon_count)
    hidden_dim = int(max(4, min(cfg.max_hidden_dim, round(in_dim * cfg.hidden_multiplier))))
    latent_dim = int(
        max(
            cfg.min_latent_dim,
            min(cfg.max_latent_dim, round(in_dim * cfg.latent_dim_ratio), max(1, in_dim - 1)),
        )
    )
    model = CrossSectionalReturnMatrixAutoEncoder(
        symbol_count=symbol_count,
        horizon_count=horizon_count,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
    ).to(device)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(x_train, dtype=torch.float32)),
        batch_size=cfg.batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, cfg.epochs))
    loss_fn = nn.MSELoss()
    losses: list[float] = []
    started = perf_counter()
    model.train()
    for _epoch in range(cfg.epochs):
        epoch_losses: list[float] = []
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
                epoch_losses.append(float(loss.detach().cpu()))
        scheduler.step()
        losses.append(float(np.mean(epoch_losses)) if epoch_losses else np.nan)

    return MatrixAutoencoderFit(
        model=model,
        standardizer=standardizer,
        dates=tensor.dates,
        symbols=tensor.symbols,
        horizons=tensor.horizons,
        device=device,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        fit_seconds=perf_counter() - started,
        losses=tuple(losses),
    )


def reconstruct_multi_horizon_returns(
    fit: MatrixAutoencoderFit,
    values: np.ndarray,
    *,
    batch_size: int = 8192,
) -> np.ndarray:
    import torch

    x = apply_multi_horizon_standardizer(values, fit.standardizer)
    frames: list[np.ndarray] = []
    fit.model.eval()
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            end = min(start + batch_size, len(x))
            batch = torch.tensor(x[start:end], dtype=torch.float32, device=fit.device)
            frames.append(fit.model(batch).detach().cpu().numpy().astype("float32"))
    recon_x = np.vstack(frames) if frames else np.empty_like(x)
    return (recon_x.astype("float64") * fit.standardizer.scale) + fit.standardizer.center


def multi_horizon_residual_panels(
    tensor: MultiHorizonReturnTensor,
    reconstructed: np.ndarray,
) -> dict[str, pd.DataFrame]:
    _validate_reconstruction_shape(tensor, reconstructed)
    residual = tensor.values - reconstructed
    return {
        horizon: pd.DataFrame(residual[:, :, idx], index=tensor.dates, columns=tensor.symbols)
        for idx, horizon in enumerate(tensor.horizons)
    }


def signed_squared_residual_score(
    residual: np.ndarray,
    *,
    horizon_weights: Sequence[float] | None = None,
) -> np.ndarray:
    """Collapse symbol x horizon residuals into one signed score per date/symbol."""

    weights = _normalized_horizon_weights(residual.shape[2], horizon_weights)
    signed_sq = np.sign(residual) * np.square(residual)
    return np.tensordot(signed_sq, weights, axes=([2], [0]))


def signed_residual_score(
    residual: np.ndarray,
    *,
    horizon_weights: Sequence[float] | None = None,
) -> np.ndarray:
    """Collapse symbol x horizon residuals into a signed linear score per date/symbol."""

    weights = _normalized_horizon_weights(residual.shape[2], horizon_weights)
    return np.tensordot(residual, weights, axes=([2], [0]))


def _normalized_horizon_weights(
    horizon_count: int,
    horizon_weights: Sequence[float] | None,
) -> np.ndarray:
    weights = np.ones(int(horizon_count), dtype="float64")
    if horizon_weights is not None:
        weights = np.asarray(list(horizon_weights), dtype="float64")
        if weights.shape != (horizon_count,):
            raise ValueError("horizon_weights must have one value per horizon")
    return weights / max(float(np.abs(weights).sum()), 1e-12)


def build_symbol_reconstruction_feature_frame(
    tensor: MultiHorizonReturnTensor,
    reconstructed: np.ndarray,
    *,
    prefix: str = "csra",
    horizon_weights: Sequence[float] | None = None,
) -> pd.DataFrame:
    """Return one classifier feature row per date/symbol from a matrix AE reconstruction.

    Each row contains the observed horizon returns for that symbol, the reconstructed
    horizon returns implied by the full cross-section for the same date, and
    residual diagnostics. Join the result to optimal labels on ``date`` and
    ``symbol`` before classifier training.
    """

    _validate_reconstruction_shape(tensor, reconstructed)
    residual = tensor.values - reconstructed
    signed_residual = signed_residual_score(residual, horizon_weights=horizon_weights)
    score = signed_squared_residual_score(residual, horizon_weights=horizon_weights)
    abs_residual = np.abs(residual)
    residual_norm = np.sqrt(np.nanmean(np.square(residual), axis=2))

    dates = np.repeat(tensor.dates.to_numpy(), len(tensor.symbols))
    symbols = np.tile(np.asarray(tensor.symbols, dtype=object), len(tensor.dates))
    out = pd.DataFrame({"date": pd.to_datetime(dates), "symbol": symbols})

    observed_2d = tensor.values.reshape(len(tensor.dates) * len(tensor.symbols), len(tensor.horizons))
    recon_2d = reconstructed.reshape(observed_2d.shape)
    residual_2d = residual.reshape(observed_2d.shape)
    abs_residual_2d = abs_residual.reshape(observed_2d.shape)

    for idx, horizon in enumerate(tensor.horizons):
        suffix = _feature_suffix(horizon)
        out[f"{prefix}_return_{suffix}"] = observed_2d[:, idx].astype("float64")
        out[f"{prefix}_recon_{suffix}"] = recon_2d[:, idx].astype("float64")
        out[f"{prefix}_residual_{suffix}"] = residual_2d[:, idx].astype("float64")
        out[f"{prefix}_abs_residual_{suffix}"] = abs_residual_2d[:, idx].astype("float64")

    out[f"{prefix}_signed_residual"] = signed_residual.reshape(-1).astype("float64")
    out[f"{prefix}_signed_sq_score"] = score.reshape(-1).astype("float64")
    out[f"{prefix}_residual_norm"] = residual_norm.reshape(-1).astype("float64")
    return out


class CrossSectionalReturnMatrixAutoEncoder:
    def __new__(cls, symbol_count: int, horizon_count: int, hidden_dim: int, latent_dim: int):
        import torch.nn as nn

        class _MatrixAutoEncoder(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.symbol_count = int(symbol_count)
                self.horizon_count = int(horizon_count)
                in_dim = self.symbol_count * self.horizon_count
                self.encoder = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(in_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, latent_dim),
                )
                self.decoder = nn.Sequential(
                    nn.Linear(latent_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, in_dim),
                )

            def forward(self, x):
                latent = self.encoder(x)
                recon = self.decoder(latent)
                return recon.view(-1, self.symbol_count, self.horizon_count)

        return _MatrixAutoEncoder()


def _normalize_wide_close(wide_close: pd.DataFrame) -> pd.DataFrame:
    if wide_close is None or wide_close.empty:
        raise ValueError("wide_close must not be empty")
    close = wide_close.copy()
    close.columns = [str(col).strip().upper() for col in close.columns]
    close.index = pd.DatetimeIndex(pd.to_datetime(close.index, errors="coerce")).normalize()
    close = close.loc[close.index.notna()].sort_index()
    close = close.loc[~close.index.duplicated(keep="last")]
    close = close.apply(pd.to_numeric, errors="coerce")
    return close


def _validate_reconstruction_shape(tensor: MultiHorizonReturnTensor, reconstructed: np.ndarray) -> None:
    if np.asarray(reconstructed).shape != tensor.values.shape:
        raise ValueError(
            "reconstructed must have shape "
            f"{tensor.values.shape}, got {np.asarray(reconstructed).shape}"
        )


def _feature_suffix(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value)).strip("_")


def _horizon_returns(close: pd.DataFrame, spec: ReturnHorizonSpec) -> pd.DataFrame:
    if spec.kind == "trailing":
        if spec.periods is None or int(spec.periods) <= 0:
            raise ValueError(f"Trailing horizon {spec.name!r} requires positive periods")
        return close.pct_change(periods=int(spec.periods))
    if spec.kind == "ytd":
        return _ytd_returns(close)
    raise ValueError(f"Unknown return horizon kind: {spec.kind}")


def _ytd_returns(close: pd.DataFrame) -> pd.DataFrame:
    out = close.copy() * np.nan
    years = pd.Series(close.index.year, index=close.index)
    for year in sorted(years.unique()):
        year_dates = years.index[years.eq(year)]
        if len(year_dates) == 0:
            continue
        previous = close.loc[close.index < year_dates[0]].tail(1)
        if previous.empty:
            continue
        base = previous.iloc[0].replace(0.0, np.nan)
        out.loc[year_dates] = close.loc[year_dates].divide(base) - 1.0
    return out
