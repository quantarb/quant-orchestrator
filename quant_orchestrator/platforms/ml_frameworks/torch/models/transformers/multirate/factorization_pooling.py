"""Learned factorization pooling for collaborative-filtering matrices."""

from __future__ import annotations

import torch
from torch import nn


class FactorizationPooling(nn.Module):
    """Factor ``N x M`` rows into one learned ``1 x M`` prototype.

    Each row is encoded into a low-rank latent factor, the latent factors are
    pooled across rows, and the pooled factor is decoded back to the original
    column space. This is deliberately separate from statistical document
    prototype pooling.

    Parameters
    ----------
    input_dim:
        Number of columns in the collaborative-filtering matrix.
    rank:
        Dimension of the learned latent factorization.
    """

    def __init__(self, input_dim: int, rank: int) -> None:
        super().__init__()
        if input_dim < 1 or rank < 1:
            raise ValueError("input_dim and rank must be positive")
        self.input_dim = int(input_dim)
        self.rank = int(rank)
        self.encoder = nn.Linear(self.input_dim, self.rank, bias=False)
        self.decoder = nn.Linear(self.rank, self.input_dim, bias=False)

    def forward(
        self,
        matrix: torch.Tensor,
        row_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return one factorized prototype per batch as ``[B, 1, M]``.

        ``matrix`` may be ``[N, M]`` or ``[B, N, M]``. ``row_mask`` uses
        ``True`` for rows to ignore and may be ``[N]`` or ``[B, N]``.
        """
        squeeze_batch = matrix.ndim == 2
        if squeeze_batch:
            matrix = matrix.unsqueeze(0)
        if matrix.ndim != 3 or matrix.shape[-1] != self.input_dim:
            raise ValueError(f"matrix must have shape [B, N, {self.input_dim}]")
        if matrix.shape[1] == 0:
            raise ValueError("matrix must contain at least one row")
        if row_mask is None:
            valid = torch.ones(matrix.shape[:2], dtype=torch.bool, device=matrix.device)
        else:
            if row_mask.ndim == 1:
                row_mask = row_mask.unsqueeze(0)
            if row_mask.shape != matrix.shape[:2]:
                raise ValueError("row_mask must have shape [N] or [B, N]")
            valid = ~row_mask.bool()
        clean = torch.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
        latent_rows = self.encoder(clean)
        weights = valid.unsqueeze(-1).to(latent_rows.dtype)
        pooled_latent = (latent_rows * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        prototype = self.decoder(pooled_latent).unsqueeze(1)
        return prototype.squeeze(0) if squeeze_batch else prototype


__all__ = ["FactorizationPooling"]
