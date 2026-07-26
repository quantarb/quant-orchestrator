"""Required feature-coverage building blocks shared by Torch models.

The model receives already aligned tensors, but the alignment contract is
explicit: absent families are represented by values plus a presence mask.  A
zero value with presence ``1`` is therefore different from an imputed zero
with presence ``0``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import torch
from torch import nn


class CoverageAwareInput(nn.Module):
    """Family adapters, missingness signals, modality adapters, and gates.

    ``values`` has shape ``[batch, sequence, features]`` and is partitioned
    into endpoint-level family slices.  All columns returned by one endpoint
    share one adapter and coverage mask; callers do not need to split wide
    endpoints such as options data into individual column families.  NaNs are
    replaced with zero after their missingness is recorded.  Presence defaults
    to finite observation of the complete family, so coverage behavior cannot
    silently be skipped.
    """

    def __init__(
        self,
        family_dims: Mapping[str, int],
        d_model: int,
        *,
        modalities: Sequence[str] = ("default",),
    ) -> None:
        super().__init__()
        if not family_dims or any(int(dim) <= 0 for dim in family_dims.values()):
            raise ValueError("family_dims must contain positive dimensions")
        self.family_names = tuple(family_dims)
        self.family_dims = {name: int(dim) for name, dim in family_dims.items()}
        self.slices: dict[str, slice] = {}
        start = 0
        for name, dim in self.family_dims.items():
            self.slices[name] = slice(start, start + dim)
            start += dim
        self.feature_dim = start
        self.family_adapters = nn.ModuleDict({
            name: nn.Sequential(nn.Linear(dim, d_model), nn.LayerNorm(d_model), nn.GELU())
            for name, dim in self.family_dims.items()
        })
        self.missingness_embeddings = nn.Parameter(torch.randn(len(self.family_names), d_model) * 0.02)
        self.coverage_gate_bias = nn.Parameter(torch.full((len(self.family_names),), -2.0))
        self.modality_names = tuple(modalities)
        if not self.modality_names:
            raise ValueError("modalities must not be empty")
        self.modality_adapters = nn.ModuleDict({
            name: nn.Sequential(nn.Linear(d_model, d_model), nn.LayerNorm(d_model))
            for name in self.modality_names
        })

    def forward(
        self,
        values: torch.Tensor,
        *,
        family_presence: torch.Tensor | None = None,
        modality_ids: torch.Tensor | None = None,
        return_family_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if values.ndim != 3 or values.shape[-1] != self.feature_dim:
            raise ValueError(f"values must have shape [batch, sequence, {self.feature_dim}]")
        observed_features = torch.isfinite(values)
        clean = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        inferred_presence = torch.stack([
            observed_features[..., self.slices[name]].all(dim=-1)
            for name in self.family_names
        ], dim=-1)
        if family_presence is None:
            presence = inferred_presence
        else:
            if family_presence.shape != values.shape[:2] + (len(self.family_names),):
                raise ValueError("family_presence must have shape [batch, sequence, families]")
            presence = family_presence.to(device=values.device, dtype=values.dtype).clamp(0, 1)
        states = []
        for index, name in enumerate(self.family_names):
            family = clean[..., self.slices[name]]
            state = self.family_adapters[name](family)
            missing = (~observed_features[..., self.slices[name]]).to(values.dtype).mean(dim=-1, keepdim=True)
            state = state + missing * self.missingness_embeddings[index]
            gate = torch.sigmoid(self.coverage_gate_bias[index]) * presence[..., index:index + 1]
            states.append(state * gate)
        combined = torch.stack(states, dim=0).sum(dim=0)
        if modality_ids is None:
            modality_ids = torch.zeros(values.shape[:2], dtype=torch.long, device=values.device)
        if modality_ids.shape != values.shape[:2]:
            raise ValueError("modality_ids must have shape [batch, sequence]")
        modality_state = torch.zeros_like(combined)
        for index, name in enumerate(self.modality_names):
            modality_state = modality_state + self.modality_adapters[name](combined) * modality_ids.eq(index).unsqueeze(-1)
        combined = combined + modality_state
        if return_family_states:
            return combined, torch.stack(states, dim=2), presence
        return combined


__all__ = ["CoverageAwareInput"]
