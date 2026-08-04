"""Learned, task-conditioned feature engineering for Transformer inputs."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn


AutoFeatureMode = Literal["temporal", "cross_sectional"]
AGGREGATION_FUNCTIONS = ("mean", "min", "max", "rmse", "q25", "q50", "q75")


class LearnedAggregationGate(nn.Module):
    """Choose a differentiable mixture of family aggregation candidates."""

    def __init__(self, d_model: int, aggregation_functions: tuple[str, ...] = AGGREGATION_FUNCTIONS) -> None:
        super().__init__()
        if not aggregation_functions:
            raise ValueError("aggregation_functions must not be empty")
        self.aggregation_functions = tuple(aggregation_functions)
        self.network = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, len(self.aggregation_functions)),
        )
        self.family_logits: nn.Parameter | None = None

    def configure_families(self, family_count: int) -> None:
        if family_count <= 0:
            raise ValueError("family_count must be positive")
        self.family_logits = nn.Parameter(torch.zeros(family_count, len(self.aggregation_functions)))

    def forward(
        self,
        candidates: torch.Tensor,
        family_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if candidates.ndim != 4:
            raise ValueError("candidates must have shape [batch, family, aggregation, d_model]")
        if candidates.shape[2] != len(self.aggregation_functions):
            raise ValueError("candidate aggregation dimension does not match the gate")
        context = candidates.mean(dim=2)
        weights = self.network(context)
        if self.family_logits is not None:
            if family_indices is None:
                if candidates.shape[1] != self.family_logits.shape[0]:
                    raise ValueError("candidate family dimension does not match configured families")
                family_logits = self.family_logits
            else:
                family_logits = self.family_logits.index_select(0, family_indices)
                if candidates.shape[1] != family_logits.shape[0]:
                    raise ValueError("family index count does not match candidate family dimension")
            weights = weights + family_logits.unsqueeze(0)
        weights = torch.softmax(weights, dim=-1)
        gated = (candidates * weights.unsqueeze(-1)).sum(dim=2)
        return gated, weights


class AutoFeatureEngineer(nn.Module):
    """Learn feature relationships with one date-aware attention policy.

    Each endpoint feature family is represented as a token.  The attention
    mask, rather than a hand-written indicator library or a task-specific
    branch, determines which relationships may be learned:

    * tokens on the same date attend bidirectionally;
    * tokens from earlier dates are visible;
    * tokens from future dates are blocked.

    Consequently, the same block can learn within-family, cross-family,
    cross-sectional, and temporal transformations without enumerating ratios,
    ranks, moving averages, or other fixed operators.
    """

    def __init__(
        self,
        d_model: int,
        *,
        num_heads: int = 4,
        max_position: int = 512,
        aggregation_functions: tuple[str, ...] = AGGREGATION_FUNCTIONS,
    ) -> None:
        super().__init__()
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        self.position = nn.Parameter(torch.randn(max_position, d_model) * 0.02)
        # Each family is an independent temporal document. Reshaping to
        # [batch * family, date, d_model] gives every (symbol, family)
        # document its own attention sequence and prevents cross-family
        # interactions inside the encoder.
        self.temporal_attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        # Cross-rate fusion happens only after each rate/family document has
        # been independently encoded and pooled.
        self.cross_rate_attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.feature_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.output_norm = nn.LayerNorm(d_model)
        self.aggregation_gate = LearnedAggregationGate(d_model, aggregation_functions)

    def gate_aggregations(
        self,
        candidates: tuple[torch.Tensor, ...],
        family_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gate prototype candidates independently for every feature family."""
        if len(candidates) != len(self.aggregation_gate.aggregation_functions):
            raise ValueError("candidate count does not match configured aggregation functions")
        if not candidates or candidates[0].ndim not in {2, 3}:
            raise ValueError("candidates must contain [batch, d_model] or [batch, family, d_model] tensors")
        family_axis = candidates[0].ndim == 3
        stacked = torch.stack(candidates, dim=2 if family_axis else 1)
        if not family_axis:
            stacked = stacked.unsqueeze(1)
        gated, weights = self.aggregation_gate(stacked, family_indices=family_indices)
        return (gated[:, 0], weights[:, 0]) if not family_axis else (gated, weights)

    def cross_rate_features(self, rate_states: tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Learn bidirectional interactions among the available rate states."""
        if not rate_states:
            raise ValueError("at least one rate state is required")
        stacked = torch.stack(rate_states, dim=1)
        attended, _ = self.cross_rate_attention(stacked, stacked, stacked)
        return attended.mean(dim=1)

    def forward(
        self,
        values: torch.Tensor,
        *,
        mode: AutoFeatureMode,
        dates: torch.Tensor | None = None,
        family_states: torch.Tensor | None = None,
        family_presence: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        return_subtoken_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if values.ndim != 3:
            raise ValueError("values must have shape [batch, sequence, d_model]")
        batch, length, d_model = values.shape
        if length > self.position.shape[0]:
            raise ValueError("sequence exceeds auto-feature positional capacity")
        if mode not in {"temporal", "cross_sectional"}:
            raise ValueError(f"unsupported auto-feature mode: {mode!r}")

        if family_states is None:
            family_states = values.unsqueeze(2)
        if family_states.ndim != 4 or family_states.shape[:2] != (batch, length):
            raise ValueError("family_states must have shape [batch, sequence, family, d_model]")
        family_count = family_states.shape[2]
        if family_states.shape[3] != d_model:
            raise ValueError("family_states d_model must match values")

        if family_presence is None:
            family_presence = torch.ones(
                (batch, length, family_count), dtype=torch.bool, device=values.device
            )
        if family_presence.shape != (batch, length, family_count):
            raise ValueError("family_presence must have shape [batch, sequence, family]")
        family_presence = family_presence.bool()

        if padding_mask is not None:
            if padding_mask.shape != (batch, length):
                raise ValueError("padding_mask must have shape [batch, sequence]")
            padding_mask = padding_mask.bool()

        if dates is None:
            dates = torch.arange(length, device=values.device)
        if dates.ndim != 1 or dates.shape[0] != length:
            raise ValueError("dates must have shape [sequence]")

        tokens = family_states + self.position[:length].view(1, length, 1, -1)
        token_padding = family_presence.eq(0)
        if padding_mask is not None:
            token_padding = token_padding | padding_mask.unsqueeze(-1)
        # One independent temporal document per (batch item, family).
        document_tokens = tokens.permute(0, 2, 1, 3).reshape(batch * family_count, length, d_model)
        document_padding = token_padding.permute(0, 2, 1).reshape(batch * family_count, length).clone()
        # Avoid all-masked rows; the placeholder is excluded from pooling by
        # the original presence/padding mask after attention. Unmasking the
        # first position for every document also gives left-padded queries a
        # legal causal key.
        document_padding[:, 0] = False
        temporal_mask = dates[None, :] > dates[:, None]
        attended, _ = self.temporal_attention(
            document_tokens,
            document_tokens,
            document_tokens,
            attn_mask=temporal_mask,
            key_padding_mask=document_padding,
        )
        attended = torch.nan_to_num(attended, nan=0.0, posinf=0.0, neginf=0.0)
        attended = attended.reshape(batch, family_count, length, d_model).permute(0, 2, 1, 3)
        attended = self.output_norm(attended + tokens)
        attended = self.output_norm(attended + self.feature_mlp(attended))
        weights = family_presence.to(attended.dtype).unsqueeze(-1)
        engineered = (attended * weights).sum(dim=2) / weights.sum(dim=2).clamp_min(1.0)
        output = self.output_norm(values + engineered)
        output = torch.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)
        if return_subtoken_states:
            return output, attended
        return output


__all__ = ["AGGREGATION_FUNCTIONS", "AutoFeatureEngineer", "AutoFeatureMode", "LearnedAggregationGate"]
