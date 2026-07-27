"""Multi-rate Transformer backbones with explicit temporal attention policies.

This module contains model mechanics only.  Dataset construction, target
engineering, training loops, walk-forward splits, and strategy selection stay
in caller-owned orchestrator workflows.

Three conventional Transformer layouts are supported:

* ``encoder_only``: one encoder per rate, followed by rate fusion;
* ``decoder_only``: one self-attention stack over concatenated rate tokens;
* ``encoder_decoder``: annual/quarterly/sparse encoders and a daily decoder.

The same date-causal mask builder is used by every layout.  Tokens on the same
date attend bidirectionally, earlier dates are visible, and future dates are
blocked.  A caller must still provide only as-of annual and quarterly
memories to prevent availability leakage.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, Mapping, Sequence

import torch
from torch import nn

from quant_orchestrator.platforms.ml_frameworks.torch.models.transformers.multirate.coverage import (
    CoverageAwareInput,
)
from quant_orchestrator.platforms.ml_frameworks.torch.models.transformers.multirate.auto_features import (
    AutoFeatureEngineer,
)


BackboneName = Literal["encoder_only", "decoder_only", "encoder_decoder"]
AttentionMode = Literal["temporal", "cross_sectional"]
DocumentPool = Literal["last", "mean"]
PredictionObjective = Literal["next_token", "masked_token"]
PredictionLevel = Literal["token", "subtoken"]
RateName = Literal["annual", "quarterly", "daily", "sparse"]


@dataclass(frozen=True)
class MultiRateTaskSpec:
    """One token-level or document-level prediction head."""

    task_name: str
    level: Literal["token", "document"]
    output_dim: int = 1
    source: Literal["daily", "annual", "quarterly", "sparse", "fused", "family"] = "daily"

    def __post_init__(self) -> None:
        if not self.task_name.strip():
            raise ValueError("task_name must not be empty")
        if self.output_dim <= 0:
            raise ValueError("output_dim must be positive")
        if self.level == "document" and self.source == "daily":
            # Daily is a valid document source after pooling; this branch is
            # intentionally accepted because it is the common issuer case.
            return


@dataclass(frozen=True)
class MultiRatePredictionTaskSpec:
    """A self-supervised next-token or masked-token prediction head."""

    task_name: str
    objective: PredictionObjective
    level: PredictionLevel
    output_dim: int = 1
    source: Literal["daily", "annual", "quarterly", "sparse"] = "daily"

    def __post_init__(self) -> None:
        if not self.task_name.strip():
            raise ValueError("task_name must not be empty")
        if self.output_dim <= 0:
            raise ValueError("output_dim must be positive")


@dataclass(frozen=True)
class MultiRateTransformerConfig:
    """Architecture and mask defaults for :class:`MultiRateTransformer`.

    The standard configuration has annual, quarterly, daily, and sparse-event
    streams. Sparse events retain their own timestamp and availability mask;
    they are never silently folded into the daily stream.
    """

    backbone: BackboneName = "encoder_decoder"
    d_model: int = 256
    num_heads: int = 8
    layers: int = 4
    dropout: float = 0.1
    document_pool: DocumentPool = "last"
    max_position: int = 512
    rates: tuple[str, ...] = ("annual", "quarterly", "daily", "sparse")

    def __post_init__(self) -> None:
        if self.d_model <= 0 or self.num_heads <= 0 or self.layers <= 0:
            raise ValueError("d_model, num_heads, and layers must be positive")
        if self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if self.document_pool not in {"last", "mean"}:
            raise ValueError("document_pool must be 'last' or 'mean'")
        if self.max_position <= 0:
            raise ValueError("max_position must be positive")
        if self.rates not in {
            ("annual", "quarterly", "daily"),
            ("annual", "quarterly", "daily", "sparse"),
        }:
            raise ValueError("rates must be annual, quarterly, daily[, sparse]")


def build_attention_mask(
    dates: torch.Tensor,
    *,
    mode: AttentionMode,
    causal: bool | None = None,
) -> torch.Tensor:
    """Build an additive ``[sequence, sequence]`` attention mask.

    ``dates`` is a one-dimensional date/session identifier per token.  For
    both supported task modes, queries may attend to same-date and earlier
    date tokens; future dates are blocked.  ``-inf`` means blocked, and ``0``
    means allowed.  ``mode`` remains part of the public API so callers can
    label task intent, but the date policy is deliberately unified.
    """
    if dates.ndim != 1:
        raise ValueError("dates must have shape [sequence]")
    length = dates.numel()
    device = dates.device
    if mode not in {"temporal", "cross_sectional"}:
        raise ValueError(f"unsupported attention mode: {mode!r}")
    if causal is False:
        return torch.zeros((length, length), device=device)
    allowed = dates[:, None] >= dates[None, :]
    return torch.zeros((length, length), device=device).masked_fill(~allowed, float("-inf"))


def _encoder(config: MultiRateTransformerConfig) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=config.d_model,
        nhead=config.num_heads,
        dim_feedforward=config.d_model * 4,
        dropout=config.dropout,
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=config.layers)


def _decoder(config: MultiRateTransformerConfig) -> nn.TransformerDecoder:
    layer = nn.TransformerDecoderLayer(
        d_model=config.d_model,
        nhead=config.num_heads,
        dim_feedforward=config.d_model * 4,
        dropout=config.dropout,
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerDecoder(layer, num_layers=config.layers)


class MultiRateTransformer(nn.Module):
    """Unified multi-rate model with token and document task heads."""

    def __init__(
        self,
        feature_dims: Mapping[str, int],
        *,
        config: MultiRateTransformerConfig | None = None,
        tasks: Sequence[MultiRateTaskSpec] = (),
        feature_families: Mapping[str, Mapping[str, int]] | None = None,
        modalities: Sequence[str] = ("default",),
        prediction_tasks: Sequence[MultiRatePredictionTaskSpec] = (),
        family_classification_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.config = config or MultiRateTransformerConfig()
        # Keep older three-rate callers loadable while making the standard
        # configuration four-rate. New training code must provide sparse.
        if set(feature_dims) == {"annual", "quarterly", "daily"} and self.config.rates == ("annual", "quarterly", "daily", "sparse"):
            self.config = replace(self.config, rates=("annual", "quarterly", "daily"))
        self.feature_dims = {rate: int(feature_dims[rate]) for rate in self.config.rates}
        family_config = feature_families or {
            rate: {"features": self.feature_dims[rate]} for rate in self.config.rates
        }
        if set(family_config) != set(self.config.rates):
            raise ValueError("feature_families must define every configured rate")
        if any(sum(int(dim) for dim in family_config[rate].values()) != self.feature_dims[rate]
               for rate in self.config.rates):
            raise ValueError("feature family dimensions must sum to feature_dims for each rate")
        self.coverage_inputs = nn.ModuleDict({
            rate: CoverageAwareInput(family_config[rate], self.config.d_model, modalities=modalities)
            for rate in self.config.rates
        })
        self.auto_feature_engineer = AutoFeatureEngineer(
            self.config.d_model,
            num_heads=self.config.num_heads,
            max_position=self.config.max_position,
        )
        self.rate_embeddings = nn.ParameterDict({
            rate: nn.Parameter(torch.randn(self.config.d_model) * 0.02)
            for rate in self.config.rates
        })
        self.encoders = nn.ModuleDict({rate: _encoder(self.config) for rate in self.config.rates})
        self.decoder_only_stack = _encoder(self.config)
        self.annual_encoder = _encoder(self.config)
        self.quarterly_encoder = _encoder(self.config)
        self.daily_decoder = _decoder(self.config)
        self.fusion = nn.Sequential(
            nn.Linear(self.config.d_model * len(self.config.rates), self.config.d_model),
            nn.LayerNorm(self.config.d_model),
            nn.GELU(),
        )
        self.family_names = tuple(dict.fromkeys(
            name for rate in self.config.rates for name in self.coverage_inputs[rate].family_names
        ))
        self.family_indices = {name: index for index, name in enumerate(self.family_names)}
        self.family_document_fusion = nn.Sequential(
            nn.Linear(self.config.d_model * len(self.config.rates), self.config.d_model),
            nn.LayerNorm(self.config.d_model),
            nn.GELU(),
        )
        self.task_specs = tuple(tasks)
        self.prediction_task_specs = tuple(prediction_tasks)
        self.family_classification_head = (
            nn.Linear(self.config.d_model, int(family_classification_dim))
            if family_classification_dim is not None else None
        )
        names = [task.task_name for task in self.task_specs]
        if len(names) != len(set(names)):
            raise ValueError("task names must be unique")
        self.task_heads = nn.ModuleDict({
            task.task_name: nn.Linear(self.config.d_model, task.output_dim)
            for task in self.task_specs
        })
        prediction_names = [task.task_name for task in self.prediction_task_specs]
        if len(prediction_names) != len(set(prediction_names)):
            raise ValueError("prediction task names must be unique")
        if set(names) & set(prediction_names):
            raise ValueError("prediction and supervised task names must be unique")
        self.prediction_heads = nn.ModuleDict({
            task.task_name: nn.Linear(self.config.d_model, task.output_dim)
            for task in self.prediction_task_specs
        })

    @staticmethod
    def _validate_stream(values: torch.Tensor, rate: str) -> None:
        if values.ndim != 3:
            raise ValueError(f"{rate} values must have shape [batch, sequence, features]")

    def _project(
        self,
        rate: str,
        values: torch.Tensor,
        family_presence: torch.Tensor | None = None,
        modality_ids: torch.Tensor | None = None,
        attention_mode: AttentionMode = "temporal",
        dates: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        return_subtoken_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        self._validate_stream(values, rate)
        projected = self.coverage_inputs[rate](
            values,
            family_presence=family_presence,
            modality_ids=modality_ids,
            return_family_states=True,
        )
        combined, family_states, presence = projected
        engineered = self.auto_feature_engineer(
            combined,
            mode=attention_mode,
            dates=dates,
            family_states=family_states,
            family_presence=presence,
            padding_mask=padding_mask,
            return_subtoken_states=return_subtoken_states,
        )
        if return_subtoken_states:
            engineered, subtokens = engineered
            return engineered + self.rate_embeddings[rate], subtokens
        return engineered + self.rate_embeddings[rate]

    @staticmethod
    def _padding_mask(mask: torch.Tensor | None, values: torch.Tensor) -> torch.Tensor | None:
        if mask is None:
            return None
        if mask.shape != values.shape[:2]:
            raise ValueError("padding masks must have shape [batch, sequence]")
        return mask.bool()

    def _pool(self, states: torch.Tensor, padding_mask: torch.Tensor | None) -> torch.Tensor:
        if states.shape[1] == 0:
            raise ValueError("cannot pool an empty sequence")
        states = torch.nan_to_num(states, nan=0.0, posinf=0.0, neginf=0.0)
        if self.config.document_pool == "last":
            if padding_mask is None:
                return states[:, -1]
            lengths = (~padding_mask).sum(dim=1).clamp_min(1) - 1
            return states[torch.arange(states.shape[0], device=states.device), lengths]
        valid = (~padding_mask).unsqueeze(-1) if padding_mask is not None else None
        if valid is None:
            return states.mean(dim=1)
        # Transformer rows that are fully masked can be NaN.  Multiplying
        # NaN by a zero mask does not remove it, so select valid rows first.
        clean = torch.where(valid, states, torch.zeros_like(states))
        return clean.sum(dim=1) / valid.sum(dim=1).clamp_min(1)

    @staticmethod
    def _pool_subtokens(
        states: torch.Tensor,
        padding_mask: torch.Tensor | None,
        family_presence: torch.Tensor | None,
    ) -> torch.Tensor:
        """Mean-pool valid ``[date, family]`` subtokens into one document."""
        if states.ndim != 4:
            raise ValueError("subtoken states must have shape [batch, sequence, family, d_model]")
        batch, length, family_count, _ = states.shape
        valid = torch.ones((batch, length, family_count), dtype=torch.bool, device=states.device)
        if padding_mask is not None:
            valid &= ~padding_mask.bool().unsqueeze(-1)
        if family_presence is not None:
            valid &= family_presence.bool()
        clean = torch.nan_to_num(states, nan=0.0, posinf=0.0, neginf=0.0)
        weights = valid.unsqueeze(-1).to(clean.dtype)
        return (clean * weights).sum(dim=(1, 2)) / weights.sum(dim=(1, 2)).clamp_min(1.0)

    @staticmethod
    def _pool_family_documents(
        states: torch.Tensor,
        padding_mask: torch.Tensor | None,
        family_presence: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pool each family document separately; return states and validity."""
        if states.ndim != 4:
            raise ValueError("subtoken states must have shape [batch, sequence, family, d_model]")
        batch, length, family_count, _ = states.shape
        valid = torch.ones((batch, length, family_count), dtype=torch.bool, device=states.device)
        if padding_mask is not None:
            valid &= ~padding_mask.bool().unsqueeze(-1)
        if family_presence is not None:
            valid &= family_presence.bool()
        clean = torch.nan_to_num(states, nan=0.0, posinf=0.0, neginf=0.0)
        weights = valid.unsqueeze(-1).to(clean.dtype)
        pooled = (clean * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return pooled, valid.any(dim=1)

    def _heads(
        self,
        *,
        daily_states: torch.Tensor,
        annual_states: torch.Tensor,
        quarterly_states: torch.Tensor,
        sparse_states: torch.Tensor | None,
        daily_padding_mask: torch.Tensor | None,
        annual_padding_mask: torch.Tensor | None,
        quarterly_padding_mask: torch.Tensor | None,
        sparse_padding_mask: torch.Tensor | None,
        annual_subtokens: torch.Tensor | None,
        quarterly_subtokens: torch.Tensor | None,
        daily_subtokens: torch.Tensor | None,
        sparse_subtokens: torch.Tensor | None,
        family_presence: Mapping[str, torch.Tensor | None],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        subtoken_states = {
            "annual": annual_subtokens,
            "quarterly": quarterly_subtokens,
            "daily": daily_subtokens,
            "sparse": sparse_subtokens,
        }
        padding_masks = {
            "annual": annual_padding_mask,
            "quarterly": quarterly_padding_mask,
            "daily": daily_padding_mask,
            "sparse": sparse_padding_mask,
        }
        rate_states = {
            rate: self._pool_subtokens(subtoken_states[rate], padding_masks[rate], family_presence[rate])
            if subtoken_states[rate] is not None
            else self._pool(states, padding_masks[rate])
            for rate, states in (
                ("annual", annual_states), ("quarterly", quarterly_states),
                ("daily", daily_states), ("sparse", sparse_states),
            ) if rate in self.config.rates
        }
        family_documents: dict[str, torch.Tensor] = {}
        for rate, subtokens in subtoken_states.items():
            if subtokens is None:
                continue
            family_documents[rate], _ = self._pool_family_documents(
                subtokens, padding_masks[rate], family_presence[rate],
            )
        total_families = len(self.family_names)
        family_parts = []
        for rate in self.config.rates:
            part = torch.zeros(
                (daily_states.shape[0], total_families, self.config.d_model),
                device=daily_states.device, dtype=daily_states.dtype,
            )
            if rate in family_documents:
                for local_index, name in enumerate(self.coverage_inputs[rate].family_names):
                    part[:, self.family_indices[name]] = family_documents[rate][:, local_index]
            family_parts.append(part)
        family_document_state = self.family_document_fusion(torch.cat(family_parts, dim=-1))
        ordered = tuple(rate_states[rate] for rate in self.config.rates)
        fused = self.fusion(torch.cat(ordered, dim=-1))
        fused = fused + self.auto_feature_engineer.cross_rate_features(ordered)
        token_outputs: dict[str, torch.Tensor] = {}
        document_outputs: dict[str, torch.Tensor] = {}
        sources = {
            rate: states for rate, states in (("annual", annual_states), ("quarterly", quarterly_states), ("daily", daily_states))
        }
        if sparse_states is not None:
            sources["sparse"] = sparse_states
        sources["fused"] = fused
        document_sources = {**rate_states, "fused": fused, "family": family_document_state}
        for task in self.task_specs:
            source = document_sources[task.source] if task.level == "document" else sources[task.source]
            source = torch.nan_to_num(source, nan=0.0, posinf=0.0, neginf=0.0)
            if task.level == "token":
                token_outputs[task.task_name] = self.task_heads[task.task_name](source)
            else:
                document_outputs[task.task_name] = self.task_heads[task.task_name](source)
        return token_outputs, document_outputs, fused, family_document_state

    def _prediction_heads(
        self,
        *,
        token_states: Mapping[str, torch.Tensor],
        subtoken_states: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        outputs: dict[str, torch.Tensor] = {}
        for task in self.prediction_task_specs:
            source = subtoken_states[task.source] if task.level == "subtoken" else token_states[task.source]
            source = torch.nan_to_num(source, nan=0.0, posinf=0.0, neginf=0.0)
            outputs[task.task_name] = self.prediction_heads[task.task_name](source)
        return outputs

    def forward(
        self,
        daily_values: torch.Tensor,
        annual_values: torch.Tensor,
        quarterly_values: torch.Tensor,
        sparse_values: torch.Tensor | None = None,
        *,
        attention_mode: AttentionMode = "temporal",
        daily_padding_mask: torch.Tensor | None = None,
        annual_padding_mask: torch.Tensor | None = None,
        quarterly_padding_mask: torch.Tensor | None = None,
        sparse_padding_mask: torch.Tensor | None = None,
        daily_dates: torch.Tensor | None = None,
        annual_dates: torch.Tensor | None = None,
        quarterly_dates: torch.Tensor | None = None,
        sparse_dates: torch.Tensor | None = None,
        daily_family_presence: torch.Tensor | None = None,
        annual_family_presence: torch.Tensor | None = None,
        quarterly_family_presence: torch.Tensor | None = None,
        daily_modality_ids: torch.Tensor | None = None,
        annual_modality_ids: torch.Tensor | None = None,
        quarterly_modality_ids: torch.Tensor | None = None,
        sparse_family_presence: torch.Tensor | None = None,
        sparse_modality_ids: torch.Tensor | None = None,
    ) -> dict[str, object]:
        """Encode a multi-rate window and return states plus task outputs.

        Date vectors are one-dimensional and shared across the batch.  For
        cross-sectional inference, pass same-day identifiers for peer tokens.
        Annual and quarterly values must already be restricted to observations
        available at the prediction timestamp.
        """
        if "sparse" in self.config.rates and sparse_values is None:
            raise ValueError("sparse_values is required for the four-rate transformer")
        streams = {
            "daily": daily_values, "annual": annual_values, "quarterly": quarterly_values,
            "sparse": sparse_values,
        }
        for rate, values in streams.items():
            if rate not in self.config.rates:
                continue
            if values is None:
                raise ValueError(f"{rate}_values is required")
            self._validate_stream(values, rate)
        daily_padding_mask = self._padding_mask(daily_padding_mask, daily_values)
        annual_padding_mask = self._padding_mask(annual_padding_mask, annual_values)
        quarterly_padding_mask = self._padding_mask(quarterly_padding_mask, quarterly_values)
        sparse_padding_mask = self._padding_mask(sparse_padding_mask, sparse_values) if sparse_values is not None else None
        daily_dates = daily_dates if daily_dates is not None else torch.arange(daily_values.shape[1], device=daily_values.device)
        annual_dates = annual_dates if annual_dates is not None else torch.arange(annual_values.shape[1], device=annual_values.device)
        quarterly_dates = quarterly_dates if quarterly_dates is not None else torch.arange(quarterly_values.shape[1], device=quarterly_values.device)
        sparse_dates = (
            sparse_dates if sparse_dates is not None else torch.arange(sparse_values.shape[1], device=sparse_values.device)
        ) if sparse_values is not None else None
        daily_mask = build_attention_mask(daily_dates, mode=attention_mode)
        annual_mask = build_attention_mask(annual_dates, mode=attention_mode)
        quarterly_mask = build_attention_mask(quarterly_dates, mode=attention_mode)
        sparse_mask = build_attention_mask(sparse_dates, mode=attention_mode) if sparse_dates is not None else None

        # Document tasks and family classification are defined over mean
        # subtoken embeddings; retain subtoken states whenever either class
        # of task is present, not only when next/masked prediction is used.
        need_subtokens = bool(
            self.prediction_task_specs
            or any(task.level == "document" for task in self.task_specs)
            or self.family_classification_head is not None
        )
        daily_projected = self._project(
            "daily", daily_values, daily_family_presence, daily_modality_ids,
            attention_mode, daily_dates, daily_padding_mask,
            need_subtokens,
        )
        annual_projected = self._project(
            "annual", annual_values, annual_family_presence, annual_modality_ids,
            attention_mode, annual_dates, annual_padding_mask,
            need_subtokens,
        )
        quarterly_projected = self._project(
            "quarterly", quarterly_values, quarterly_family_presence, quarterly_modality_ids,
            attention_mode, quarterly_dates, quarterly_padding_mask,
            need_subtokens,
        )
        sparse_projected = self._project(
            "sparse", sparse_values, sparse_family_presence, sparse_modality_ids,
            attention_mode, sparse_dates, sparse_padding_mask,
            need_subtokens,
        ) if sparse_values is not None else None
        if need_subtokens:
            daily_input, daily_subtokens = daily_projected
            annual_input, annual_subtokens = annual_projected
            quarterly_input, quarterly_subtokens = quarterly_projected
            if sparse_projected is not None:
                sparse_input, sparse_subtokens = sparse_projected
            else:
                sparse_input = sparse_subtokens = None
        else:
            daily_input, annual_input, quarterly_input = daily_projected, annual_projected, quarterly_projected
            sparse_input = sparse_subtokens = None if sparse_projected is None else sparse_projected
            daily_subtokens = annual_subtokens = quarterly_subtokens = None
        if self.config.backbone == "encoder_only":
            daily_states = self.encoders["daily"](daily_input, mask=daily_mask, src_key_padding_mask=daily_padding_mask)
            annual_states = self.encoders["annual"](annual_input, mask=annual_mask, src_key_padding_mask=annual_padding_mask)
            quarterly_states = self.encoders["quarterly"](quarterly_input, mask=quarterly_mask, src_key_padding_mask=quarterly_padding_mask)
            sparse_states = self.encoders["sparse"](sparse_input, mask=sparse_mask, src_key_padding_mask=sparse_padding_mask) if sparse_input is not None else None
        elif self.config.backbone == "decoder_only":
            inputs = [annual_input, quarterly_input, daily_input]
            dates = [annual_dates, quarterly_dates, daily_dates]
            if sparse_input is not None:
                inputs.append(sparse_input)
                dates.append(sparse_dates)
            combined = torch.cat(inputs, dim=1)
            combined_dates = torch.cat(dates, dim=0)
            combined_mask = build_attention_mask(combined_dates, mode=attention_mode)
            combined_padding = None
            if any(mask is not None for mask in (annual_padding_mask, quarterly_padding_mask, daily_padding_mask, sparse_padding_mask)):
                masks = [mask if mask is not None else torch.zeros(values.shape[:2], dtype=torch.bool, device=values.device) for mask, values in (
                    (annual_padding_mask, annual_values), (quarterly_padding_mask, quarterly_values), (daily_padding_mask, daily_values), (sparse_padding_mask, sparse_values)
                )]
                combined_padding = torch.cat(masks, dim=1)
            combined_states = self.decoder_only_stack(combined, mask=combined_mask, src_key_padding_mask=combined_padding)
            annual_end = annual_values.shape[1]
            quarterly_end = annual_end + quarterly_values.shape[1]
            daily_end = quarterly_end + daily_values.shape[1]
            annual_states = combined_states[:, :annual_end]
            quarterly_states = combined_states[:, annual_end:quarterly_end]
            daily_states = combined_states[:, quarterly_end:daily_end]
            sparse_states = combined_states[:, daily_end:] if sparse_input is not None else None
        elif self.config.backbone == "encoder_decoder":
            annual_states = self.annual_encoder(annual_input, mask=annual_mask, src_key_padding_mask=annual_padding_mask)
            quarterly_states = self.quarterly_encoder(quarterly_input, mask=quarterly_mask, src_key_padding_mask=quarterly_padding_mask)
            sparse_states = self.encoders["sparse"](sparse_input, mask=sparse_mask, src_key_padding_mask=sparse_padding_mask) if sparse_input is not None else None
            memory_parts = [annual_states, quarterly_states]
            if sparse_states is not None:
                memory_parts.append(sparse_states)
            memory = torch.cat(memory_parts, dim=1)
            memory_padding = None
            if annual_padding_mask is not None or quarterly_padding_mask is not None or sparse_padding_mask is not None:
                annual_padding_mask = annual_padding_mask if annual_padding_mask is not None else torch.zeros(annual_values.shape[:2], dtype=torch.bool, device=annual_values.device)
                quarterly_padding_mask = quarterly_padding_mask if quarterly_padding_mask is not None else torch.zeros(quarterly_values.shape[:2], dtype=torch.bool, device=quarterly_values.device)
                memory_parts = [annual_padding_mask, quarterly_padding_mask]
                if sparse_values is not None:
                    sparse_padding_mask = sparse_padding_mask if sparse_padding_mask is not None else torch.zeros(sparse_values.shape[:2], dtype=torch.bool, device=sparse_values.device)
                    memory_parts.append(sparse_padding_mask)
                memory_padding = torch.cat(memory_parts, dim=1)
            daily_states = self.daily_decoder(
                daily_input, memory, tgt_mask=daily_mask,
                tgt_key_padding_mask=daily_padding_mask,
                memory_key_padding_mask=memory_padding,
            )
        else:
            raise ValueError(f"unsupported backbone: {self.config.backbone!r}")
        # Padded causal rows can still yield NaNs in PyTorch Transformer
        # kernels when a stream has no usable family token. Keep those rows
        # inert so auxiliary document/token heads receive finite states.
        daily_states = torch.nan_to_num(daily_states, nan=0.0, posinf=0.0, neginf=0.0)
        annual_states = torch.nan_to_num(annual_states, nan=0.0, posinf=0.0, neginf=0.0)
        quarterly_states = torch.nan_to_num(quarterly_states, nan=0.0, posinf=0.0, neginf=0.0)
        if sparse_states is not None:
            sparse_states = torch.nan_to_num(sparse_states, nan=0.0, posinf=0.0, neginf=0.0)
        family_presence = {
            "daily": self.coverage_inputs["daily"](daily_values, return_family_states=True)[2],
            "annual": self.coverage_inputs["annual"](annual_values, return_family_states=True)[2],
            "quarterly": self.coverage_inputs["quarterly"](quarterly_values, return_family_states=True)[2],
            "sparse": self.coverage_inputs["sparse"](sparse_values, return_family_states=True)[2] if sparse_values is not None else None,
        }
        token_outputs, document_outputs, fused_document_state, family_document_state = self._heads(
            daily_states=daily_states,
            annual_states=annual_states,
            quarterly_states=quarterly_states,
            sparse_states=sparse_states,
            daily_padding_mask=daily_padding_mask,
            annual_padding_mask=annual_padding_mask,
            quarterly_padding_mask=quarterly_padding_mask,
            sparse_padding_mask=sparse_padding_mask,
            annual_subtokens=annual_subtokens,
            quarterly_subtokens=quarterly_subtokens,
            daily_subtokens=daily_subtokens,
            sparse_subtokens=sparse_subtokens,
            family_presence=family_presence,
        )
        prediction_outputs = self._prediction_heads(
            token_states={"daily": daily_states, "annual": annual_states, "quarterly": quarterly_states, "sparse": sparse_states},
            subtoken_states={"daily": daily_subtokens, "annual": annual_subtokens, "quarterly": quarterly_subtokens, "sparse": sparse_subtokens},
        ) if need_subtokens else {}
        family_outputs = (
            self.family_classification_head(fused_document_state)
            if self.family_classification_head is not None else None
        )
        return {
            "token_states": daily_states,
            "document_state": fused_document_state,
            "family_document_state": family_document_state,
            "rate_states": {
                "annual": annual_states,
                "quarterly": quarterly_states,
                "daily": daily_states,
                "sparse": sparse_states,
            },
            "token_outputs": token_outputs,
            "document_outputs": document_outputs,
            "prediction_outputs": prediction_outputs,
            "family_outputs": family_outputs,
        }


__all__ = [
    "AttentionMode",
    "BackboneName",
    "MultiRateTaskSpec",
    "MultiRatePredictionTaskSpec",
    "MultiRateTransformer",
    "MultiRateTransformerConfig",
    "PredictionLevel",
    "PredictionObjective",
    "build_attention_mask",
]
