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
from collections import OrderedDict
from typing import Callable, Hashable, Literal, Mapping, Sequence

import torch
from torch import nn

from quant_orchestrator.platforms.ml_frameworks.torch.models.transformers.multirate.coverage import (
    CoverageAwareInput,
)
from quant_orchestrator.platforms.ml_frameworks.torch.models.transformers.multirate.auto_features import (
    AutoFeatureEngineer,
    AGGREGATION_FUNCTIONS,
)


BackboneName = Literal["encoder_only", "decoder_only", "encoder_decoder"]
AttentionMode = Literal["temporal", "cross_sectional"]
DocumentPool = Literal["last", "mean"]
DOCUMENT_PROTOTYPE_STATS = ("mean", "min", "max", "rmse", "q25", "q50", "q75")


def pool_prototype_statistics(
    states: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    """Reduce ``[batch, rows, features]`` into reusable row prototypes.

    This is intentionally independent of the transformer so the same reducer
    can be used for collaborative-filtering matrices: rows may represent
    users, assets, dates, or any other entities, while columns remain the
    feature dimensions.
    """
    if states.ndim != 3 or states.shape[1] == 0:
        raise ValueError("states must have shape [batch, rows, features] with non-empty rows")
    clean = torch.nan_to_num(states, nan=0.0, posinf=0.0, neginf=0.0)
    valid = (
        torch.ones(states.shape[:2], dtype=torch.bool, device=states.device)
        if valid_mask is None else valid_mask.bool()
    )
    weights = valid.unsqueeze(-1).to(clean.dtype)
    count = weights.sum(dim=1).clamp_min(1.0)
    mean = (clean * weights).sum(dim=1) / count
    min_values = clean.masked_fill(~valid.unsqueeze(-1), float("inf")).amin(dim=1)
    max_values = clean.masked_fill(~valid.unsqueeze(-1), float("-inf")).amax(dim=1)
    rmse_variance = (((clean - mean.unsqueeze(1)).square()) * weights).sum(dim=1) / count
    rmse = torch.sqrt(rmse_variance.clamp_min(torch.finfo(states.dtype).eps))
    # Keep the quantile path finite for autograd. ``nanquantile`` can produce
    # NaN gradients for fully masked rows, even when its result is sanitized.
    # Sorting finite values and interpolating valid positions avoids that.
    sorted_values = clean.masked_fill(
        ~valid.unsqueeze(-1), torch.finfo(states.dtype).max,
    ).sort(dim=1).values
    has_valid = valid.any(dim=1).unsqueeze(-1)
    count = valid.sum(dim=1).clamp_min(1).to(states.dtype)
    quantile_values = []
    for quantile in (0.25, 0.50, 0.75):
        position = quantile * (count - 1.0)
        lower_index = position.floor().long().unsqueeze(1).unsqueeze(-1)
        upper_index = position.ceil().long().unsqueeze(1).unsqueeze(-1)
        lower = sorted_values.gather(1, lower_index.expand(-1, 1, states.shape[-1])).squeeze(1)
        upper = sorted_values.gather(1, upper_index.expand(-1, 1, states.shape[-1])).squeeze(1)
        quantile = lower + (upper - lower) * (position - position.floor()).unsqueeze(-1)
        quantile_values.append(torch.where(has_valid, quantile, torch.zeros_like(quantile)))
    return (
        mean,
        torch.nan_to_num(min_values, nan=0.0, posinf=0.0, neginf=0.0),
        torch.nan_to_num(max_values, nan=0.0, posinf=0.0, neginf=0.0),
        torch.nan_to_num(rmse, nan=0.0, posinf=0.0, neginf=0.0),
        *(torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0) for values in quantile_values),
    )
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
    learned_aggregation_gate: bool = False
    cacheable_rate_states: bool = True

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


class IssuerContextCache:
    """Inference-time cache for annual/quarterly issuer states.

    A key must include the issuer and the as-of date (and, when applicable,
    the feature/version identity). Cached states are detached deliberately:
    this cache is for evaluation/inference, not gradient-bearing training.
    """

    def __init__(self, max_entries: int = 4096, cache_dtype: torch.dtype | None = None) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self.max_entries = max_entries
        self.cache_dtype = cache_dtype
        self._values: OrderedDict[Hashable, dict[str, dict[str, torch.Tensor]]] = OrderedDict()

    def get_or_compute(
        self,
        key: Hashable,
        factory: Callable[[], Mapping[str, Mapping[str, torch.Tensor]]],
    ) -> dict[str, dict[str, torch.Tensor]]:
        if key in self._values:
            value = self._values.pop(key)
            self._values[key] = value
            return value
        with torch.no_grad():
            source = factory()
        value = {
            rate: {name: tensor.detach().to(dtype=self.cache_dtype) if self.cache_dtype is not None and tensor.is_floating_point() else tensor.detach() for name, tensor in payload.items()}
            for rate, payload in source.items()
        }
        self._values[key] = value
        while len(self._values) > self.max_entries:
            self._values.popitem(last=False)
        return value

    def clear(self) -> None:
        self._values.clear()

    def __len__(self) -> int:
        return len(self._values)


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
        # This path never receives annual/quarterly/sparse memory. Its output
        # is the reusable instrument-only representation.
        self.instrument_encoder = _encoder(self.config)
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
        self.auto_feature_engineer.aggregation_gate.configure_families(len(self.family_names))
        self.family_document_fusion = nn.Sequential(
            nn.Linear(self.config.d_model * len(self.config.rates), self.config.d_model),
            nn.LayerNorm(self.config.d_model),
            nn.GELU(),
        )
        self.document_prototype_count = len(DOCUMENT_PROTOTYPE_STATS)
        self.family_document_prototype_count = self.document_prototype_count + int(self.config.learned_aggregation_gate)
        self.document_prototype_dim = self.config.d_model * self.document_prototype_count
        self.family_document_prototype_dim = self.config.d_model * self.family_document_prototype_count
        self.task_specs = tuple(tasks)
        self.prediction_task_specs = tuple(prediction_tasks)
        self.family_classification_head = (
            nn.Linear(self.family_document_prototype_dim, int(family_classification_dim))
            if family_classification_dim is not None else None
        )
        names = [task.task_name for task in self.task_specs]
        if len(names) != len(set(names)):
            raise ValueError("task names must be unique")
        self.task_heads = nn.ModuleDict({
            task.task_name: nn.Linear(
                (self.family_document_prototype_dim if task.source == "family" else self.document_prototype_dim)
                if task.level == "document" else self.config.d_model,
                task.output_dim,
            )
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
    def _pool_prototypes(
        states: torch.Tensor,
        padding_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, ...]:
        """Return all configured prototypes over valid temporal rows."""
        valid = None if padding_mask is None else ~padding_mask.bool()
        return pool_prototype_statistics(states, valid)

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
    def _pool_subtoken_tokens(
        states: torch.Tensor,
        padding_mask: torch.Tensor | None,
        family_presence: torch.Tensor | None,
    ) -> torch.Tensor:
        """Pool family subtokens into one token state per temporal position."""
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
        return (clean * weights).sum(dim=2) / weights.sum(dim=2).clamp_min(1.0)

    @staticmethod
    def _pool_family_documents(
        states: torch.Tensor,
        padding_mask: torch.Tensor | None,
        family_presence: torch.Tensor | None,
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        """Pool each family document into the configured prototypes."""
        if states.ndim != 4:
            raise ValueError("subtoken states must have shape [batch, sequence, family, d_model]")
        batch, length, family_count, _ = states.shape
        valid = torch.ones((batch, length, family_count), dtype=torch.bool, device=states.device)
        if padding_mask is not None:
            valid &= ~padding_mask.bool().unsqueeze(-1)
        if family_presence is not None:
            valid &= family_presence.bool()
        flat_states = states.permute(0, 2, 1, 3).reshape(batch * family_count, length, -1)
        flat_valid = valid.permute(0, 2, 1).reshape(batch * family_count, length)
        pooled = pool_prototype_statistics(flat_states, flat_valid)
        return (
            tuple(values.reshape(batch, family_count, -1) for values in pooled),
            valid.any(dim=1),
        )

    def _heads(
        self,
        *,
        compute_document_outputs: bool,
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
        token_states: Mapping[str, torch.Tensor],
    ) -> tuple[
        dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor,
    ]:
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
        if not compute_document_outputs:
            # Training runs that do not include document tasks should not pay
            # for mean/min/max document prototype pooling. Keep the fused
            # mean state because token-level heads still use it.
            rate_states = {
                rate: self._pool_subtoken_tokens(
                    subtokens, padding_masks[rate], family_presence[rate],
                ).mean(dim=1) if subtokens is not None else states.mean(dim=1)
                for rate, subtokens, states in (
                    ("annual", annual_subtokens, annual_states),
                    ("quarterly", quarterly_subtokens, quarterly_states),
                    ("daily", daily_subtokens, daily_states),
                    ("sparse", sparse_subtokens, sparse_states),
                ) if rate in self.config.rates and states is not None
            }
            fused = self.fusion(torch.cat(tuple(rate_states[rate] for rate in self.config.rates), dim=-1))
            fused = fused + self.auto_feature_engineer.cross_rate_features(
                tuple(rate_states[rate] for rate in self.config.rates)
            )
            token_outputs: dict[str, torch.Tensor] = {}
            sources = dict(token_states)
            sources["fused"] = fused
            for task in self.task_specs:
                if task.level == "token":
                    token_outputs[task.task_name] = self.task_heads[task.task_name](sources[task.source])
            return token_outputs, {}, fused, None, None, None
        rate_prototypes = {
            rate: self._pool_prototypes(
                self._pool_subtoken_tokens(subtoken_states[rate], padding_masks[rate], family_presence[rate]),
                None,
            )
            if subtoken_states[rate] is not None
            else self._pool_prototypes(states, padding_masks[rate])
            for rate, states in (
                ("annual", annual_states), ("quarterly", quarterly_states),
                ("daily", daily_states), ("sparse", sparse_states),
            ) if rate in self.config.rates
        }
        rate_states = {rate: prototypes[0] for rate, prototypes in rate_prototypes.items()}
        family_documents: dict[str, torch.Tensor] = {}
        for rate, subtokens in subtoken_states.items():
            if subtokens is None:
                continue
            family_documents[rate], _ = self._pool_family_documents(
                subtokens, padding_masks[rate], family_presence[rate],
            )
            if self.config.learned_aggregation_gate:
                family_indices = torch.tensor(
                    [self.family_indices[name] for name in self.coverage_inputs[rate].family_names],
                    device=daily_states.device,
                    dtype=torch.long,
                )
                gated, _ = self.auto_feature_engineer.gate_aggregations(
                    tuple(family_documents[rate]), family_indices=family_indices,
                )
                family_documents[rate] = (*family_documents[rate], gated)
        total_families = len(self.family_names)
        family_prototype_parts = [[] for _ in range(self.family_document_prototype_count)]
        for rate in self.config.rates:
            for stat_index in range(self.family_document_prototype_count):
                part = torch.zeros(
                    (daily_states.shape[0], total_families, self.config.d_model),
                    device=daily_states.device, dtype=daily_states.dtype,
                )
                if rate in family_documents:
                    for local_index, name in enumerate(self.coverage_inputs[rate].family_names):
                        part[:, self.family_indices[name]] = family_documents[rate][stat_index][:, local_index]
                family_prototype_parts[stat_index].append(part)
        family_document_prototypes = torch.cat(
            [self.family_document_fusion(torch.cat(parts, dim=-1)) for parts in family_prototype_parts],
            dim=-1,
        )
        family_document_state = family_document_prototypes[..., :self.config.d_model]
        ordered = tuple(rate_states[rate] for rate in self.config.rates)
        fused = self.fusion(torch.cat(ordered, dim=-1))
        fused_prototypes = torch.cat(
            [self.fusion(torch.cat([rate_prototypes[rate][stat_index] for rate in self.config.rates], dim=-1))
             for stat_index in range(self.document_prototype_count)],
            dim=-1,
        )
        fused = fused + self.auto_feature_engineer.cross_rate_features(ordered)
        token_outputs: dict[str, torch.Tensor] = {}
        document_outputs: dict[str, torch.Tensor] = {}
        sources = dict(token_states)
        sources["fused"] = fused
        document_sources = {
            **{rate: torch.cat(rate_prototypes[rate], dim=-1) for rate in self.config.rates},
            "fused": fused_prototypes,
            "family": family_document_prototypes,
        }
        for task in self.task_specs:
            source = document_sources[task.source] if task.level == "document" else sources[task.source]
            source = torch.nan_to_num(source, nan=0.0, posinf=0.0, neginf=0.0)
            if task.level == "token":
                token_outputs[task.task_name] = self.task_heads[task.task_name](source)
            else:
                document_outputs[task.task_name] = self.task_heads[task.task_name](source)
        return (
            token_outputs, document_outputs, fused, family_document_state,
            fused_prototypes, family_document_prototypes,
        )

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

    @staticmethod
    def _cached_state(rate, cache, values, encoder, mask, padding_mask, context_ids=None):
        # PyTorch warns (and will eventually reject) mixed mask types. The
        # temporal mask is additive, so represent padding in the same form
        # while preserving the boolean-mask semantics (True means blocked).
        if mask is not None and padding_mask is not None and mask.is_floating_point() and not padding_mask.is_floating_point():
            padding_mask = padding_mask.to(dtype=mask.dtype).masked_fill(padding_mask, float("-inf"))
        cached = cache.get(rate)
        if cached is not None and "states" in cached:
            states = cached["states"].to(dtype=values.dtype, device=values.device)
            if states.shape[0] == 1 and values.shape[0] > 1:
                states = states.expand(values.shape[0], *states.shape[1:])
            return states
        # Training-safe token reuse: duplicate issuer contexts are encoded
        # once, while the returned states retain one row per instrument. The
        # gathered tensor remains attached to the computation graph, so all
        # instruments contribute gradients to the shared rate encoder.
        if context_ids is not None and values.shape[0] > 1:
            unique_ids, inverse = torch.unique(context_ids, sorted=True, return_inverse=True)
            if unique_ids.numel() < values.shape[0]:
                first = torch.stack([(context_ids == identifier).nonzero(as_tuple=False)[0, 0] for identifier in unique_ids])
                unique_values = values.index_select(0, first)
                unique_mask = mask.index_select(0, first) if mask is not None and mask.ndim > 1 else mask
                unique_padding = padding_mask.index_select(0, first) if padding_mask is not None else None
                unique_states = encoder(unique_values, mask=mask, src_key_padding_mask=unique_padding)
                return unique_states.index_select(0, inverse)
        return encoder(values, mask=mask, src_key_padding_mask=padding_mask)

    @staticmethod
    def issuer_context_from_output(output: Mapping[str, object]) -> dict[str, dict[str, torch.Tensor]]:
        """Extract reusable annual/quarterly/sparse issuer streams."""
        cache = MultiRateTransformer.rate_cache_from_output(output)
        return {rate: cache[rate] for rate in ("annual", "quarterly", "sparse") if rate in cache}

    @staticmethod
    def rate_cache_from_output(output: Mapping[str, object]) -> dict[str, dict[str, torch.Tensor]]:
        """Extract all reusable per-rate states from a prior model result."""
        cache = output.get("rate_cache")
        if not isinstance(cache, Mapping):
            raise ValueError("model output does not contain rate_cache")
        result = {}
        for rate in ("annual", "quarterly", "daily", "sparse"):
            payload = cache.get(rate)
            if isinstance(payload, Mapping) and "projected" in payload and "states" in payload:
                result[rate] = {
                    "projected": payload["projected"],
                    "states": payload["states"],
                }
        return result

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
        rate_cache: Mapping[str, Mapping[str, torch.Tensor]] | None = None,
        issuer_context_cache: IssuerContextCache | None = None,
        issuer_context_key: Hashable | None = None,
        rate_context_ids: Mapping[str, torch.Tensor] | None = None,
        compute_document_outputs: bool = True,
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

        # Document tasks and family classification use mean/min/max subtoken
        # prototypes; retain subtoken states whenever either class of task is
        # present, not only when next/masked prediction is used.
        need_subtokens = bool(
            self.prediction_task_specs
            or any(task.level == "token" for task in self.task_specs)
            or any(task.level == "document" for task in self.task_specs)
            or self.family_classification_head is not None
        )
        rate_cache = dict(rate_cache or {})

        def compute_issuer_context() -> Mapping[str, Mapping[str, torch.Tensor]]:
            local_cache = {}
            for rate, values, family_presence, modality_ids, dates, padding, encoder in (
                ("annual", annual_values, annual_family_presence, annual_modality_ids, annual_dates, annual_padding_mask, self.annual_encoder),
                ("quarterly", quarterly_values, quarterly_family_presence, quarterly_modality_ids, quarterly_dates, quarterly_padding_mask, self.quarterly_encoder),
            ):
                projected = self._project(rate, values, family_presence, modality_ids, attention_mode, dates, padding, need_subtokens)
                if need_subtokens:
                    projected_values, subtokens = projected
                else:
                    projected_values, subtokens = projected, None
                states = encoder(projected_values, mask=build_attention_mask(dates, mode=attention_mode), src_key_padding_mask=padding)
                local_cache[rate] = {"projected": projected_values.detach(), "states": states.detach()}
                if subtokens is not None:
                    local_cache[rate]["subtokens"] = subtokens.detach()
            return local_cache

        if issuer_context_cache is not None and issuer_context_key is not None:
            cached_context = issuer_context_cache.get_or_compute(issuer_context_key, compute_issuer_context)
            rate_cache = {**cached_context, **rate_cache}

        def cached_or_project(rate: str, values: torch.Tensor | None, family_presence, modality_ids, dates, padding):
            cached = rate_cache.get(rate)
            if cached is not None and "projected" in cached:
                target_dtype = values.dtype if values is not None else daily_values.dtype
                projected = cached["projected"].to(dtype=target_dtype, device=daily_values.device)
                subtokens = cached.get("subtokens")
                if subtokens is not None:
                    subtokens = subtokens.to(dtype=target_dtype, device=daily_values.device)
                batch = values.shape[0] if values is not None else projected.shape[0]
                if projected.shape[0] == 1 and batch > 1:
                    projected = projected.expand(batch, *projected.shape[1:])
                    if subtokens is not None:
                        subtokens = subtokens.expand(batch, *subtokens.shape[1:])
                return (projected, subtokens) if need_subtokens else projected
            return self._project(rate, values, family_presence, modality_ids, attention_mode, dates, padding, need_subtokens)

        daily_projected = cached_or_project(
            "daily", daily_values, daily_family_presence, daily_modality_ids,
            daily_dates, daily_padding_mask,
        )
        instrument_input = daily_projected[0] if need_subtokens else daily_projected
        context_ids = rate_context_ids or {}
        instrument_states = self._cached_state(
            "daily", rate_cache, instrument_input, self.encoders["daily"], daily_mask, daily_padding_mask,
        )
        annual_projected = cached_or_project("annual", annual_values, annual_family_presence, annual_modality_ids, annual_dates, annual_padding_mask)
        quarterly_projected = cached_or_project("quarterly", quarterly_values, quarterly_family_presence, quarterly_modality_ids, quarterly_dates, quarterly_padding_mask)
        sparse_projected = cached_or_project("sparse", sparse_values, sparse_family_presence, sparse_modality_ids, sparse_dates, sparse_padding_mask) if sparse_values is not None else None
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
            sparse_input = sparse_projected
            sparse_subtokens = None
            daily_subtokens = annual_subtokens = quarterly_subtokens = None
        if self.config.backbone == "encoder_only":
            daily_states = self._cached_state("daily", rate_cache, daily_input, self.encoders["daily"], daily_mask, daily_padding_mask)
            annual_states = self._cached_state("annual", rate_cache, annual_input, self.encoders["annual"], annual_mask, annual_padding_mask)
            quarterly_states = self._cached_state("quarterly", rate_cache, quarterly_input, self.encoders["quarterly"], quarterly_mask, quarterly_padding_mask)
            sparse_states = self._cached_state("sparse", rate_cache, sparse_input, self.encoders["sparse"], sparse_mask, sparse_padding_mask) if sparse_input is not None else None
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
            annual_states = self._cached_state("annual", rate_cache, annual_input, self.annual_encoder, annual_mask, annual_padding_mask, context_ids.get("annual"))
            quarterly_states = self._cached_state("quarterly", rate_cache, quarterly_input, self.quarterly_encoder, quarterly_mask, quarterly_padding_mask, context_ids.get("quarterly"))
            sparse_states = self._cached_state("sparse", rate_cache, sparse_input, self.encoders["sparse"], sparse_mask, sparse_padding_mask, context_ids.get("sparse")) if sparse_input is not None else None
            if self.config.cacheable_rate_states:
                # All rate states are independent intermediate products. This
                # is the fast path: the decoder/task fusion can be rerun for
                # each instrument while the four streams remain reusable.
                daily_states = self._cached_state("daily", rate_cache, daily_input, self.encoders["daily"], daily_mask, daily_padding_mask)
            else:
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
        token_states = {
            rate: self._pool_subtoken_tokens(
                subtoken_states,
                {"annual": annual_padding_mask, "quarterly": quarterly_padding_mask,
                 "daily": daily_padding_mask, "sparse": sparse_padding_mask}[rate],
                family_presence[rate],
            )
            if subtoken_states is not None else states
            for rate, subtoken_states, states in (
                ("annual", annual_subtokens, annual_states),
                ("quarterly", quarterly_subtokens, quarterly_states),
                ("daily", daily_subtokens, daily_states),
                ("sparse", sparse_subtokens, sparse_states),
            ) if rate in self.config.rates and states is not None
        }
        (
            token_outputs, document_outputs, fused_document_state, family_document_state,
            fused_document_prototypes, family_document_prototypes,
        ) = self._heads(
            compute_document_outputs=compute_document_outputs,
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
            token_states=token_states,
        )
        prediction_outputs = self._prediction_heads(
            token_states=token_states,
            subtoken_states={"daily": daily_subtokens, "annual": annual_subtokens, "quarterly": quarterly_subtokens, "sparse": sparse_subtokens},
        ) if need_subtokens else {}
        family_outputs = (
            self.family_classification_head(fused_document_prototypes)
            if self.family_classification_head is not None and fused_document_prototypes is not None else None
        )
        return {
            "token_states": token_states["daily"],
            "instrument_states": torch.nan_to_num(instrument_states, nan=0.0, posinf=0.0, neginf=0.0),
            "document_state": fused_document_state,
            "family_document_state": family_document_state,
            "document_prototypes": fused_document_prototypes,
            "family_document_prototypes": family_document_prototypes,
            "rate_states": {
                "annual": annual_states,
                "quarterly": quarterly_states,
                "daily": daily_states,
                "sparse": sparse_states,
            },
            "rate_cache": {
                rate: {
                    "projected": projected[0] if isinstance(projected, tuple) else projected,
                    "subtokens": projected[1] if isinstance(projected, tuple) else None,
                    "states": states,
                }
                for rate, projected, states in (
                    ("annual", annual_projected, annual_states),
                    ("quarterly", quarterly_projected, quarterly_states),
                    ("daily", daily_projected, instrument_states),
                    ("sparse", sparse_projected, sparse_states),
                ) if projected is not None and states is not None
            },
            "token_outputs": token_outputs,
            "document_outputs": document_outputs,
            "prediction_outputs": prediction_outputs,
            "family_outputs": family_outputs,
        }


__all__ = [
    "AttentionMode",
    "BackboneName",
    "DOCUMENT_PROTOTYPE_STATS",
    "pool_prototype_statistics",
    "MultiRateTaskSpec",
    "MultiRatePredictionTaskSpec",
    "MultiRateTransformer",
    "MultiRateTransformerConfig",
    "IssuerContextCache",
    "PredictionLevel",
    "PredictionObjective",
    "build_attention_mask",
]
