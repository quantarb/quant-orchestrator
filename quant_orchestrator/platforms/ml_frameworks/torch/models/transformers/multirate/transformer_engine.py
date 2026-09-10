"""Optional NVIDIA Transformer Engine backbones for the multi-rate model."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


def _te_module() -> Any:
    try:
        import transformer_engine.pytorch as te
    except (ImportError, OSError) as exc:  # pragma: no cover - exercised on CUDA installs
        raise RuntimeError(
            "Transformer Engine backend requested but transformer_engine is not installed; "
            "install quant-orchestrator[cuda-te] on a supported NVIDIA CUDA environment"
        ) from exc
    return te


def _blocked_mask(
    mask: torch.Tensor | None,
    padding_mask: torch.Tensor | None,
    *,
    batch_size: int,
    query_length: int,
    key_length: int,
    device: torch.device,
) -> torch.Tensor:
    """Convert native PyTorch masks to TE's bool arbitrary-mask contract."""
    if mask is None:
        blocked = torch.zeros(
            (batch_size, 1, query_length, key_length), dtype=torch.bool,
            device=device,
        )
    else:
        if mask.ndim not in (2, 3) or mask.shape[-2:] != (query_length, key_length):
            raise ValueError("Transformer Engine masks must be [query, key]")
        blocked_2d = mask if mask.dtype == torch.bool else mask < 0
        if mask.ndim == 2:
            blocked = blocked_2d.to(torch.bool).view(1, 1, query_length, key_length).expand(batch_size, 1, query_length, key_length)
        else:
            blocked = blocked_2d.to(torch.bool).reshape(batch_size, -1, query_length, key_length)
    if padding_mask is not None:
        if padding_mask.shape != (batch_size, key_length):
            raise ValueError("padding mask must be [batch, key]")
        blocked = blocked | padding_mask.bool().view(batch_size, 1, 1, key_length)
    return blocked


def _fp8_length(length: int) -> int:
    """Return a token length accepted by TE's FP8 linear kernels."""
    return (length + 7) // 8 * 8


def _pad_attention_mask(
    mask: torch.Tensor | None,
    query_length: int,
    key_length: int,
    padded_query_length: int,
    padded_key_length: int,
) -> torch.Tensor | None:
    if mask is None:
        return None
    if mask.shape[-2:] != (query_length, key_length):
        raise ValueError("Transformer Engine masks must be [query, key]")
    if (query_length, key_length) == (padded_query_length, padded_key_length):
        return mask
    padded = torch.ones(
        (*mask.shape[:-2], padded_query_length, padded_key_length), dtype=torch.bool, device=mask.device,
    )
    padded[..., :query_length, :key_length] = mask.bool() if mask.dtype == torch.bool else mask < 0
    return padded


def _pad_key_mask(
    mask: torch.Tensor | None,
    batch_size: int,
    key_length: int,
    padded_key_length: int,
) -> torch.Tensor | None:
    if mask is None:
        return None
    if mask.shape != (batch_size, key_length):
        raise ValueError("padding mask must be [batch, key]")
    if key_length == padded_key_length:
        return mask
    padded = torch.ones((batch_size, padded_key_length), dtype=torch.bool, device=mask.device)
    padded[:, :key_length] = mask.bool()
    return padded


class TransformerEngineEncoder(nn.Module):
    """Drop-in ``TransformerEncoder``-like wrapper backed by TE layers."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        te = _te_module()
        self.layers = nn.ModuleList([
            te.TransformerLayer(
                hidden_size=config.d_model,
                ffn_hidden_size=config.d_model * 4,
                num_attention_heads=config.num_heads,
                layernorm_epsilon=1e-5,
                hidden_dropout=config.dropout,
                attention_dropout=config.dropout,
                layer_type="encoder",
                self_attn_mask_type="arbitrary",
            )
            for _ in range(config.layers)
        ])

    def forward(
        self,
        values: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        original_length = values.shape[1]
        padded_length = _fp8_length(original_length)
        mask = _pad_attention_mask(mask, original_length, original_length, padded_length, padded_length)
        src_key_padding_mask = _pad_key_mask(
            src_key_padding_mask, values.shape[0], original_length, padded_length,
        )
        if padded_length != original_length:
            values = nn.functional.pad(values, (0, 0, 0, padded_length - original_length))
        attention_mask = _blocked_mask(
            mask,
            src_key_padding_mask,
            batch_size=values.shape[0],
            query_length=padded_length,
            key_length=padded_length,
            device=values.device,
        )
        values = values.transpose(0, 1).contiguous()
        for layer in self.layers:
            values = layer(
                values,
                attention_mask=attention_mask,
                self_attn_mask_type="arbitrary",
            )
        return values[:original_length].transpose(0, 1).contiguous()


class TransformerEngineDecoder(nn.Module):
    """Drop-in ``TransformerDecoder``-like wrapper backed by TE layers."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        te = _te_module()
        self.layers = nn.ModuleList([
            te.TransformerLayer(
                hidden_size=config.d_model,
                ffn_hidden_size=config.d_model * 4,
                num_attention_heads=config.num_heads,
                layernorm_epsilon=1e-5,
                hidden_dropout=config.dropout,
                attention_dropout=config.dropout,
                layer_type="decoder",
                self_attn_mask_type="arbitrary",
                enc_dec_attn_mask_type="arbitrary",
            )
            for _ in range(config.layers)
        ])

    def forward(
        self,
        target: torch.Tensor,
        memory: torch.Tensor,
        *,
        tgt_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        original_target_length = target.shape[1]
        original_memory_length = memory.shape[1]
        padded_target_length = _fp8_length(original_target_length)
        padded_memory_length = _fp8_length(original_memory_length)
        tgt_mask = _pad_attention_mask(
            tgt_mask, original_target_length, original_target_length,
            padded_target_length, padded_target_length,
        )
        tgt_key_padding_mask = _pad_key_mask(
            tgt_key_padding_mask, target.shape[0], original_target_length, padded_target_length,
        )
        memory_key_padding_mask = _pad_key_mask(
            memory_key_padding_mask, memory.shape[0], original_memory_length, padded_memory_length,
        )
        if padded_target_length != original_target_length:
            target = nn.functional.pad(target, (0, 0, 0, padded_target_length - original_target_length))
        if padded_memory_length != original_memory_length:
            memory = nn.functional.pad(memory, (0, 0, 0, padded_memory_length - original_memory_length))
        self_mask = _blocked_mask(
            tgt_mask,
            tgt_key_padding_mask,
            batch_size=target.shape[0],
            query_length=padded_target_length,
            key_length=padded_target_length,
            device=target.device,
        )
        memory_mask = _blocked_mask(
            None,
            memory_key_padding_mask,
            batch_size=target.shape[0],
            query_length=padded_target_length,
            key_length=padded_memory_length,
            device=target.device,
        )
        target = target.transpose(0, 1).contiguous()
        memory = memory.transpose(0, 1).contiguous()
        for layer in self.layers:
            target = layer(
                target,
                attention_mask=self_mask,
                self_attn_mask_type="arbitrary",
                encoder_output=memory,
                enc_dec_attn_mask=memory_mask,
                enc_dec_attn_mask_type="arbitrary",
            )
        return target[:original_target_length].transpose(0, 1).contiguous()


def transformer_engine_available() -> bool:
    try:
        _te_module()
    except RuntimeError:
        return False
    return True
