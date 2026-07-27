"""Frozen deterministic key/value text embeddings for mixed feature rows."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch


def canonical_row_text(frame: pd.DataFrame, columns: Sequence[str]) -> list[str]:
    """Serialize nonnumeric row fields deterministically as sorted key/value text."""
    available = tuple(sorted(column for column in columns if column in frame.columns))
    texts: list[str] = []
    for row in frame.loc[:, available].itertuples(index=False, name=None):
        pairs = []
        for key, value in zip(available, row, strict=True):
            if value is None or pd.isna(value):
                value = "<missing>"
            pairs.append(f"{key}={str(value).strip().casefold() or '<missing>'}")
        texts.append(" | ".join(pairs))
    return texts


@torch.inference_mode()
def encode_frozen_text_rows(
    texts: Sequence[str],
    *,
    model_name: str = "axiotic/ogma-small",
    batch_size: int = 512,
    device: str | torch.device = "cpu",
    local_files_only: bool = True,
) -> np.ndarray:
    """Encode unique deterministic text rows with a frozen tiny Transformer."""
    from transformers import AutoModel, AutoTokenizer

    unique = list(dict.fromkeys(str(text) for text in texts))
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, local_files_only=local_files_only, trust_remote_code=True,
    )
    model = AutoModel.from_pretrained(
        model_name, local_files_only=local_files_only, trust_remote_code=True,
    ).to(device).eval()
    vectors: list[np.ndarray] = []
    for start in range(0, len(unique), max(1, int(batch_size))):
        batch = unique[start:start + max(1, int(batch_size))]
        encoded = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        model_output = model(**encoded)
        # Ogma returns the hidden-state tensor directly, while standard
        # Hugging Face encoders return a BaseModelOutput object.
        output = model_output if isinstance(model_output, torch.Tensor) else model_output.last_hidden_state
        if output.ndim == 2:
            # Ogma's native forward already returns one pooled vector per row.
            pooled = output
        else:
            mask = encoded["attention_mask"].unsqueeze(-1).to(output.dtype)
            pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        vectors.append(pooled.float().cpu().numpy())
    lookup = dict(zip(unique, np.concatenate(vectors, axis=0), strict=True))
    return np.asarray([lookup[str(text)] for text in texts], dtype="float32")


__all__ = ["canonical_row_text", "encode_frozen_text_rows"]
