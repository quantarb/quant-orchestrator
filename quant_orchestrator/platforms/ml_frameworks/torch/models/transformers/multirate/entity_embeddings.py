"""Global nonnumeric entity vocabulary and learned embeddings."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn


def canonical_entity(value: object) -> str:
    """Normalize one nonnumeric value into the shared global vocabulary."""

    if value is None or pd.isna(value):
        return "<missing>"
    return " ".join(str(value).strip().casefold().split()) or "<missing>"


class GlobalEntityVocabulary:
    """One field-independent vocabulary shared by all feature and target data."""

    def __init__(self) -> None:
        self.values = ["<pad>", "<unknown>", "<missing>"]
        self.ids = {value: index for index, value in enumerate(self.values)}

    def fit(self, values: Iterable[object]) -> "GlobalEntityVocabulary":
        for value in values:
            key = canonical_entity(value)
            if key not in self.ids:
                self.ids[key] = len(self.values)
                self.values.append(key)
        return self

    def fit_frame(self, frame: pd.DataFrame, columns: Sequence[str]) -> "GlobalEntityVocabulary":
        for column in columns:
            if column in frame:
                self.fit(frame[column].tolist())
        return self

    def encode_rows(self, frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
        output = np.full((len(frame), len(columns)), self.ids["<missing>"], dtype="int64")
        unknown = self.ids["<unknown>"]
        for index, column in enumerate(columns):
            if column not in frame:
                continue
            output[:, index] = [self.ids.get(canonical_entity(value), unknown) for value in frame[column]]
        return output

    def state(self) -> dict[str, object]:
        return {"values": self.values}

    @property
    def size(self) -> int:
        return len(self.values)


class GlobalEntityEmbedding(nn.Module):
    """Pool field-independent entity embeddings into the model state."""

    def __init__(self, vocabulary_size: int, d_model: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, entity_ids: torch.Tensor) -> torch.Tensor:
        if entity_ids.ndim != 2:
            raise ValueError("entity_ids must have shape [batch, entities]")
        states = self.embedding(entity_ids)
        present = entity_ids.ne(0).to(states.dtype).unsqueeze(-1)
        return self.norm((states * present).sum(dim=1) / present.sum(dim=1).clamp_min(1.0))


__all__ = ["GlobalEntityEmbedding", "GlobalEntityVocabulary", "canonical_entity"]
