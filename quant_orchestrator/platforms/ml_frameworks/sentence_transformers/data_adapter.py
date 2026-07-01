from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class SentenceTransformersTextClassificationColumns:
    text: str = "text"
    label: str = "label"


@dataclass(frozen=True)
class SentenceTransformersTextClassificationData:
    train_texts: list[str]
    test_texts: list[str]
    train_labels: list[str]
    test_labels: list[str]


def build_text_classification_data(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    *,
    columns: SentenceTransformersTextClassificationColumns | None = None,
) -> SentenceTransformersTextClassificationData:
    cols = columns or SentenceTransformersTextClassificationColumns()
    return SentenceTransformersTextClassificationData(
        train_texts=train_frame[cols.text].astype(str).tolist(),
        test_texts=test_frame[cols.text].astype(str).tolist(),
        train_labels=train_frame[cols.label].astype(str).tolist(),
        test_labels=test_frame[cols.label].astype(str).tolist(),
    )


def encode_texts(
    model: Any,
    texts: list[str],
    *,
    batch_size: int = 128,
    normalize_embeddings: bool = True,
):
    return model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=normalize_embeddings,
    )
