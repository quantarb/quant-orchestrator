from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class TransformersTextClassificationColumns:
    text: str = "text"
    label: str = "label"


@dataclass(frozen=True)
class TransformersTextClassificationData:
    train: Dataset
    test: Dataset
    label_to_id: dict[str, int]
    id_to_label: dict[int, str]


class TextClassificationDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        tokenizer,
        label_to_id: Mapping[str, int],
        columns: TransformersTextClassificationColumns | None = None,
        max_length: int = 512,
    ) -> None:
        self.columns = columns or TransformersTextClassificationColumns()
        self.texts = frame[self.columns.text].astype(str).tolist()
        self.labels = [label_to_id[str(label)] for label in frame[self.columns.label].tolist()]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            self.texts[index],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[index], dtype=torch.long),
        }


def build_label_maps(
    frame: pd.DataFrame,
    *,
    label_column: str = "label",
) -> tuple[dict[str, int], dict[int, str]]:
    labels = sorted(frame[label_column].astype(str).unique())
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    return label_to_id, id_to_label


def build_text_classification_datasets(
    splits: Mapping[str, pd.DataFrame],
    *,
    tokenizer,
    columns: TransformersTextClassificationColumns | None = None,
    max_length: int = 512,
) -> TransformersTextClassificationData:
    cols = columns or TransformersTextClassificationColumns()
    label_to_id, id_to_label = build_label_maps(splits["train"], label_column=cols.label)
    return TransformersTextClassificationData(
        train=TextClassificationDataset(
            splits["train"],
            tokenizer=tokenizer,
            label_to_id=label_to_id,
            columns=cols,
            max_length=max_length,
        ),
        test=TextClassificationDataset(
            splits["test"],
            tokenizer=tokenizer,
            label_to_id=label_to_id,
            columns=cols,
            max_length=max_length,
        ),
        label_to_id=label_to_id,
        id_to_label=id_to_label,
    )
