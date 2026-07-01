from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd

try:  # Flair is an optional dependency for the base package.
    from flair.data import FlairDataset as _FlairDataset
except Exception:  # pragma: no cover - exercised only without optional dependency.
    _FlairDataset = object


@dataclass(frozen=True)
class FlairTextClassificationColumns:
    text: str = "text"
    label: str = "label"
    label_type: str = "class"


class LazyTextClassificationDataset(_FlairDataset):
    """Create Flair Sentence objects on demand for dataframe-backed text classification."""

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        columns: FlairTextClassificationColumns | None = None,
    ) -> None:
        self.columns = columns or FlairTextClassificationColumns()
        self.texts = frame[self.columns.text].astype(str).tolist()
        self.labels = frame[self.columns.label].astype(str).tolist()

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int):
        from flair.data import Sentence

        sentence = Sentence(self.texts[index])
        sentence.add_label(self.columns.label_type, self.labels[index])
        return sentence

    def is_in_memory(self) -> bool:
        return False


def make_text_classification_sentence(
    text: Any,
    label: Any,
    *,
    label_type: str = "class",
):
    from flair.data import Sentence

    sentence = Sentence(str(text))
    sentence.add_label(label_type, str(label))
    return sentence


def frame_to_text_classification_sentences(
    frame: pd.DataFrame,
    *,
    columns: FlairTextClassificationColumns | None = None,
) -> list[Any]:
    cols = columns or FlairTextClassificationColumns()
    return [
        make_text_classification_sentence(row[cols.text], row[cols.label], label_type=cols.label_type)
        for _, row in frame.iterrows()
    ]


def build_text_classification_corpus(
    splits: Mapping[str, pd.DataFrame],
    *,
    columns: FlairTextClassificationColumns | None = None,
    lazy: bool = True,
):
    from flair.data import Corpus

    cols = columns or FlairTextClassificationColumns()
    if lazy:
        train = LazyTextClassificationDataset(splits["train"], columns=cols)
        dev = LazyTextClassificationDataset(splits["dev"], columns=cols)
        test = LazyTextClassificationDataset(splits["test"], columns=cols)
    else:
        train = frame_to_text_classification_sentences(splits["train"], columns=cols)
        dev = frame_to_text_classification_sentences(splits["dev"], columns=cols)
        test = frame_to_text_classification_sentences(splits["test"], columns=cols)
    return Corpus(train=train, dev=dev, test=test, sample_missing_splits=False)


def build_label_dictionary(frame: pd.DataFrame, *, label_column: str = "label"):
    from flair.data import Dictionary

    dictionary = Dictionary(add_unk=False)
    for label in sorted(frame[label_column].astype(str).unique()):
        dictionary.add_item(label)
    return dictionary
