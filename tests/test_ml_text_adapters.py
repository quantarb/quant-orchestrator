from __future__ import annotations

import importlib.util

import pandas as pd
import pytest

from quant_orchestrator.platforms.ml_frameworks.sentence_transformers.data_adapter import (
    SentenceTransformersTextClassificationColumns,
    build_text_classification_data,
)
from quant_orchestrator.platforms.ml_frameworks.transformers.data_adapter import (
    TransformersTextClassificationColumns,
    build_label_maps,
    build_text_classification_datasets,
)


class _Tokenizer:
    def __call__(
        self,
        text,
        *,
        truncation: bool,
        padding: str,
        max_length: int,
        return_tensors: str,
    ):
        import torch

        assert truncation
        assert padding == "max_length"
        assert return_tensors == "pt"
        return {
            "input_ids": torch.zeros((1, max_length), dtype=torch.long),
            "attention_mask": torch.ones((1, max_length), dtype=torch.long),
        }


def test_sentence_transformers_text_classification_data_uses_columns() -> None:
    train = pd.DataFrame({"body": ["a", "b"], "side": ["buy", "sell"]})
    test = pd.DataFrame({"body": ["c"], "side": ["buy"]})

    data = build_text_classification_data(
        train,
        test,
        columns=SentenceTransformersTextClassificationColumns(text="body", label="side"),
    )

    assert data.train_texts == ["a", "b"]
    assert data.test_texts == ["c"]
    assert data.train_labels == ["buy", "sell"]
    assert data.test_labels == ["buy"]


def test_transformers_text_classification_dataset_encodes_labels() -> None:
    train = pd.DataFrame({"body": ["a", "b"], "side": ["buy", "sell"]})
    test = pd.DataFrame({"body": ["c"], "side": ["buy"]})

    data = build_text_classification_datasets(
        {"train": train, "test": test},
        tokenizer=_Tokenizer(),
        columns=TransformersTextClassificationColumns(text="body", label="side"),
        max_length=8,
    )
    item = data.train[0]

    assert data.label_to_id == {"buy": 0, "sell": 1}
    assert data.id_to_label == {0: "buy", 1: "sell"}
    assert tuple(item["input_ids"].shape) == (8,)
    assert int(item["labels"]) == 0


def test_build_label_maps_sorts_string_labels() -> None:
    frame = pd.DataFrame({"label": ["sell", "buy", "buy"]})
    label_to_id, id_to_label = build_label_maps(frame)

    assert label_to_id == {"buy": 0, "sell": 1}
    assert id_to_label == {0: "buy", 1: "sell"}


def test_ml_runtime_adapters_configure_cpu_device() -> None:
    from quant_orchestrator.platforms.ml_frameworks.sentence_transformers.runtime import (
        configure_sentence_transformers_runtime,
    )
    from quant_orchestrator.platforms.ml_frameworks.torch.runtime import configure_torch_runtime
    from quant_orchestrator.platforms.ml_frameworks.transformers.runtime import (
        configure_transformers_runtime,
    )

    assert configure_torch_runtime(device="cpu").torch_device == "cpu"
    assert configure_transformers_runtime(device="cpu").torch_device == "cpu"
    assert configure_sentence_transformers_runtime(device="cpu").torch_device == "cpu"


@pytest.mark.skipif(importlib.util.find_spec("flair") is None, reason="flair is an optional ML dependency")
def test_flair_lazy_text_classification_dataset_creates_sentence() -> None:
    from quant_orchestrator.platforms.ml_frameworks.flair.data_adapter import (
        FlairMultitaskColumns,
        FlairTextClassificationColumns,
        LazyMultitaskDataset,
        LazyTextClassificationDataset,
        build_label_dictionary,
        build_multitask_corpus,
        frame_to_text_classification_sentences,
    )
    from quant_orchestrator.platforms.ml_frameworks.flair.runtime import configure_flair_runtime

    frame = pd.DataFrame({"body": ["symbol=AAPL"], "side": ["buy"]})
    columns = FlairTextClassificationColumns(text="body", label="side", label_type="trade_side")
    dataset = LazyTextClassificationDataset(frame, columns=columns)
    sentence = dataset[0]

    assert len(dataset) == 1
    assert not dataset.is_in_memory()
    assert sentence.to_original_text() == "symbol=AAPL"
    assert sentence.get_labels("trade_side")[0].value == "buy"
    assert build_label_dictionary(frame, label_column="side").get_items() == ["buy"]
    assert frame_to_text_classification_sentences(frame, columns=columns)[0].get_labels("trade_side")[0].value == "buy"

    multitask_frame = pd.DataFrame(
        {
            "body": ["symbol=AAPL"],
            "side": ["buy"],
            "label_type": ["task_event"],
            "task_id": ["task_event"],
        }
    )
    multitask_columns = FlairMultitaskColumns(text="body", label="side")
    multitask_dataset = LazyMultitaskDataset(multitask_frame, columns=multitask_columns)
    multitask_sentence = multitask_dataset[0]

    assert multitask_sentence.get_labels("task_event")[0].value == "buy"
    assert multitask_sentence.get_labels("multitask_id")[0].value == "task_event"
    assert build_multitask_corpus(
        {"train": multitask_frame, "dev": multitask_frame, "test": multitask_frame},
        columns=multitask_columns,
    ).train[0].get_labels("task_event")[0].value == "buy"
    assert configure_flair_runtime(device="cpu").torch_device == "cpu"
