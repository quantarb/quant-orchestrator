from __future__ import annotations

import importlib.util

import pandas as pd
import pytest

from quant_orchestrator.platforms.ml_frameworks.flair.shared import (
    build_classification_regression_corpus,
    patch_multitask_evaluate_for_regression,
)


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("flair") is None,
    reason="flair is an optional ML dependency",
)


def test_patch_multitask_evaluate_for_regression_is_idempotent() -> None:
    from flair.models import MultitaskModel

    patch_multitask_evaluate_for_regression()
    patched = MultitaskModel.evaluate
    patch_multitask_evaluate_for_regression()
    assert MultitaskModel.evaluate is patched
    assert getattr(MultitaskModel.evaluate, "_regression_safe_patch", False)


def test_build_classification_regression_corpus_adds_task_labels() -> None:
    frame = pd.DataFrame(
        {
            "text": ["symbol=AAPL close=100", "symbol=AAPL close=101"],
            "target": [1, 0],
            "rank_y": [0.75, 0.25],
        }
    )
    corpus = build_classification_regression_corpus(
        {"train": frame, "dev": frame.iloc[:1], "test": frame.iloc[1:]},
        text_column="text",
        classification_column="target",
        regression_column="rank_y",
        class_label_fn=lambda value: "long" if int(value) == 1 else "not_long",
    )

    sentence = corpus.train[0]
    assert sentence.get_labels("direction")[0].value == "long"
    assert sentence.get_labels("return_percentile")[0].value == "0.75000000"
    assert [label.value for label in sentence.get_labels("multitask_id")] == [
        "direction",
        "return_percentile",
    ]
