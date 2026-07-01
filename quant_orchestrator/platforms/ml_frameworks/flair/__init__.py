"""FlairNLP ML framework helpers."""

from quant_orchestrator.platforms.ml_frameworks.flair.data_adapter import (
    FlairTextClassificationColumns,
    LazyTextClassificationDataset,
    build_label_dictionary,
    build_text_classification_corpus as build_flair_text_classification_corpus,
    frame_to_text_classification_sentences,
    make_text_classification_sentence,
)
from quant_orchestrator.platforms.ml_frameworks.flair.shared import (
    FlairClassificationRegressionResult,
    build_classification_regression_corpus,
    build_classification_regression_multitask_model,
    patch_multitask_evaluate_for_regression,
    predict_classification_regression,
    train_classification_regression_multitask,
)

__all__ = [
    "FlairClassificationRegressionResult",
    "FlairTextClassificationColumns",
    "LazyTextClassificationDataset",
    "build_flair_text_classification_corpus",
    "build_label_dictionary",
    "build_classification_regression_corpus",
    "build_classification_regression_multitask_model",
    "frame_to_text_classification_sentences",
    "make_text_classification_sentence",
    "patch_multitask_evaluate_for_regression",
    "predict_classification_regression",
    "train_classification_regression_multitask",
]
