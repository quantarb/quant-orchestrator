"""FlairNLP ML framework helpers."""

from quant_orchestrator.platforms.ml_frameworks.flair.data_adapter import (
    FlairMultitaskColumns,
    FlairTextClassificationColumns,
    LazyMultitaskDataset,
    LazyTextClassificationDataset,
    build_label_dictionary,
    build_multitask_corpus,
    build_text_classification_corpus as build_flair_text_classification_corpus,
    frame_to_text_classification_sentences,
    make_text_classification_sentence,
)
from quant_orchestrator.platforms.ml_frameworks.flair.runtime import (
    FlairRuntimeInfo,
    configure_flair_device,
    configure_flair_runtime,
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
    "FlairMultitaskColumns",
    "FlairRuntimeInfo",
    "FlairTextClassificationColumns",
    "LazyMultitaskDataset",
    "LazyTextClassificationDataset",
    "build_flair_text_classification_corpus",
    "build_label_dictionary",
    "build_classification_regression_corpus",
    "build_classification_regression_multitask_model",
    "build_multitask_corpus",
    "configure_flair_device",
    "configure_flair_runtime",
    "frame_to_text_classification_sentences",
    "make_text_classification_sentence",
    "patch_multitask_evaluate_for_regression",
    "predict_classification_regression",
    "train_classification_regression_multitask",
]
