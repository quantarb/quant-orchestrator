"""FlairNLP ML framework helpers."""

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
    "build_classification_regression_corpus",
    "build_classification_regression_multitask_model",
    "patch_multitask_evaluate_for_regression",
    "predict_classification_regression",
    "train_classification_regression_multitask",
]
