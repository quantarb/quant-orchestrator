from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd
import torch


@dataclass(frozen=True)
class FlairClassificationRegressionResult:
    model: Any
    corpus: Any
    classifier: Any
    regressor: Any
    train_result: Any
    classification_label_type: str
    regression_label_type: str
    classification_task_id: str
    regression_task_id: str


def patch_multitask_evaluate_for_regression() -> None:
    """Patch Flair 0.15.x mixed classification/regression MTL evaluation.

    Flair's MultitaskModel.evaluate expects every task report to include
    classification averages. TextRegressor reports regression scores instead,
    so ModelTrainer evaluation raises KeyError for mixed MTL jobs.
    """
    import flair
    from flair.models import MultitaskModel, TextRegressor
    from flair.training_utils import Result

    if getattr(MultitaskModel.evaluate, "_regression_safe_patch", False):
        return

    def regression_safe_evaluate(
        self,
        data_points,
        gold_label_type,
        out_path=None,
        main_evaluation_metric=("micro avg", "f1-score"),
        evaluate_all=True,
        **evalargs,
    ):
        if not evaluate_all:
            if gold_label_type not in self.tasks:
                raise ValueError(
                    "evaluating a single task on a multitask model requires "
                    "'gold_label_type' to be a valid task."
                )
            data = [
                dp
                for dp in data_points
                if any(label.value == gold_label_type for label in dp.get_labels("multitask_id"))
            ]
            task = self.tasks[gold_label_type]
            task_metric = main_evaluation_metric
            if isinstance(task, TextRegressor) and task_metric == ("micro avg", "f1-score"):
                task_metric = ("correlation", "spearman")
            return task.evaluate(
                data,
                gold_label_type=task.label_type,
                out_path=out_path,
                main_evaluation_metric=task_metric,
                **evalargs,
            )

        batch_split = self.split_batch_to_task_ids(data_points, all_tasks=True)
        loss = torch.tensor(0.0, device=flair.device)
        main_score = 0.0
        all_detailed_results = ""
        all_classification_report = {}
        scores = {}

        for task_id, split in batch_split.items():
            task = self.tasks[task_id]
            task_metric = main_evaluation_metric
            if isinstance(task, TextRegressor) and task_metric == ("micro avg", "f1-score"):
                task_metric = ("correlation", "spearman")

            result = task.evaluate(
                data_points=[data_points[i] for i in split],
                gold_label_type=task.label_type,
                out_path=f"{out_path}_{task_id}.txt" if out_path is not None else None,
                main_evaluation_metric=task_metric,
                **evalargs,
            )

            loss += result.loss
            main_score += result.main_score
            all_detailed_results += (
                50 * "-"
                + "\n\n"
                + task_id
                + " - Label type: "
                + task.label_type
                + "\n\n"
                + result.detailed_results
            )
            all_classification_report[task_id] = result.classification_report

            report = result.classification_report or {}
            for avg_type in ("micro avg", "macro avg"):
                if avg_type not in report:
                    continue
                for metric_type in ("f1-score", "precision", "recall"):
                    if metric_type in report[avg_type]:
                        scores[(task_id, avg_type, metric_type)] = report[avg_type][metric_type]

            for metric_name, metric_value in result.scores.items():
                scores[(task_id, metric_name)] = metric_value

        scores["loss"] = loss.item() / len(batch_split)
        return Result(
            main_score=main_score / len(batch_split),
            detailed_results=all_detailed_results,
            scores=scores,
            classification_report=all_classification_report,
        )

    regression_safe_evaluate._regression_safe_patch = True
    MultitaskModel.evaluate = regression_safe_evaluate


def _make_multitask_sentence(
    text: str,
    classification_label: str,
    regression_label: float,
    *,
    classification_label_type: str,
    regression_label_type: str,
    classification_task_id: str,
    regression_task_id: str,
):
    from flair.data import Sentence

    sentence = Sentence(text)
    sentence.add_label(classification_label_type, classification_label)
    sentence.add_label(regression_label_type, f"{float(regression_label):.8f}")
    sentence.add_label("multitask_id", classification_task_id)
    sentence.add_label("multitask_id", regression_task_id)
    return sentence


def _frame_to_multitask_sentences(
    frame: pd.DataFrame,
    *,
    text_column: str,
    classification_column: str,
    regression_column: str,
    class_label_fn: Callable[[Any], str],
    classification_label_type: str,
    regression_label_type: str,
    classification_task_id: str,
    regression_task_id: str,
):
    return [
        _make_multitask_sentence(
            str(row[text_column]),
            class_label_fn(row[classification_column]),
            float(row[regression_column]),
            classification_label_type=classification_label_type,
            regression_label_type=regression_label_type,
            classification_task_id=classification_task_id,
            regression_task_id=regression_task_id,
        )
        for _, row in frame.iterrows()
    ]


def build_classification_regression_corpus(
    splits: Mapping[str, pd.DataFrame],
    *,
    text_column: str,
    classification_column: str,
    regression_column: str,
    class_label_fn: Callable[[Any], str],
    classification_label_type: str = "direction",
    regression_label_type: str = "return_percentile",
    classification_task_id: str = "direction",
    regression_task_id: str = "return_percentile",
):
    from flair.data import Corpus

    return Corpus(
        train=_frame_to_multitask_sentences(
            splits["train"],
            text_column=text_column,
            classification_column=classification_column,
            regression_column=regression_column,
            class_label_fn=class_label_fn,
            classification_label_type=classification_label_type,
            regression_label_type=regression_label_type,
            classification_task_id=classification_task_id,
            regression_task_id=regression_task_id,
        ),
        dev=_frame_to_multitask_sentences(
            splits["dev"],
            text_column=text_column,
            classification_column=classification_column,
            regression_column=regression_column,
            class_label_fn=class_label_fn,
            classification_label_type=classification_label_type,
            regression_label_type=regression_label_type,
            classification_task_id=classification_task_id,
            regression_task_id=regression_task_id,
        ),
        test=_frame_to_multitask_sentences(
            splits["test"],
            text_column=text_column,
            classification_column=classification_column,
            regression_column=regression_column,
            class_label_fn=class_label_fn,
            classification_label_type=classification_label_type,
            regression_label_type=regression_label_type,
            classification_task_id=classification_task_id,
            regression_task_id=regression_task_id,
        ),
        sample_missing_splits=False,
    )


def build_classification_regression_multitask_model(
    corpus,
    *,
    transformer_model: str,
    classification_label_type: str = "direction",
    regression_label_type: str = "return_percentile",
    classification_task_id: str = "direction",
    regression_task_id: str = "return_percentile",
    classification_loss_factor: float = 1.0,
    regression_loss_factor: float = 0.5,
    fine_tune_transformer: bool = False,
    layers: str = "-1",
    layer_mean: bool = False,
    allow_long_sentences: bool = False,
    use_all_tasks: bool = True,
):
    from flair.embeddings import TransformerDocumentEmbeddings
    from flair.models import MultitaskModel, TextClassifier, TextRegressor

    patch_multitask_evaluate_for_regression()
    embeddings = TransformerDocumentEmbeddings(
        transformer_model,
        fine_tune=fine_tune_transformer,
        layers=layers,
        layer_mean=layer_mean,
        allow_long_sentences=allow_long_sentences,
    )
    classifier = TextClassifier(
        embeddings,
        label_type=classification_label_type,
        label_dictionary=corpus.make_label_dictionary(classification_label_type, add_unk=False),
    )
    regressor = TextRegressor(embeddings, label_name=regression_label_type)
    model = MultitaskModel(
        [classifier, regressor],
        task_ids=[classification_task_id, regression_task_id],
        loss_factors=[classification_loss_factor, regression_loss_factor],
        use_all_tasks=use_all_tasks,
    )
    return model, classifier, regressor


def train_classification_regression_multitask(
    splits: Mapping[str, pd.DataFrame],
    *,
    base_path: str | Path,
    transformer_model: str,
    text_column: str,
    classification_column: str,
    regression_column: str,
    class_label_fn: Callable[[Any], str],
    classification_label_type: str = "direction",
    regression_label_type: str = "return_percentile",
    classification_task_id: str = "direction",
    regression_task_id: str = "return_percentile",
    classification_loss_factor: float = 1.0,
    regression_loss_factor: float = 0.5,
    fine_tune_transformer: bool = False,
    max_epochs: int = 1,
    learning_rate: float = 1e-4,
    mini_batch_size: int = 16,
    mini_batch_chunk_size: int | None = None,
    eval_batch_size: int = 32,
    save_final_model: bool = False,
    create_file_logs: bool = False,
    create_loss_file: bool = False,
    embeddings_storage_mode: str = "none",
    **trainer_kwargs: Any,
) -> FlairClassificationRegressionResult:
    import flair
    from flair.trainers import ModelTrainer

    patch_multitask_evaluate_for_regression()
    corpus = build_classification_regression_corpus(
        splits,
        text_column=text_column,
        classification_column=classification_column,
        regression_column=regression_column,
        class_label_fn=class_label_fn,
        classification_label_type=classification_label_type,
        regression_label_type=regression_label_type,
        classification_task_id=classification_task_id,
        regression_task_id=regression_task_id,
    )
    model, classifier, regressor = build_classification_regression_multitask_model(
        corpus,
        transformer_model=transformer_model,
        classification_label_type=classification_label_type,
        regression_label_type=regression_label_type,
        classification_task_id=classification_task_id,
        regression_task_id=regression_task_id,
        classification_loss_factor=classification_loss_factor,
        regression_loss_factor=regression_loss_factor,
        fine_tune_transformer=fine_tune_transformer,
    )
    model = model.to(flair.device)
    trainer = ModelTrainer(model, corpus)
    train_result = trainer.fine_tune(
        base_path=Path(base_path),
        learning_rate=learning_rate,
        mini_batch_size=mini_batch_size,
        mini_batch_chunk_size=mini_batch_chunk_size,
        eval_batch_size=eval_batch_size,
        max_epochs=max_epochs,
        embeddings_storage_mode=embeddings_storage_mode,
        save_final_model=save_final_model,
        create_file_logs=create_file_logs,
        create_loss_file=create_loss_file,
        **trainer_kwargs,
    )
    return FlairClassificationRegressionResult(
        model=model,
        corpus=corpus,
        classifier=classifier,
        regressor=regressor,
        train_result=train_result,
        classification_label_type=classification_label_type,
        regression_label_type=regression_label_type,
        classification_task_id=classification_task_id,
        regression_task_id=regression_task_id,
    )


def predict_classification_regression(
    result: FlairClassificationRegressionResult,
    sentences,
    *,
    classification_prediction_label: str = "pred_direction",
    regression_prediction_label: str = "pred_return_percentile",
    mini_batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    result.classifier.predict(
        sentences,
        label_name=classification_prediction_label,
        mini_batch_size=mini_batch_size,
    )
    result.regressor.predict(
        sentences,
        label_name=regression_prediction_label,
        mini_batch_size=mini_batch_size,
    )
    class_predictions = np.array(
        [sentence.get_labels(classification_prediction_label)[0].value for sentence in sentences]
    )
    regression_predictions = np.array(
        [float(sentence.get_labels(regression_prediction_label)[0].value) for sentence in sentences]
    )
    return class_predictions, regression_predictions
