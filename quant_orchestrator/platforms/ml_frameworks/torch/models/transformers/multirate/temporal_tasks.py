"""The complete temporal multi-rate MTL task contract.

This module is the single source of truth for the temporal task heads used by
the temporal MTL experiment.  The classification labels are attached to
temporal documents; this module does not create documents or define attention
between documents.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from quant_orchestrator.platforms.ml_frameworks.torch.models.transformers.multirate.model import (
    MultiRatePredictionTaskSpec,
    MultiRateTaskSpec,
)
from quant_orchestrator.platforms.ml_frameworks.torch.models.transformers.multirate.multitask import Task
from quant_orchestrator.platforms.ml_frameworks.torch.models.transformers.multirate.multitask import Corpus


DOCUMENT_TASK_NAMES = (
    "family", "issuer", "symbol", "industry", "sector", "subsector", "year",
)
SUBTOKEN_PREDICTION_TASK_NAMES = (
    "next_annual_subtoken",
    "next_quarterly_subtoken",
    "next_daily_subtoken",
    "next_sparse_subtoken",
    "masked_annual_subtoken",
    "masked_quarterly_subtoken",
    "masked_daily_subtoken",
    "masked_sparse_subtoken",
)
TOKEN_PREDICTION_TASK_NAMES = (
    "next_annual_token",
    "next_quarterly_token",
    "next_daily_token",
    "next_sparse_token",
    "masked_annual_token",
    "masked_quarterly_token",
    "masked_daily_token",
    "masked_sparse_token",
)
ORACLE_SUPERVISED_TASK_NAMES = (
    "oracle_is_buy", "oracle_is_sell", "oracle_is_short", "oracle_is_cover",
)
HITS_SUPERVISED_TASK_NAMES = (
    "hits_long_return_hub", "hits_long_return_authority",
    "hits_short_return_hub", "hits_short_return_authority",
    "hits_long_speed_hub", "hits_long_speed_authority",
    "hits_short_speed_hub", "hits_short_speed_authority",
)
FUND_ACTIVITY_SUPERVISED_TASK_NAMES = (
    "fund_activity_etf_buy",
    "fund_activity_mutual_fund_buy",
    "fund_activity_institutional_buy",
    "fund_activity_hedge_fund_buy",
    "fund_activity_add",
    "fund_activity_reduce",
    "fund_activity_exit",
)
SUPERVISED_TARGET_TASK_NAMES = (
    ORACLE_SUPERVISED_TASK_NAMES
    + HITS_SUPERVISED_TASK_NAMES
    + FUND_ACTIVITY_SUPERVISED_TASK_NAMES
)
PREDICTION_TASK_NAMES = SUBTOKEN_PREDICTION_TASK_NAMES + TOKEN_PREDICTION_TASK_NAMES
TEMPORAL_MTL_TASK_NAMES = DOCUMENT_TASK_NAMES + SUPERVISED_TARGET_TASK_NAMES + PREDICTION_TASK_NAMES


@dataclass(frozen=True)
class TemporalMTLTaskBundle:
    """All task specs and names for one temporal MTL model."""

    corpus: Corpus
    document_tasks: tuple[MultiRateTaskSpec, ...]
    supervised_tasks: tuple[MultiRateTaskSpec, ...]
    prediction_tasks: tuple[MultiRatePredictionTaskSpec, ...]

    @property
    def task_names(self) -> tuple[str, ...]:
        return tuple(task.task_name for task in self.document_tasks + self.supervised_tasks + self.prediction_tasks)

    @property
    def tasks(self) -> tuple[Task, ...]:
        return tuple(Task(task.task_name, task) for task in self.document_tasks + self.supervised_tasks + self.prediction_tasks)


def add_subtoken_temporal_tasks(
    rows: Sequence[Any],
    family_names: Sequence[str],
    label_names: Mapping[str, Sequence[str]],
    *,
    batch_size: int = 1,
    corpus_name: str = "temporal",
) -> TemporalMTLTaskBundle:
    """Create shared corpus with co-trained token and subtoken task heads.

    ``family_names`` contains the union of feature and target families.  The
    remaining document labels must contain ``issuer``, ``symbol``,
    ``industry``, ``sector``, ``subsector``, and ``year``.  Those labels
    classify temporal documents; they are not cross-sectional document sources.
    """
    corpus = Corpus(rows, name=corpus_name, batch_size=batch_size)
    required_labels = set(DOCUMENT_TASK_NAMES[1:])
    missing = sorted(required_labels - set(label_names))
    if missing:
        raise ValueError(f"missing temporal document labels: {missing}")

    document_tasks = tuple(
        MultiRateTaskSpec(
            task_name,
            level="document",
            output_dim=len(family_names) if task_name == "family" else len(label_names[task_name]),
            source="family" if task_name == "family" else "fused",
        )
        for task_name in DOCUMENT_TASK_NAMES
    )
    supervised_tasks = tuple(
        MultiRateTaskSpec(task_name, level="token", output_dim=1, source="daily")
        for task_name in SUPERVISED_TARGET_TASK_NAMES
    )
    prediction_tasks = tuple(
        MultiRatePredictionTaskSpec(
            task_name,
            objective="next_token" if task_name.startswith("next_") else "masked_token",
            level="token" if task_name.endswith("_token") else "subtoken",
            output_dim=1,
            source=task_name.split("_")[1],
        )
        for task_name in PREDICTION_TASK_NAMES
    )
    bundle = TemporalMTLTaskBundle(corpus, document_tasks, supervised_tasks, prediction_tasks)
    if bundle.task_names != TEMPORAL_MTL_TASK_NAMES:
        raise RuntimeError("temporal MTL task bundle does not match the token+subtoken contract")
    return bundle


__all__ = [
    "DOCUMENT_TASK_NAMES",
    "SUBTOKEN_PREDICTION_TASK_NAMES",
    "TOKEN_PREDICTION_TASK_NAMES",
    "ORACLE_SUPERVISED_TASK_NAMES",
    "HITS_SUPERVISED_TASK_NAMES",
    "FUND_ACTIVITY_SUPERVISED_TASK_NAMES",
    "SUPERVISED_TARGET_TASK_NAMES",
    "PREDICTION_TASK_NAMES",
    "TEMPORAL_MTL_TASK_NAMES",
    "TemporalMTLTaskBundle",
    "add_subtoken_temporal_tasks",
]
