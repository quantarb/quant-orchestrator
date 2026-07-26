"""Sparse 1T subtokens grouped into symbol/feature-family documents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd
import torch


@dataclass(frozen=True)
class SubtokenDocumentCorpus:
    """Documents whose identity is exactly ``(symbol, feature_family)``."""

    documents: pd.DataFrame
    subtokens: pd.DataFrame
    document_subtokens: pd.DataFrame
    prototype_targets: pd.DataFrame
    feature_columns: Mapping[str, tuple[str, ...]]


def build_1t_subtoken_documents(
    frame: pd.DataFrame,
    feature_columns: Mapping[str, list[str] | tuple[str, ...]],
    *,
    symbol_column: str = "symbol",
    timestamp_column: str = "timestamp",
    family_column: str = "feature_family",
) -> SubtokenDocumentCorpus:
    """Build sparse ``(symbol, feature_family)`` documents from 1T events.

    Every source row becomes one subtoken and is stored exactly once.  A
    document contains the ordered ID links for one symbol/family sequence;
    there is no date or minute-level sample axis in the document table.  A
    document prototype is computed by mean-pooling its valid subtoken
    embeddings with :func:`mean_document_embeddings`.
    """
    required = {symbol_column, timestamp_column, family_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    families = {str(name): tuple(columns) for name, columns in feature_columns.items()}
    if not families:
        raise ValueError("feature_columns must define at least one family")
    columns = [column for family in families.values() for column in family]
    missing_features = sorted(set(columns) - set(frame.columns))
    if missing_features:
        raise ValueError(f"missing feature columns: {missing_features}")

    events = frame[[symbol_column, timestamp_column, family_column, *columns]].copy()
    events = events.rename(columns={
        symbol_column: "symbol",
        timestamp_column: "timestamp",
        family_column: "feature_family",
    })
    events["timestamp"] = pd.to_datetime(events["timestamp"], utc=True)
    events["feature_family"] = events["feature_family"].astype(str)
    if not events["feature_family"].isin(families).all():
        unknown = sorted(set(events.loc[~events["feature_family"].isin(families), "feature_family"]))
        raise ValueError(f"feature families missing from feature_columns: {unknown}")
    if events.duplicated(["symbol", "timestamp", "feature_family"]).any():
        raise ValueError("input must contain at most one observation per symbol/timestamp/family")

    events = events.sort_values(["symbol", "feature_family", "timestamp"]).reset_index(drop=True)
    events.insert(0, "subtoken_id", events.index.astype("int64"))
    documents = events[["symbol", "feature_family"]].drop_duplicates()
    documents = documents.sort_values(["symbol", "feature_family"]).reset_index(drop=True)
    documents.insert(0, "document_id", documents.index.astype("int64"))
    events = events.merge(documents, on=["symbol", "feature_family"], how="left", validate="many_to_one")

    document_subtokens = events[[
        "document_id", "symbol", "feature_family", "subtoken_id", "timestamp",
    ]].rename(columns={"timestamp": "subtoken_timestamp"})
    prototype_targets = documents.merge(
        document_subtokens.groupby("document_id", as_index=False).size().rename(columns={"size": "subtoken_count"}),
        on="document_id",
        how="left",
    )
    prototype_targets["target_type"] = "mean_subtoken_embedding"

    return SubtokenDocumentCorpus(
        documents=documents,
        subtokens=events,
        document_subtokens=document_subtokens,
        prototype_targets=prototype_targets,
        feature_columns=families,
    )


def mean_document_embeddings(
    subtoken_embeddings: torch.Tensor,
    document_ids: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
    num_documents: int | None = None,
) -> torch.Tensor:
    """Mean-pool subtoken embeddings into one prototype per document."""
    if subtoken_embeddings.ndim != 2:
        raise ValueError("subtoken_embeddings must have shape [subtokens, d_model]")
    if document_ids.ndim != 1 or document_ids.shape[0] != subtoken_embeddings.shape[0]:
        raise ValueError("document_ids must have shape [subtokens]")
    if (document_ids < 0).any():
        raise ValueError("document_ids must be non-negative")
    if valid_mask is None:
        valid_mask = torch.ones(document_ids.shape, dtype=torch.bool, device=document_ids.device)
    if valid_mask.shape != document_ids.shape:
        raise ValueError("valid_mask must have shape [subtokens]")
    count = int(document_ids.max().item()) + 1 if document_ids.numel() else 0
    count = max(count, int(num_documents or 0))
    sums = torch.zeros(count, subtoken_embeddings.shape[1], device=subtoken_embeddings.device, dtype=subtoken_embeddings.dtype)
    counts = torch.zeros(count, 1, device=subtoken_embeddings.device, dtype=subtoken_embeddings.dtype)
    weights = valid_mask.to(subtoken_embeddings.dtype).unsqueeze(-1)
    sums.index_add_(0, document_ids, subtoken_embeddings * weights)
    counts.index_add_(0, document_ids, weights)
    return sums / counts.clamp_min(1.0)


__all__ = ["SubtokenDocumentCorpus", "build_1t_subtoken_documents", "mean_document_embeddings"]
