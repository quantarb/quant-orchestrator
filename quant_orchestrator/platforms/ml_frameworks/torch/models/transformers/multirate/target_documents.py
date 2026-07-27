"""Target-family subtokens and documents.

This intentionally mirrors the feature-family document path while remaining
separate until target-family semantics stabilize.  Multiple insider events on
the same date are valid and are therefore retained as distinct subtokens.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd


@dataclass(frozen=True)
class TargetSubtokenDocumentCorpus:
    """Sparse target subtokens grouped into ``(symbol, target_family)`` documents."""

    documents: pd.DataFrame
    subtokens: pd.DataFrame
    document_subtokens: pd.DataFrame
    prototype_targets: pd.DataFrame
    target_columns: Mapping[str, tuple[str, ...]]


def build_endpoint_target_subtoken_documents(
    frame: pd.DataFrame,
    target_columns: Mapping[str, list[str] | tuple[str, ...]],
    *,
    symbol_column: str = "symbol",
    timestamp_column: str,
    disclosure_timestamp_column: str | None = None,
    target_family_column: str = "target_family",
) -> TargetSubtokenDocumentCorpus:
    """Build sparse subtokens from one endpoint-backed target family.

    Each source event becomes a ``happened`` subtoken and, when a disclosure
    timestamp is available, a separate ``disclosed`` subtoken.  Same-day
    events are not deduplicated because separate owners, securities, or Form 4
    transactions carry distinct predictive information.  Documents are
    grouped only by symbol and target family, matching the feature-family
    document layout.  ``availability_timestamp`` is the causal timestamp: a
    happened event is a future target, while its disclosed event is observable
    only from the disclosure timestamp onward.
    """

    required = {symbol_column, timestamp_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    families = {str(name): tuple(columns) for name, columns in target_columns.items()}
    if len(families) != 1:
        raise ValueError("target_columns must define exactly one target family")
    target_family = next(iter(families))
    columns = [column for family in families.values() for column in family]
    missing_columns = sorted(set(columns) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"missing target columns: {missing_columns}")

    selected = [symbol_column, timestamp_column, *columns]
    if disclosure_timestamp_column and disclosure_timestamp_column in frame.columns:
        selected.append(disclosure_timestamp_column)
    events = frame[selected].copy()
    events = events.rename(columns={
        symbol_column: "symbol",
        timestamp_column: "event_timestamp",
    })
    if disclosure_timestamp_column and disclosure_timestamp_column in events.columns:
        events = events.rename(columns={disclosure_timestamp_column: "disclosure_timestamp"})
    events["symbol"] = events["symbol"].astype(str).str.upper()
    events["event_timestamp"] = pd.to_datetime(events["event_timestamp"], utc=True)
    events = events.loc[events["event_timestamp"].notna()].copy()
    events["source_event_id"] = events.index.astype("int64")
    if "disclosure_timestamp" in events:
        events["disclosure_timestamp"] = pd.to_datetime(events["disclosure_timestamp"], utc=True)

    happened = events.copy()
    happened["event_role"] = "happened"
    happened["availability_timestamp"] = (
        happened["disclosure_timestamp"] if "disclosure_timestamp" in happened
        else happened["event_timestamp"]
    )
    happened["availability_timestamp"] = happened["availability_timestamp"].fillna(happened["event_timestamp"])

    parts = [happened]
    if "disclosure_timestamp" in events:
        disclosed = events.loc[events["disclosure_timestamp"].notna()].copy()
        disclosed["event_role"] = "disclosed"
        disclosed["event_timestamp"] = disclosed["disclosure_timestamp"]
        disclosed["availability_timestamp"] = disclosed["disclosure_timestamp"]
        parts.append(disclosed)
    events = pd.concat(parts, ignore_index=True)
    events["target_family"] = target_family
    events = events.sort_values(["symbol", "target_family", "event_timestamp", "event_role"]).reset_index(drop=True)
    events.insert(0, "subtoken_id", events.index.astype("int64"))
    events["event_id"] = events["source_event_id"]

    documents = events[["symbol", "target_family"]].drop_duplicates()
    documents = documents.sort_values(["symbol", "target_family"]).reset_index(drop=True)
    documents.insert(0, "document_id", documents.index.astype("int64"))
    events = events.merge(documents, on=["symbol", "target_family"], how="left", validate="many_to_one")

    document_subtokens = events[[
        "document_id", "symbol", "target_family", "subtoken_id", "event_id",
        "event_role", "event_timestamp", "availability_timestamp",
    ]]
    prototype_targets = documents.merge(
        document_subtokens.groupby("document_id", as_index=False).size().rename(columns={"size": "subtoken_count"}),
        on="document_id",
        how="left",
    )
    prototype_targets["target_type"] = "mean_subtoken_embedding"
    prototype_targets["target_family"] = target_family

    return TargetSubtokenDocumentCorpus(
        documents=documents,
        subtokens=events,
        document_subtokens=document_subtokens,
        prototype_targets=prototype_targets,
        target_columns=families,
    )


def build_event_target_family_subtoken_documents(
    frame: pd.DataFrame,
    target_family: str,
    *,
    event_family: str | None = None,
    symbol_column: str = "symbol",
    timestamp_column: str = "event_date",
    disclosure_timestamp_column: str | None = "reported_date",
) -> TargetSubtokenDocumentCorpus:
    """Convert normalized event rows into one sparse target-family document."""

    work = frame.copy()
    if event_family is not None and "event_family" in work.columns:
        work = work.loc[work["event_family"].astype(str).eq(event_family)].copy()
    excluded = {symbol_column, timestamp_column, disclosure_timestamp_column, "target_family"}
    columns = tuple(column for column in work.columns if column not in excluded)
    if not columns:
        raise ValueError("normalized event frame has no event detail columns")
    return build_endpoint_target_subtoken_documents(
        work,
        {str(target_family): columns},
        symbol_column=symbol_column,
        timestamp_column=timestamp_column,
        disclosure_timestamp_column=disclosure_timestamp_column,
    )


def build_ownership_insider_trading_subtoken_documents(
    frame: pd.DataFrame,
    target_columns: Mapping[str, list[str] | tuple[str, ...]],
    *,
    symbol_column: str = "symbol",
    timestamp_column: str = "event_date",
    disclosure_timestamp_column: str | None = "reported_date",
    target_family_column: str = "target_family",
) -> TargetSubtokenDocumentCorpus:
    """Build happened/disclosed subtokens for insider endpoint events."""

    return build_endpoint_target_subtoken_documents(
        frame,
        target_columns,
        symbol_column=symbol_column,
        timestamp_column=timestamp_column,
        disclosure_timestamp_column=disclosure_timestamp_column,
        target_family_column=target_family_column,
    )


def build_earnings_report_subtoken_documents(
    frame: pd.DataFrame,
    target_columns: Mapping[str, list[str] | tuple[str, ...]],
    *,
    symbol_column: str = "symbol",
    timestamp_column: str = "report_date",
    target_family_column: str = "target_family",
) -> TargetSubtokenDocumentCorpus:
    """Build sparse subtokens from the FMP earnings-calendar endpoint."""

    return build_endpoint_target_subtoken_documents(
        frame,
        target_columns,
        symbol_column=symbol_column,
        timestamp_column=timestamp_column,
        target_family_column=target_family_column,
    )


def build_analyst_rating_subtoken_documents(
    frame: pd.DataFrame,
    target_columns: Mapping[str, list[str] | tuple[str, ...]],
    *,
    symbol: str | None = None,
    symbol_column: str = "symbol",
    timestamp_column: str = "published_date",
    target_family_column: str = "target_family",
) -> TargetSubtokenDocumentCorpus:
    """Build sparse subtokens from FMP's analyst price-target endpoint.

    The endpoint row is retained as-is, including analyst identity, headline,
    price target, and posted price.  No upgrade/downgrade collapse is applied
    in the document builder.
    """

    work = frame.copy()
    if symbol_column not in work.columns and symbol is not None:
        work.insert(0, symbol_column, str(symbol).upper())
    return build_endpoint_target_subtoken_documents(
        work,
        target_columns,
        symbol_column=symbol_column,
        timestamp_column=timestamp_column,
        target_family_column=target_family_column,
    )


__all__ = [
    "TargetSubtokenDocumentCorpus",
    "build_endpoint_target_subtoken_documents",
    "build_event_target_family_subtoken_documents",
    "build_earnings_report_subtoken_documents",
    "build_analyst_rating_subtoken_documents",
    "build_ownership_insider_trading_subtoken_documents",
]
