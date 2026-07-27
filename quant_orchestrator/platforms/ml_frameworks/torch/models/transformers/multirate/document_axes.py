"""Explicit temporal and cross-sectional document contracts."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TemporalDocument:
    """One issuer's as-of history with causal temporal attention."""

    document_id: int
    symbol: str
    as_of_date: pd.Timestamp
    tokens: pd.DataFrame
    attention_mode: str = "temporal"

    def __post_init__(self) -> None:
        if self.attention_mode != "temporal":
            raise ValueError("TemporalDocument must use temporal attention")
        if self.tokens.empty:
            raise ValueError("TemporalDocument must contain tokens")
        dates = pd.to_datetime(self.tokens["date"], errors="coerce")
        if dates.isna().any() or (dates > self.as_of_date).any():
            raise ValueError("temporal tokens must be valid and available by as_of_date")


@dataclass(frozen=True)
class CrossSectionalDocument:
    """All peer tokens observed on one date."""

    document_id: int
    date: pd.Timestamp
    tokens: pd.DataFrame
    attention_mode: str = "cross_sectional"

    def __post_init__(self) -> None:
        if self.attention_mode != "cross_sectional":
            raise ValueError("CrossSectionalDocument must use cross-sectional attention")
        if self.tokens.empty:
            raise ValueError("CrossSectionalDocument must contain tokens")
        dates = pd.to_datetime(self.tokens["date"], errors="coerce")
        if dates.isna().any() or not dates.eq(self.date).all():
            raise ValueError("cross-sectional tokens must share the document date")


def build_temporal_documents(frame: pd.DataFrame, *, symbol_column: str = "symbol", date_column: str = "date") -> tuple[TemporalDocument, ...]:
    work = frame.rename(columns={symbol_column: "symbol", date_column: "date"}).copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    documents: list[TemporalDocument] = []
    for document_id, (symbol, as_of_date) in enumerate(work[["symbol", "date"]].drop_duplicates().sort_values(["symbol", "date"]).itertuples(index=False, name=None)):
        tokens = work.loc[work["symbol"].eq(symbol) & work["date"].le(as_of_date)].sort_values("date").copy()
        documents.append(TemporalDocument(document_id, str(symbol), pd.Timestamp(as_of_date), tokens))
    return tuple(documents)


def build_cross_sectional_documents(frame: pd.DataFrame, *, date_column: str = "date") -> tuple[CrossSectionalDocument, ...]:
    work = frame.rename(columns={date_column: "date"}).copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    documents: list[CrossSectionalDocument] = []
    for document_id, date in enumerate(sorted(work["date"].dropna().unique())):
        timestamp = pd.Timestamp(date)
        documents.append(CrossSectionalDocument(document_id, timestamp, work.loc[work["date"].eq(timestamp)].copy()))
    return tuple(documents)


__all__ = ["TemporalDocument", "CrossSectionalDocument", "build_temporal_documents", "build_cross_sectional_documents"]
