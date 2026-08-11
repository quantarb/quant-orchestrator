"""Warehouse-backed, non-materializing option document stream.

This module deliberately returns one issuer's frozen DTE groups at a time. It
does not write a corpus or retain other issuers' raw option chains.
"""

from __future__ import annotations

from collections.abc import Iterator

import pandas as pd

from scripts.build_annual_option_documents import _load_raw_first_day, _select


def iter_frozen_dte_documents(
    symbols: list[str] | set[str],
    *,
    start_year: int,
    end_year: int,
    dte: int | None = None,
) -> Iterator[tuple[str, pd.DataFrame]]:
    """Yield ``(issuer, documents)`` one underlying at a time."""
    for symbol in sorted({str(value).upper() for value in symbols}):
        raw = _load_raw_first_day({symbol}, start_year=start_year, end_year=end_year)
        if raw.empty:
            continue
        selected = _select(raw, max_contracts=0, group_by_dte=True)
        selected["underlying_symbol"] = selected["symbol"].astype(str).str.upper()
        if dte is not None:
            selected = selected.loc[pd.to_numeric(selected["dte"], errors="coerce").eq(dte)].copy()
        if selected.empty:
            continue
        selected["document_symbol"] = [
            f"OPT_{symbol}_{int(year)}_{str(option_type)[:1].upper()}_DTE_{int(group_dte)}"
            for year, option_type, group_dte in zip(selected["year"], selected["option_type"], selected["dte"])
        ]
        yield symbol, selected.reset_index(drop=True)


__all__ = ["iter_frozen_dte_documents"]
