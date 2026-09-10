from datetime import datetime
import polars as pl
import pytest
from quant_orchestrator.research_tools.multirate_corpus import bounded_dates, missing_statement_years


def test_long_history_allows_later_statement_floor_but_rejects_internal_gaps():
    frame = pl.DataFrame({"date": [datetime(year, 12, 31) for year in [1985, 1986, 1988]]})
    assert missing_statement_years(
        frame, column="date", start="1900-01-01", end="1988-12-31", minimum=1
    ) == [1987]


def test_long_history_checks_quarter_counts_after_partial_initial_year():
    dates = [datetime(1985, 12, 31)]
    dates += [datetime(1986, month, 28) for month in [3, 6, 9, 12]]
    dates += [datetime(1987, month, 28) for month in [3, 6, 12]]
    frame = pl.DataFrame({"date": dates})
    assert missing_statement_years(
        frame, column="date", start="1900-01-01", end="1987-12-31", minimum=4
    ) == [1987]


def test_corpus_rejects_out_of_range_warehouse_reads():
    frame = pl.DataFrame({"period_ending": [datetime(2023, 3, 31), datetime(2026, 3, 31)]})
    with pytest.raises(ValueError, match="violate requested range"):
        bounded_dates(frame, "2021-01-01", "2025-12-31", column="period_ending")
    assert (
        bounded_dates(frame.head(1), "2021-01-01", "2025-12-31", column="period_ending").height == 1
    )


def test_corpus_rejects_undated_rows():
    frame = pl.DataFrame({"date": pl.Series([None], dtype=pl.Datetime)})
    with pytest.raises(ValueError):
        bounded_dates(frame, "2021-01-01", "2025-12-31")


def test_input_fingerprint_rejects_changed_corpus(tmp_path):
    import hashlib
    from quant_orchestrator.research_tools.multirate_audit import verify_corpus_files

    path = tmp_path / "daily.parquet"
    path.write_bytes(b"original")
    manifest = {"input_sha256": {"daily.parquet": hashlib.sha256(b"original").hexdigest()}}
    assert len(verify_corpus_files(tmp_path, manifest)) == 64
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed after assembly"):
        verify_corpus_files(tmp_path, manifest)
