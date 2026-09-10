from datetime import datetime
import polars as pl
import pytest
from quant_orchestrator.research_tools.multirate_corpus import bounded_dates, missing_statement_years


def test_statement_fields_preserve_financial_values_and_exclude_metadata():
    from quant_orchestrator.research_tools.multirate_corpus import statement_fields
    frame = pl.DataFrame({'revenue': [100.], 'bottom_line_net_income': [12.],
        'filing_date': [12345.], 'accepted_date': [54321.], 'fiscal_period': ['FY'],
        'unused': pl.Series([None], dtype=pl.Float64)})
    assert statement_fields(frame) == ['revenue', 'bottom_line_net_income']


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


def test_evaluation_baseline_uses_model_cutoff_across_later_years(tmp_path):
    from quant_orchestrator.research_tools.multirate_audit import evaluate_predictions, TASKS
    pl.DataFrame({'symbol': ['X'], 'issuer': ['X'], 'asset_class': ['equity']}).write_csv(tmp_path / 'taxonomy.csv')
    dates = [datetime(year, 6, 1) for year in (2023, 2024, 2025)]
    pl.DataFrame({'symbol': ['X'], 'date': [dates[-1]]}).with_columns(pl.col('date').cast(pl.Datetime('ns'))).write_parquet(tmp_path / 'daily.parquet')
    events = pl.DataFrame({'symbol': ['X'] * 3, 'date': dates, 'event_date': dates,
        'target_family': ['equity.strategy.hits_graph'] * 3,
        'signal_value': [1., 9., 3.], **{f'text_{i}': [1., 9., 3.] for i in range(7)}})
    events.with_columns(pl.col('date', 'event_date').cast(pl.Datetime('ns'))).write_parquet(tmp_path / 'sparse_events.parquet')
    scores = tmp_path / 'scores.csv'
    pl.DataFrame({'symbol': ['X'], 'date': [dates[-1]], **{task: [3.] for task in TASKS}}).write_csv(scores)
    report = evaluate_predictions(tmp_path, scores, start='2025-01-01', end='2025-12-31', training_cutoff='2024-01-01')
    metric = next(row for row in report['event_metrics'] if row['task'] == 'hits_long_return_hub')
    assert metric['mse'] == 0
    assert metric['train_mean_baseline_mse'] == 4  # (2023 mean 1 - 2025 target 3)^2.
    assert report['training_cutoff'] == '2024-01-01'
    with pytest.raises(ValueError, match='cutoff'):
        evaluate_predictions(tmp_path, scores, start='2025-01-01', end='2025-12-31', training_cutoff='2026-01-01')
