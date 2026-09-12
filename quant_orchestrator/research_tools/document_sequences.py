"""Calendar-stable, causal multi-date documents over lazy native-rate histories."""
from datetime import datetime
import polars as pl
import torch


DOCUMENT_CONTRACT = 'calendar_quarter_prefix_v1'
TRAINING_DOCUMENT_CONTRACT = 'issuer_observation_quarters_v1'


def issuer_observation_dates(scan):
    """Select native issuer observations without removing shared context history.

    Macro, peer aggregates and calendar features cannot create a company
    document on their own. Both scalar and expanded family columns are handled.
    Sparse disclosures qualify through their observed signal/text channels.
    """
    context = {'economic_indicators', 'treasury_rates', 'time_calendar',
               'sector_pe', 'sector_performance', 'industry_pe', 'industry_performance'}
    fields = [name for name, dtype in scan.collect_schema().items()
              if dtype.is_numeric() and (
                  (name.startswith('value__') and name[7:].split('.')[0].split('__')[0] not in context)
                  or name == 'signal_value' or name.startswith('text_'))]
    observed = pl.any_horizontal([pl.col(name).is_finite().fill_null(False) for name in fields]) if fields else pl.lit(False)
    return scan.filter(observed).select('symbol', 'date')


def document_anchors(scans, *, start=None, end=None, cutoff=None, period='3mo'):
    """Collect date metadata; each observation belongs to one calendar period.

    Boundaries are fixed independently of future observations and reporting
    dates. A partial live quarter therefore has the same start as its replay.
    Native rate updates within the quarter are retained by sequence_window.
    """
    dates = pl.concat([scan.select('symbol', pl.col('date').cast(pl.Datetime('ns')))
                       for scan in scans]).unique()
    if cutoff:
        dates = dates.filter(pl.col('date') < datetime.fromisoformat(cutoff))
    if end:
        dates = dates.filter(pl.col('date') <= datetime.fromisoformat(end))
    if period not in ('3mo', '1y'):
        raise ValueError('Document period must be 3mo or 1y')
    dates = dates.with_columns(pl.col('date').dt.truncate(period).alias('document_start'))
    anchors = dates.group_by('symbol', 'document_start').agg(pl.col('date').max()).sort('symbol', 'date')
    if start:
        anchors = anchors.filter(pl.col('date') >= datetime.fromisoformat(start))
    return anchors.collect(engine='streaming')


def document_window(index, symbol, start, end, history):
    """Keep fixed start history plus every update, with stable token positions.

    A permanent leading placeholder gives empty causal queries a legal key.
    Real observations start at position one. All batching pads to the right;
    adding future observations never moves an existing token's position.
    """
    values, padding, dates, labels = index.sequence_window(symbol, start, end, history)
    real = values[~padding]
    length = max(2, len(real) + 1)
    if length > 512:
        raise ValueError(f'Document exceeds positional capacity: {symbol} {start} {end}: {length}; no observations were dropped')
    result = torch.full((length, values.shape[-1]), float('nan'), dtype=values.dtype)
    mask = torch.ones(length, dtype=torch.bool)
    timestamps = torch.full((length,), torch.iinfo(torch.long).max, dtype=torch.long)
    timestamps[0] = torch.iinfo(torch.long).min
    if len(real):
        result[1:len(real)+1] = real
        mask[1:len(real)+1] = False
        timestamps[1:len(real)+1] = dates
    return result, mask, dates, timestamps, labels


def prediction_positions(item, *, start=None):
    """Map owned daily dates to tensor slots, never to the document end date."""
    for position, date in enumerate(item['daily_dates']):
        if start is not None and date < start:
            continue
        if item.get('sequence_mode') in ('documents', 'annual_memory'):
            if date < item['document_start'] or date > item['date']:
                continue
            yield position + 1, date
        elif date == item['date']:
            yield len(item['daily']) - len(item['daily_dates']) + position, date


def validate_document_predictions(predictions, expected_dates):
    """Require exactly one finite score row for every requested symbol/date."""
    keys=['symbol','date']
    actual=predictions.with_columns(pl.col('date').cast(pl.Date))
    expected=expected_dates.select('symbol',pl.col('date').cast(pl.Date)).unique()
    counts=actual.select(pl.len()).collect(engine='streaming').item()
    duplicate_keys=actual.group_by(keys).len().filter(pl.col('len')!=1).select(pl.len()).collect(engine='streaming').item()
    missing=expected.join(actual.select(keys),on=keys,how='anti').select(pl.len()).collect(engine='streaming').item()
    extra=actual.select(keys).join(expected,on=keys,how='anti').select(pl.len()).collect(engine='streaming').item()
    numeric=[name for name,dtype in actual.collect_schema().items() if dtype.is_numeric()]
    nonfinite=actual.filter(pl.any_horizontal(pl.col(name).is_null() | ~pl.col(name).is_finite() for name in numeric)).select(pl.len()).collect(engine='streaming').item() if numeric else 0
    report=dict(prediction_rows=counts,duplicate_keys=duplicate_keys,missing_dates=missing,unexpected_dates=extra,nonfinite_rows=nonfinite)
    if duplicate_keys or missing or extra or nonfinite:
        raise ValueError(f'Document score coverage failed: {report}')
    return report
