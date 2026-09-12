"""Event-owned overlapping sequences; feature histories remain in lazy storage."""
from datetime import datetime
import polars as pl
import torch
import numpy as np


def sequence_anchors(daily: pl.LazyFrame, events: pl.DataFrame, *, cutoff: str,
                     stride: int, window: int) -> tuple[pl.DataFrame, dict]:
    if not 0 < stride < window:
        raise ValueError('Sequence stride must be positive and smaller than the daily window')
    dates = daily.filter(pl.col('date') < datetime.fromisoformat(cutoff)).select('symbol',pl.col('date').cast(pl.Datetime('ns'))).unique()
    # Only the compact date index is collected, never the feature matrix.
    dates = dates.sort('symbol','date').with_columns(
        ((pl.col('date').cum_count().over('symbol')-1)//stride).alias('_chunk'))
    events = events.select('symbol',pl.col('date').cast(pl.Datetime('ns'))).unique().lazy()
    matched = dates.join(events, on=['symbol','date'], how='inner')
    owners = matched.group_by('symbol','_chunk').agg(
        pl.col('date').min().alias('supervision_start'), pl.col('date').max().alias('date'),
        pl.len().alias('supervised_dates')).collect(engine='streaming')
    unmatched = events.join(dates.select('symbol','date'), on=['symbol','date'], how='anti').collect(engine='streaming')
    report = dict(stride=stride, window=window, context_overlap=window-stride,
                  sequence_documents=owners.height,
                  matched_event_dates=int(owners['supervised_dates'].sum() or 0),
                  unmatched_event_dates=unmatched.height,
                  unmatched_by_symbol=unmatched.group_by('symbol').len().sort('symbol').to_dicts())
    return owners.drop('_chunk').sort('symbol','date'), report


def window_supervision(store, symbol: str, dates: torch.Tensor, *, length: int,
                       tasks: tuple[str, ...], start: datetime, end: datetime, positions=None):
    """Exact-date lookup over bounded cached labels; context-only positions have no label."""
    targets=torch.zeros((length,len(tasks)),dtype=torch.float32)
    valid=torch.zeros_like(targets,dtype=torch.bool)
    if not len(dates):
        return targets,valid
    indexed = store.window_index.get(symbol, tasks)
    if indexed is None:
        index=pl.DataFrame({'date':pl.Series(dates.cpu().tolist(),dtype=pl.Int64).cast(pl.Datetime('ns'))})
        labels=store.scan.filter((pl.col('symbol')==symbol) & pl.col('date').is_between(start,end)).select(
            pl.col('date').cast(pl.Datetime('ns')), *tasks)
        joined=index.lazy().join(labels,on='date',how='left',maintain_order='left').select(*tasks).collect(engine='streaming')
        block=joined.select(pl.all().cast(pl.Float32).fill_nan(None).fill_null(float('nan'))).to_torch()
    else:
        label_dates, values = indexed
        query = dates.cpu().long()
        block = torch.full((len(query), len(tasks)), float('nan'), dtype=torch.float32)
        if len(label_dates):
            indices = torch.searchsorted(label_dates, query)
            safe = indices.clamp(max=len(label_dates) - 1)
            matched = (indices < len(label_dates)) & (label_dates[safe] == query)
            matched &= query >= int(np.datetime64(start, 'ns').astype('int64'))
            matched &= query <= int(np.datetime64(end, 'ns').astype('int64'))
            block[matched] = values[safe[matched]]
    observed=torch.isfinite(block)
    positions = slice(-len(dates), None) if positions is None else positions
    targets[positions]=block.nan_to_num()
    valid[positions]=observed
    return targets,valid
