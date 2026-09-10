"""Event-owned overlapping sequences; feature histories remain in lazy storage."""
from datetime import datetime
import polars as pl
import torch


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
                       tasks: tuple[str, ...], start: datetime, end: datetime):
    """One bounded query and exact-date join; context-only positions have no label."""
    targets=torch.zeros((length,len(tasks)),dtype=torch.float32)
    valid=torch.zeros_like(targets,dtype=torch.bool)
    if not len(dates):
        return targets,valid
    index=pl.DataFrame({'date':pl.Series(dates.cpu().tolist(),dtype=pl.Int64).cast(pl.Datetime('ns'))})
    labels=store.scan.filter((pl.col('symbol')==symbol) & pl.col('date').is_between(start,end)).select(
        pl.col('date').cast(pl.Datetime('ns')), *tasks)
    joined=index.lazy().join(labels,on='date',how='left',maintain_order='left').select(*tasks).collect(engine='streaming')
    block=joined.select(pl.all().cast(pl.Float32).fill_nan(None).fill_null(float('nan'))).to_torch()
    observed=torch.isfinite(block)
    targets[-len(dates):]=block.nan_to_num()
    valid[-len(dates):]=observed
    return targets,valid
