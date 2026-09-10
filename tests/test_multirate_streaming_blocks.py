from datetime import datetime, timedelta

import polars as pl
import torch

from quant_orchestrator.research_tools.streaming_context import StreamingContext


def test_cached_blocks_match_asof_windows_and_never_return_future_rows():
    start = datetime(2020, 1, 1)
    frame = pl.DataFrame({'symbol': ['A'] * 1500,
        'date': [start + timedelta(days=i) for i in range(1500)],
        'x': list(range(1500))})
    index = StreamingContext(frame.lazy(), ['x'])
    for offset in [1000, 900, 1100, 901, 0, 1499]:
        anchor = start + timedelta(days=offset)
        values, padding, dates, _ = index.window('A', anchor, 32)
        expected = frame.filter(pl.col('date') <= anchor).tail(32)
        torch.testing.assert_close(values[~padding, 0], expected['x'].cast(pl.Float32).to_torch())
        torch.testing.assert_close(dates, expected['date'].dt.epoch('ns').to_torch())
    assert index.cache_hits > 0
    assert all(frame.height <= 1056 for frame, _, _ in index._blocks.values())


def test_dense_year_and_old_version_fall_back_without_losing_history():
    start = datetime(2020, 1, 1)
    frame = pl.DataFrame({'symbol': ['A'] * 5000,
        'date': [start + timedelta(hours=i) for i in range(5000)],
        'x': list(range(5000))})
    index = StreamingContext(frame.lazy(), ['x'])
    anchor = start + timedelta(hours=20)
    assert index.version('A', anchor) == frame['date'].dt.epoch('ns')[20]
    assert len(index._versions['A']) == 4096
    values, padding, dates, _ = index.window('A', anchor, 8)
    assert values[:, 0].tolist() == list(range(13, 21))
    assert not padding.any()
    assert dates[-1] == index.version('A', anchor)
    assert index.version('A', start - timedelta(days=1)) == -1


def test_cache_eviction_and_same_date_family_union():
    rows = [{'symbol': str(i), 'date': datetime(2020, 1, 1), 'x': 1., 'y': None}
            for i in range(40)]
    rows += [{'symbol': '0', 'date': datetime(2020, 1, 1), 'x': None, 'y': 2.}]
    index = StreamingContext(pl.DataFrame(rows).lazy(), ['x', 'y'])
    for i in range(40):
        index.version(str(i), datetime(2021, 1, 1))
        index.window(str(i), datetime(2021, 1, 1), 2)
    assert len(index._versions) == len(index._blocks) == 32
    values, padding, _, _ = index.window('0', datetime(2021, 1, 1), 2)
    assert values[-1].tolist() == [1., 2.] and padding.tolist() == [True, False]


def test_sparse_families_keep_history_despite_frequent_other_events():
    from quant_orchestrator.research_tools.streaming_context import StreamingFamilyContext
    rows = [dict(symbol='X', date=datetime(2020, 1, day), target_family='fast', a=None, b=float(day)) for day in range(1, 31)]
    rows += [dict(symbol='X', date=datetime(2020, 1, day), target_family='slow', a=float(day), b=None) for day in (1, 15, 30)]
    context = StreamingFamilyContext(pl.DataFrame(rows).lazy(), ['a', 'b'], families=['fast', 'slow'], history_per_family=2)
    values, padding, dates, _ = context.window('X', datetime(2020, 1, 30), 4)
    observed = values[~padding]
    assert observed.shape == (3, 2)  # Date 30 contains both families.
    assert observed[:, 0].isfinite().sum() == 2
    assert observed[-1].tolist() == [30., 30.]
    earlier, early_padding, _, _ = context.window('X', datetime(2020, 1, 20), 4)
    assert earlier[~early_padding, 0].nan_to_num().max() == 15
    assert earlier[~early_padding, 1].nan_to_num().max() == 20


def test_anchor_ordering_preserves_rows_and_reuses_bounded_issuer_metadata():
    from collections import OrderedDict
    from quant_orchestrator.research_tools.streaming_context import context_ordered_anchors
    # Interleaving >32 issuers previously evicted every cached version on each pass.
    rows=[{'symbol':f'S{i:03d}', 'date':datetime(2020,1,day)}
          for day in (1,2,3) for i in range(114)]
    rows += [{'symbol':'OTHER_SHARE','date':datetime(2020,1,2)}]
    original=pl.DataFrame(rows)
    ordered=context_ordered_anchors(original,{'OTHER_SHARE':'S000'})
    assert ordered.sort('symbol','date').equals(original.sort('symbol','date'))
    cache=OrderedDict();misses=0
    for symbol in ordered['symbol']:
        issuer='S000' if symbol=='OTHER_SHARE' else symbol
        if issuer not in cache:misses+=1
        cache[issuer]=True;cache.move_to_end(issuer)
        if len(cache)>32:cache.popitem(last=False)
    assert misses==114
    assert len(cache)==32
