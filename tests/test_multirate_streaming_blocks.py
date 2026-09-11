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


def test_indexed_sequences_match_query_reference_across_years_and_cache_eviction():
    start = datetime(2018, 1, 1)
    frame = pl.DataFrame({'symbol': ['A'] * 1800,
        'date': [start + timedelta(days=i) for i in range(1800)],
        'x': [None if i % 7 == 0 else float(i) for i in range(1800)]})
    index = StreamingContext(frame.lazy(), ['x'])
    for begin, end in [(500,700),(600,850),(3,9),(1600,1799),(700,750)]:
        args=('A', start+timedelta(days=begin), start+timedelta(days=end),32)
        actual=index.sequence_window(*args); expected=index._uncached_sequence_window(*args)
        for a,b in zip(actual,expected):
            if a is not None:torch.testing.assert_close(a,b,equal_nan=True,rtol=0,atol=0)
    assert index.cache_hits > 0
    index.sequence_cache_limit=128
    actual=index.sequence_window('A',start,start+timedelta(days=900),32)
    expected=index._uncached_sequence_window('A',start,start+timedelta(days=900),32)
    for a,b in zip(actual,expected):
        if a is not None:torch.testing.assert_close(a,b,equal_nan=True,rtol=0,atol=0)


def test_sparse_sequence_reserves_each_family_history_and_matches_reference():
    from quant_orchestrator.research_tools.streaming_context import StreamingFamilyContext
    rows=[]
    for year in range(2010,2025):
        for month in range(1,13):
            rows.append({'symbol':'A','date':datetime(year,month,1),'target_family':'fast','x':float(month),'y':None})
        if year % 2 == 0:
            rows.append({'symbol':'A','date':datetime(year,1,1),'target_family':'slow','x':None,'y':float(year)})
    index=StreamingFamilyContext(pl.DataFrame(rows).lazy(),['x','y'],families=['fast','slow','absent'],history_per_family=3)
    for start,end in [(datetime(2020,5,1),datetime(2021,2,1)),(datetime(2020,6,1),datetime(2021,3,1))]:
        actual=index.sequence_window('A',start,end,9)
        expected=index._uncached_family_sequence_window('A',start,end,9)
        for a,b in zip(actual,expected):
            if a is not None:torch.testing.assert_close(a,b,equal_nan=True,rtol=0,atol=0)
    assert index.cache_hits > 0
    assert index.sequence_cache_bytes <= index.sequence_cache_limit


def test_disk_index_reopens_without_source_reads_and_preserves_causal_slices(tmp_path):
    frame=pl.DataFrame({'symbol':['A']*5,'date':[datetime(2020,1,d) for d in range(1,6)],'x':[1.,None,3.,4.,5.]})
    first=StreamingContext(frame.lazy(),['x'],index_directory=tmp_path)
    args=('A',datetime(2020,1,2),datetime(2020,1,4),2)
    expected=first._uncached_sequence_window(*args)
    actual=first.sequence_window(*args)
    reopened=StreamingContext(frame.head(0).lazy(),['x'],index_directory=tmp_path)
    cached=reopened.sequence_window(*args)
    for values in [actual,cached]:
        for a,b in zip(values,expected):
            if a is not None:torch.testing.assert_close(a,b,equal_nan=True,rtol=0,atol=0)
    values,padding,dates,_=reopened.window('A',datetime(2020,1,3),2)
    torch.testing.assert_close(values[:,0],torch.tensor([float('nan'),3.]),equal_nan=True)
    assert dates.max()==pl.Series([datetime(2020,1,3)]).dt.epoch('ns')[0]


def test_oversized_disk_index_falls_back_to_bounded_windows(tmp_path):
    frame=pl.DataFrame({'symbol':['A']*10,'date':[datetime(2020,1,d) for d in range(1,11)],'x':list(range(10))})
    context=StreamingContext(frame.lazy(),['x'],index_directory=tmp_path)
    context.index_build_limit=40
    result=context.sequence_window('A',datetime(2020,1,2),datetime(2020,1,5),2)
    assert result[0][:,0].tolist()==[0.,1.,2.,3.,4.]
    assert 'A' in context._oversized_issuers
    assert not list(tmp_path.glob('*.pt'))
