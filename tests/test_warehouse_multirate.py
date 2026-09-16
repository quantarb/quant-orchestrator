from datetime import datetime, timedelta

import polars as pl
import pytest
import torch

from quant_orchestrator.research_tools.warehouse_multirate import annual_tensor
from quant_orchestrator.research_tools.warehouse_multirate_training import document_batches
from quant_orchestrator.research_tools.frozen_option_adjustments import first_session_baskets, basket_quotes, split_adjusted_members


def test_universe_and_option_scope_are_read_fresh_without_any_corpus(monkeypatch,tmp_path):
    from types import SimpleNamespace
    from quant_orchestrator.research_tools import warehouse_multirate as module
    requested=[];option_reads=[]
    def profiles(**kwargs):
        requested.append(kwargs)
        return [SimpleNamespace(symbol=s) for s in ('GOOG','GOOGL','MSFT')]
    def prices(symbol,**kwargs):
        return pl.DataFrame({'date':[datetime(2023,1,3),datetime(2023,1,4)],'close':[1.,2.]})
    def options(symbol,**kwargs):
        option_reads.append(symbol)
        return pl.DataFrame({'snapshot_date':[datetime(2023,1,3),datetime(2023,1,4)]})
    def forbidden(*args,**kwargs):raise AssertionError('Attempted to read a prebuilt corpus')
    monkeypatch.setattr(module,'read_option_chain_arctic',options)
    monkeypatch.setattr(module.pl,'read_parquet',forbidden)
    monkeypatch.setattr(module.pl,'scan_parquet',forbidden)
    warehouse=SimpleNamespace(catalog=SimpleNamespace(query_symbol_profiles=profiles),read_prices=prices,
        backend=SimpleNamespace(list_symbols=lambda library:['GOOG','GOOGL']))
    for _ in range(2):
        stream=module.WarehouseAnnualStream(min_market_cap=1e12,start='1900-01-01',end='2026-09-09',cutoff='2024-01-01',output=tmp_path,warehouse=warehouse)
        assert set(stream.prices)=={'GOOG','GOOGL','MSFT'}
        assert stream.expected_option_symbols=={'GOOG','GOOGL'}
    assert len(requested)==2 and all(r['min_market_cap']==1e12 for r in requested)
    assert option_reads==['GOOG','GOOGL','GOOG','GOOGL']


def test_annual_tensor_preserves_missingness_dates_and_fixed_transform():
    start=datetime(2023,1,1)
    frame=pl.DataFrame({'date':[start,start+timedelta(days=3)],'x':[99.,None],'y':[-9.,0.]})
    values,padding,dates=annual_tensor(frame,['x','y'],start)
    assert values.shape==(3,2)
    assert padding.tolist()==[True,False,False]
    assert torch.isnan(values[0]).all() and torch.isnan(values[2,0])
    assert values[1,0].item()==pytest.approx(torch.log(torch.tensor(100.)).item()/10)
    assert values[1,1]<0 and values[2,1]==0
    assert dates[2]-dates[1]==3*86400*10**9


def test_batches_preserve_recurrent_chronology_without_preparing_all_documents():
    loaded=[]
    def documents():
        for symbol,year in [('A',2022),('B',2022),('A',2023),('B',2023),('C',2023)]:
            loaded.append((symbol,year))
            yield dict(symbol=symbol,year=year)
    batches=document_batches(documents(),16)
    assert [r['symbol'] for r in next(batches)]==['A','B']
    assert len(loaded)==3  # One lookahead, never materialize the remaining stream.
    assert [[r['year'] for r in b] for b in batches]==[[2023,2023,2023]]


def chain():
    first=datetime(2024,1,2);rows=[]
    for right in ('call','put'):
        for dte in (3,10,30,90,365):
            expiry=first+timedelta(days=dte)
            for strike in (100.,200.):
                rows.append(dict(snapshot_date=first,underlying_symbol='X',option_type=right,
                    expiration=expiry,strike=strike,contract_symbol=f'X{expiry:%y%m%d}{right[0].upper()}{int(strike*1000):08d}',bid=10.,ask=12.,volume=1.))
    return pl.DataFrame(rows)


def test_frozen_members_ignore_future_chain_and_never_renormalize_missing_constituents():
    q=chain();first=q['snapshot_date'][0]
    future=q.with_columns(pl.col('snapshot_date')+timedelta(days=1),pl.lit(999.).alias('strike'),pl.lit('NEW').alias('contract_symbol'))
    members=first_session_baskets(pl.concat([q,future]),first)
    assert members.height==20 and members['document_symbol'].n_unique()==10
    assert set(members['weight'])=={.5}
    assert 'NEW' not in members['contract_symbol']
    partial=q.filter(pl.col('strike')==100.).with_columns(pl.col('snapshot_date')+timedelta(days=1))
    path=basket_quotes(pl.concat([q,partial]),members)
    assert path.height==10 and set(path['date'])=={first}
    assert set(path['close'])=={11.}


def test_split_adjustment_conserves_basket_value_and_original_membership():
    q=chain();members=first_session_baskets(q,q['snapshot_date'][0])
    split=q.with_columns((pl.col('strike')/10).alias('strike'),(pl.col('bid')/10).alias('bid'),(pl.col('ask')/10).alias('ask'))
    split=split.with_columns(pl.concat_str([pl.col('contract_symbol').str.replace(r'\d{8}$',''),
        (pl.col('strike')*1000).cast(pl.Int64).cast(pl.String).str.pad_start(8,'0')]).alias('contract_symbol'))
    adjusted=split_adjusted_members(members,split,10.)
    assert adjusted.height==members.height
    assert set(adjusted['weight'])=={5.}
    assert set(basket_quotes(split,adjusted)['close'])=={11.}
    assert set(adjusted['document_symbol'])==set(members['document_symbol'])


def test_annual_capacity_rejects_silent_truncation():
    start=datetime(2023,1,1)
    frame=pl.DataFrame({'date':[start+timedelta(minutes=i) for i in range(512)],'x':[1.]*512})
    with pytest.raises(ValueError,match='refusing to truncate'):
        annual_tensor(frame,['x'],start)


def test_streamed_inference_scores_only_observed_prices_without_warmup():
    from quant_orchestrator.research_tools.warehouse_multirate_training import predict_batch
    from quant_orchestrator.research_tools.warehouse_multirate import SPARSE_FAMILIES
    from quant_orchestrator.research_tools.annual_memory import AnnualMemory
    from quant_orchestrator.platforms.ml_frameworks.torch.models.transformers.multirate import (
        MultiRateTransformer, MultiRateTransformerConfig, add_subtoken_temporal_tasks,
    )
    from quant_orchestrator.platforms.ml_frameworks.torch.models.transformers.multirate.temporal_tasks import DOCUMENT_TASK_NAMES
    layout={'price':2}
    widths={**{r:(2,) for r in ('annual','quarterly','daily')},'sparse':(8,)*len(SPARSE_FAMILIES)}
    bundle=add_subtoken_temporal_tasks([],['price',*SPARSE_FAMILIES],{n:['unused'] for n in DOCUMENT_TASK_NAMES[1:]},feature_dimensions=widths)
    model=MultiRateTransformer({**{r:2 for r in ('annual','quarterly','daily')},'sparse':8*len(SPARSE_FAMILIES)},
        config=MultiRateTransformerConfig(d_model=8,num_heads=2,layers=1,max_position=16,cacheable_rate_states=True),
        feature_families={**{r:layout for r in ('annual','quarterly','daily')},'sparse':dict.fromkeys(SPARSE_FAMILIES,8)},
        modalities=['equity','option'],tasks=bundle.supervised_tasks,prediction_tasks=bundle.prediction_tasks).eval()
    first=datetime(2024,1,2)
    sample=dict(symbol='X',underlying_symbol='X',asset_class='equity',document_start='2024-01-01',date='2024-12-31',
        sequence_mode='annual_memory',daily_dates=[first,first+timedelta(days=1)],daily_score_valid=[True,False])
    for rate in ('annual','quarterly','daily','sparse','issuer_daily','issuer_sparse'):
        width=8*len(SPARSE_FAMILIES) if rate.endswith('sparse') else 2
        frame=pl.DataFrame({'date':[first,first+timedelta(days=1)],**{str(i):[1.,2.] for i in range(width)}})
        value,mask,dates=annual_tensor(frame,[str(i) for i in range(width)],first)
        sample.update({rate:value,rate+'_padding':mask,rate+'_timestamps':dates})
    with torch.inference_mode():
        result=predict_batch(model,[sample],AnnualMemory(),layout)
    assert len(result)==1 and result[0]['date']==first and result[0]['symbol']=='X'
    assert 0<=result[0]['oracle_is_buy']<=1
    # Exercise the exact shared training step after its extraction, including
    # issuer streams and option-specific supervised contributions.
    from collections import Counter
    from types import SimpleNamespace
    from quant_orchestrator.research_tools.multirate_training_step import make_training_step
    from quant_orchestrator.platforms.ml_frameworks.torch.models.transformers.multirate.temporal_tasks import SUPERVISED_TARGET_TASK_NAMES, PREDICTION_TASK_NAMES
    sample.update(asset_class='option',issuer_context_key='X',annual_context_key='X',quarterly_context_key='X',
        supervised_targets=torch.ones(3,len(SUPERVISED_TARGET_TASK_NAMES)),
        supervised_valid=torch.zeros(3,len(SUPERVISED_TARGET_TASK_NAMES),dtype=torch.bool))
    sample['supervised_valid'][1,0]=True
    observations=Counter()
    args=SimpleNamespace(sequence_mode='annual_memory',issuer_context='full',self_supervision='both',disable_document_tasks=True,training_sequence_stride=0)
    step=make_training_step(args=args,trainer=SimpleNamespace(current_epoch=0,current_step=0),device=torch.device('cpu'),
        annual_state=AnnualMemory(),reconstruction_widths=widths,feature_family_dimensions=layout,sparse_input_families=list(SPARSE_FAMILIES),
        raw_sparse_columns=list(range(8)),family_names=['price',*SPARSE_FAMILIES],asset_class_ids={'equity':0,'option':1},
        enabled_document_tasks=(),prediction_names=set(PREDICTION_TASK_NAMES),expected_task_names=SUPERVISED_TARGET_TASK_NAMES+PREDICTION_TASK_NAMES,
        mrl_dimensions=(),task_observations=observations,task_family_observations=Counter(),task_loss_sums=Counter(),epoch_clocks={},alignment_loss=None)
    model.train()
    tasks=tuple(t for t in bundle.tasks if t.name not in DOCUMENT_TASK_NAMES)
    loss=sum(step(model,[sample],tasks).values())
    assert torch.isfinite(loss)
    loss.backward()
    assert model.task_heads['oracle_is_buy'].weight.grad.abs().sum()>0
    assert observations['option:oracle_is_buy']==1


@pytest.mark.parametrize('expiry_count', [1, 4])
def test_first_session_uses_all_available_expirations_when_fewer_than_five(expiry_count):
    q = chain()
    expiries = sorted(q['expiration'].unique().to_list())[:expiry_count]
    q = q.filter(pl.col('expiration').is_in(expiries))
    members = first_session_baskets(q, q['snapshot_date'][0])
    assert members['document_symbol'].n_unique() == 2 * expiry_count
    assert set(members['contract_symbol']) == set(q['contract_symbol'])
    assert set(members['weight']) == {.5}


def test_option_target_builder_receives_contract_prices_not_issuer_prices(monkeypatch):
    from quant_orchestrator.research_tools import warehouse_multirate as module
    from quant_orchestrator.research_tools.multirate_targets import VALUE_COLUMNS
    stream=object.__new__(module.WarehouseAnnualStream)
    stream.start=datetime(2023,1,1);stream.end=datetime(2023,12,31);stream.cutoff=datetime(2024,1,1)
    stream.columns=['value__price.close'];stream.features=['price.close']
    dates=[datetime(2023,1,3),datetime(2023,1,4)]
    def prices(values):
        return pl.DataFrame({'date':dates,**{k:values for k in ('open','high','low','close','volume')}})
    stream.prices={'AAPL':prices([150.,160.])}
    empty=pl.DataFrame(schema={'symbol':pl.String,'date':pl.Datetime('ns'),'event_date':pl.Datetime('ns'),
        'target_family':pl.String,**dict.fromkeys(VALUE_COLUMNS,pl.Float32)})
    stream.source=lambda s: ({r:[] for r in ('annual','quarterly','daily')},empty)
    stream.common=lambda:[];stream.peer_context=lambda s:[]
    seen=[]
    def targets(symbol,p):
        seen.append((symbol,p['close'].to_list()))
        return empty
    monkeypatch.setattr(module,'materialize_instrument_targets',targets)
    members=pl.DataFrame([dict(document_symbol='AAPL230217C00150000',strike=150.,dte=45,option_type='call',
        expiration=datetime(2023,2,17),settlement=datetime(2023,2,17))])
    call=stream.sample('AAPL',2023,option_symbol='AAPL230217C00150000',members=members,prices=prices([2.,4.]))
    stream.prices['AAPL']=prices([900.,800.])
    stream.sample('AAPL',2023,option_symbol='AAPL230217C00150000',members=members,prices=prices([2.,4.]))
    equity=stream.sample('AAPL',2023)
    assert equity['daily'] is equity['issuer_daily']
    assert equity['sparse'] is equity['issuer_sparse']
    assert call['daily'] is not call['issuer_daily']
    from quant_orchestrator.research_tools.multirate_batch import BatchTensors
    staged = BatchTensors([equity], 'cpu')
    before = staged('issuer_daily').clone()
    staged('daily').zero_()
    torch.testing.assert_close(staged('issuer_daily'), before, equal_nan=True)
    assert seen==[('AAPL230217C00150000',[2.,4.]),('AAPL230217C00150000',[2.,4.]),('AAPL',[900.,800.])]
    assert call['option_type']=='call' and call['expiration']==datetime(2023,2,17)


def test_option_start_keeps_earlier_fmp_history(monkeypatch,tmp_path):
    from types import SimpleNamespace
    from quant_orchestrator.research_tools import warehouse_multirate as module
    price_reads=[];option_reads=[]
    def prices(symbol,**kwargs):
        price_reads.append(kwargs)
        return pl.DataFrame({'date':[datetime(2019,1,2),datetime(2023,1,3)],'close':[1.,2.]})
    def options(symbol,**kwargs):
        option_reads.append(kwargs)
        return pl.DataFrame({'snapshot_date':[datetime(y,1,4) for y in (2020,2021,2022,2023)]})
    monkeypatch.setattr(module,'read_option_chain_arctic',options)
    warehouse=SimpleNamespace(catalog=SimpleNamespace(query_symbol_profiles=lambda **k:[SimpleNamespace(symbol='AAPL')]),
        read_prices=prices,backend=SimpleNamespace(list_symbols=lambda library:['AAPL']))
    stream=module.WarehouseAnnualStream(min_market_cap=1e11,start='1900-01-01',option_start='2021-01-01',
        end='2024-12-31',cutoff='2024-01-01',output=tmp_path,warehouse=warehouse)
    assert price_reads[0]['start']=='1900-01-01'
    assert stream.prices['AAPL']['date'].min()==datetime(2019,1,2)
    assert option_reads[0]['start_date']=='2021-01-01'
    assert stream.option_years['AAPL']==[2021,2022,2023]
    assert stream.start==datetime(1900,1,1) and stream.option_start==datetime(2021,1,1)


def test_cached_raw_document_day_matches_fresh_query_and_excludes_other_dates():
    from quant_orchestrator.research_tools.warehouse_multirate import inference_day
    days=[datetime(2024,1,n) for n in (2,3,4)]
    day=days[1]
    frame=pl.DataFrame({'date':days,'x':[1.,2.,99999.]})
    document=dict(underlying_symbol='X',daily_dates=days,daily_score_valid=[True]*3,
                  prices=frame,supervised_targets=torch.zeros(4,2),supervised_valid=torch.zeros(4,2,dtype=torch.bool))
    for rate in ('annual','quarterly','daily','sparse','issuer_daily','issuer_sparse'):
        source=frame.filter(pl.col('date')!=day) if rate=='quarterly' else frame
        values,mask,timestamps=annual_tensor(source,['x'],days[0])
        document.update({rate:values,rate+'_padding':mask,rate+'_timestamps':timestamps})
    query=inference_day(document,day)
    for rate in ('annual','quarterly','daily','sparse','issuer_daily','issuer_sparse'):
        source=frame.head(0) if rate=='quarterly' else frame.filter(pl.col('date')==day)
        expected=annual_tensor(source,['x'],day)
        for suffix,value in zip(('','_padding','_timestamps'),expected):
            torch.testing.assert_close(query[rate+suffix],value,equal_nan=True)
    assert query['daily_dates']==[day]
    assert query['prices']['x'].to_list()==[2.]
    assert not query['supervised_valid'].any()
    assert document['prices'].height==3  # Shared raw document is unchanged.


def test_issuer_batches_train_before_preparing_next_issuer_options():
    from quant_orchestrator.research_tools.warehouse_multirate import WarehouseAnnualStream
    from quant_orchestrator.research_tools.warehouse_multirate_training import issuer_training_batches
    stream=WarehouseAnnualStream.__new__(WarehouseAnnualStream)
    stream.cutoff=datetime(2024,1,1)
    stream.prices={s:pl.DataFrame({'date':[datetime(y,1,3) for y in (2020,2021,2022)]}) for s in ('A','B')}
    stream.option_years={s:[2021,2022] for s in stream.prices}
    stream.sources={'unused_context_source':object()}
    events=[]
    def cohort(symbol,year):
        events.append(('audit',symbol,year))
        identities=[f'{symbol}_{year}_{right}' for right in ('call','put')]
        return pl.DataFrame({'document_symbol':identities}),pl.DataFrame({'symbol':identities})
    def sample(symbol,year,option_symbol=None,**kwargs):
        events.append(('build',symbol,year))
        return dict(symbol=option_symbol or symbol,issuer=symbol,year=year)
    stream.cohorts=cohort;stream.sample=sample
    batches=[]
    for batch in issuer_training_batches(stream,64,0):
        assert len({r['issuer'] for r in batch})==1
        assert len({r['symbol'] for r in batch})==len(batch)
        batches.append(batch)
        events.append(('train',batch[0]['issuer'],None))
    order=[group[0] for group in stream.issuer_groups(0)]
    first,second=order
    assert max(i for i,e in enumerate(events) if e[:2]==('train',first)) < min(i for i,e in enumerate(events) if e[1]==second)
    for symbol in stream.prices:
        assert [r['year'] for b in batches for r in b if r['symbol']==symbol]==[2020,2021,2022]
    assert sum(len(b) for b in batches)==14
    assert not stream.sources


def test_option_preparation_only_reads_requested_symbol_year(monkeypatch,tmp_path):
    from types import SimpleNamespace
    from quant_orchestrator.research_tools import warehouse_multirate as module
    stream=module.WarehouseAnnualStream.__new__(module.WarehouseAnnualStream)
    stream.output=tmp_path;stream.end=datetime(2023,12,31)
    stream.selection_policy=module.selection_policy(8)
    stream.selected_cohorts=set();stream.option_years={'A':[2021],'B':[2021]}
    stream.coverage={s:dict(cohorts=[]) for s in ('A','B')}
    stream.warehouse=SimpleNamespace(backend=None);stream.write_coverage=lambda:None
    reads=[]
    def candidates(warehouse,symbol,year,*args):
        reads.append((symbol,year))
        audit=pl.DataFrame([dict(underlying_symbol=symbol,contract_symbol=f'{symbol}C{i}',document_symbol=f'{symbol}C{i}',
            option_type='call',moneyness=1.,profit_pct=20.,valid_quote_days=30,quote_coverage=1.) for i in range(20)])
        return audit,pl.DataFrame({'symbol':audit['contract_symbol'],'date':[datetime(year,1,4)]*20}),'audited'
    monkeypatch.setattr(module,'contract_candidates',candidates)
    assert stream.cohorts('A',2021)[0].height==8
    assert stream.cohorts('A',2021)[0].height==8
    assert reads==[('A',2021)]
    assert stream.coverage['B']['cohorts']==[]
    assert not (tmp_path/'sampled_contracts/2021/B').exists()


def test_invalid_option_sample_size_fails_before_warehouse_access():
    import pytest
    from quant_orchestrator.research_tools.warehouse_multirate import WarehouseAnnualStream
    for count in (-1, True, 1.5):
        with pytest.raises(ValueError, match='options_per_side'):
            WarehouseAnnualStream(min_market_cap=1e10, start='2021-01-01', end='2024-01-01',
                cutoff='2024-01-01', output='unused', options_per_side=count)


def test_equities_only_never_discovers_or_loads_options(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from quant_orchestrator.research_tools import warehouse_multirate as module
    def forbidden(*args, **kwargs):
        raise AssertionError('Options must not be accessed')
    prices = pl.DataFrame({'date': [datetime(2023,1,3), datetime(2023,1,4)], 'close': [1.,2.]})
    warehouse = SimpleNamespace(
        catalog=SimpleNamespace(query_symbol_profiles=lambda **k: [SimpleNamespace(symbol='A')]),
        read_prices=lambda *a, **k: prices, backend=SimpleNamespace(list_symbols=forbidden))
    monkeypatch.setattr(module, 'read_option_chain_arctic', forbidden)
    stream = module.WarehouseAnnualStream(min_market_cap=1e10, start='1900-01-01',
        end='2024-12-31', cutoff='2024-01-01', option_start='unused', options_per_side=0,
        output=tmp_path, warehouse=warehouse)
    assert stream.expected_option_symbols == set()
    assert stream.option_years == {'A': []}
    stream.cohorts = forbidden
    stream.sample = lambda symbol, year, **kwargs: (symbol, year)
    assert list(stream.documents()) == [('A', 2023)]


def test_equities_only_evaluation_does_not_run_option_backtests(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from quant_orchestrator.research_tools import warehouse_multirate_training as module
    from quant_orchestrator.platforms.backtesting_frameworks import existing_multirate_backtest as equity
    from quant_orchestrator.platforms.backtesting_frameworks import equity_option_trade_backtest as option
    day = datetime(2024,1,2)
    prices = pl.DataFrame({'date': [day], **{k: [1.] for k in ('open','high','low','close','volume')}})
    sample = dict(symbol='A', asset_class='equity', prices=prices)
    stream = SimpleNamespace(documents=lambda **k: iter([sample]), layout={},
        selection_policy={'contracts_per_side': 0})
    monkeypatch.setattr(module, 'predict_batch', lambda *a: [dict(symbol='A', date=day,
        **{name: .5 for name in module.SUPERVISED_TARGET_TASK_NAMES})])
    monkeypatch.setattr(equity, 'run_existing_multirate_backtest', lambda *a, **k:
        [dict(side=side, capital_return=0.) for side in ('long', 'short')])
    def forbidden(*a, **k):
        raise AssertionError('Option backtest must not run')
    monkeypatch.setattr(option, 'run_equity_option_trade_backtest', forbidden)
    args = SimpleNamespace(output_dir=tmp_path, prediction_start_date='2024-01-01',
        prediction_end_date='2024-12-31', batch_size=1)
    reports = module.evaluate_epoch(SimpleNamespace(eval=lambda: None), stream, args, 1)
    assert [(r['asset_class'], r['side']) for r in reports] == [('equity','long'), ('equity','short')]


def test_equity_batches_fill_across_issuers_and_preserve_memory_order():
    from types import SimpleNamespace
    from collections import Counter
    from quant_orchestrator.research_tools.warehouse_multirate_training import training_batches
    years = {'A': [2020,2021,2022], 'B': [2019,2022], 'C': [2021,2022,2023],
             'D': [2020], 'E': [2018,2023]}
    builds, processed = [], Counter()
    stream = SimpleNamespace(
        prices={s: pl.DataFrame({'date': [datetime(y,1,3) for y in ys+[2024]]}) for s,ys in years.items()},
        cutoff=datetime(2024,1,1), selection_policy={'contracts_per_side': 0}, sources={},
        prepare_equity_sources=lambda symbols: None)
    def sample(symbol, year, **kwargs):
        builds.append((symbol,year))
        return dict(symbol=symbol, issuer=symbol, year=year)
    stream.sample = sample
    batches = training_batches(stream, 3, 0)
    assert not builds
    seen = []
    for batch in batches:
        assert len(batch) <= 3
        assert len({r['symbol'] for r in batch}) == len(batch)
        for row in batch:
            symbol = row['symbol']
            # Simulate the memory update after each forward/optimizer step.
            assert row['year'] == years[symbol][processed[symbol]]
            processed[symbol] += 1
        seen.append(batch)
    assert len(seen[0]) == 3
    assert sum(map(len, seen)) == sum(map(len, years.values()))
    assert len(seen) < sum(map(len, years.values()))
    assert not stream.sources
    repeat = list(training_batches(stream, 3, 0))
    assert repeat == seen


def test_equity_scheduler_reduces_full_epoch_optimizer_steps():
    from types import SimpleNamespace
    from quant_orchestrator.research_tools.warehouse_multirate_training import training_batches
    stream = SimpleNamespace(
        prices={f'S{i}': pl.DataFrame({'date':[datetime(y,1,3) for y in range(1994,2024)]}) for i in range(840)},
        cutoff=datetime(2024,1,1), selection_policy={'contracts_per_side':0}, sources={},
        prepare_equity_sources=lambda symbols: None,
        sample=lambda symbol, year, **k: dict(symbol=symbol,year=year))
    batches = list(training_batches(stream,64,0))
    assert sum(map(len,batches)) == 25200
    assert len(batches) == 420  # 13 full groups and one partial group, 30 years each.
    assert all(len({r['symbol'] for r in b})==len(b) for b in batches)


def test_merge_observations_preserves_order_nulls_and_missing_columns():
    from quant_orchestrator.research_tools.warehouse_multirate import merge_observations
    from polars.testing import assert_frame_equal
    d = datetime(2023,1,3)
    frames = [pl.DataFrame({'date':[d,d], 'x':[1.,None]}),
              pl.DataFrame({'date':[d], 'x':[2.], 'y':[3.]})]
    result = merge_observations(frames, ['y','missing','x'])
    expected = pl.DataFrame({'date':[d], 'y':[3.], 'missing':pl.Series([None],dtype=pl.Float32), 'x':[2.]}).with_columns(pl.col('date').cast(pl.Datetime('ns')))
    assert_frame_equal(result, expected)
    assert merge_observations([], ['x']).schema == {'date':pl.Datetime('ns'), 'x':pl.Float32}


def test_sparse_schema_tensor_matches_explicit_null_padding():
    from quant_orchestrator.research_tools.warehouse_multirate import annual_tensor, merge_observations
    frame = pl.DataFrame({'date':[datetime(2021,1,4),datetime(2021,1,5)],
                          'observed':[1.,float('inf')]})
    columns = ['absent_before','observed','absent_after']
    full = merge_observations([frame],columns)
    compact = merge_observations([frame],columns,pad_schema=False)
    for a,b in zip(annual_tensor(full,columns,datetime(2021,1,1)),
                   annual_tensor(compact,columns,datetime(2021,1,1))):
        torch.testing.assert_close(a,b,rtol=0,atol=0,equal_nan=True)


def test_peer_context_cache_preserves_native_dates():
    from types import SimpleNamespace
    from quant_orchestrator.research_tools.warehouse_multirate import WarehouseAnnualStream
    stream = WarehouseAnnualStream.__new__(WarehouseAnnualStream)
    stream.peer_frames = {}
    stream.profiles = {'A':SimpleNamespace(sector='Tech'), 'B':SimpleNamespace(sector='Tech')}
    stream.contexts = [('peer','sector',pl.DataFrame({'date':[datetime(2021,1,4)],'sector':['Tech'],'value':[2.]}))]
    first,second = stream.peer_context('A'),stream.peer_context('B')
    assert first[0] is second[0]
    assert first[0]['date'].to_list() == [datetime(2021,1,4)]
    assert first[0]['value__peer.value'].to_list() == [2.]


def test_equity_sources_warm_after_peer_initialization():
    from quant_orchestrator.research_tools.warehouse_multirate import WarehouseAnnualStream
    stream = WarehouseAnnualStream.__new__(WarehouseAnnualStream)
    stream.source_cache_limit = 1
    calls = []
    stream.common = lambda: calls.append(('macro',None))
    stream.peer_context = lambda s: calls.append(('peer',s))
    stream.source = lambda s: calls.append(('source',s))
    stream.prepare_equity_sources(['A','B'])
    assert stream.source_cache_limit == 2
    assert calls == [('macro',None),('peer','A'),('peer','B'),('source','A'),('source','B')]
