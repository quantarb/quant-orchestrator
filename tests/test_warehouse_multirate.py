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
