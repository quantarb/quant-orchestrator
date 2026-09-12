from datetime import datetime
import copy
import polars as pl
import pytest
import torch
from quant_orchestrator.research_tools.annual_memory import AnnualCorpus, AnnualMemory, annual_window
from quant_orchestrator.research_tools.document_sequences import document_anchors
from quant_orchestrator.research_tools.annual_memory import cold_inference_anchors, inference_interval
from quant_orchestrator.research_tools.streaming_context import StreamingContext, StreamingFamilyContext
from quant_orchestrator.platforms.ml_frameworks.torch.models.transformers.multirate import MultiRateTransformer, MultiRateTransformerConfig, MultiRateTaskSpec


def test_year_boundaries_include_january_first_and_no_previous_raw_history():
    dates=[datetime(2023,12,31),datetime(2024,1,1),datetime(2024,2,29),datetime(2024,12,31),datetime(2025,1,1)]
    scan=pl.DataFrame({'symbol':['X']*5,'date':dates,'x':[1.,2.,3.,4.,5.]}).lazy()
    anchors=document_anchors([scan],period='1y',cutoff='2025-01-01')
    assert anchors['document_start'].to_list()==[datetime(2023,1,1),datetime(2024,1,1)]
    index=StreamingContext(scan,['x'])
    result=annual_window(index,'X',datetime(2024,1,1),datetime(2024,12,31))
    assert result[0][~result[1]].flatten().tolist()==[2.,3.,4.]
    assert result[3][0]<result[3][1]
    sparse=StreamingFamilyContext(scan.with_columns(pl.lit('f').alias('target_family')),['x'],families=['f'])
    result=annual_window(sparse,'X',datetime(2024,1,1),datetime(2024,12,31))
    assert result[0][~result[1]].flatten().tolist()==[2.,3.,4.]


def test_cold_inference_excludes_prior_years_and_pre_start_days():
    dates=[datetime(2023,12,31),datetime(2024,1,2),datetime(2024,6,3),datetime(2024,12,31),datetime(2025,1,2)]
    scan=pl.DataFrame({'symbol':['X']*5,'date':dates,'x':[1.,2.,3.,4.,5.]}).lazy()
    anchors=cold_inference_anchors([scan],'2024-06-03','2025-01-02')
    assert anchors['document_start'].to_list()==[datetime(2024,6,3),datetime(2025,1,1)]
    first=anchors.row(0,named=True)
    values,padding,*_=annual_window(StreamingContext(scan,['x']),'X',first['document_start'],first['date'])
    assert values[~padding].flatten().tolist()==[3.,4.]
    last_day=cold_inference_anchors([scan],'2025-01-02','2025-01-02')
    assert last_day.height==1 and last_day['document_start'][0]==dates[-1]
    assert inference_interval(scan,'2025-01-02','2025-01-02').collect().height==1
    with pytest.raises(ValueError,match='requires explicit'):
        inference_interval(scan,None,'2025-01-02')


def rows():
    return [{'symbol':s,'date':f'{y}-12-31','document_start':f'{y}-01-01'} for s in ['X','Y','Z'] for y in range(2020,2024)]


def test_batches_preserve_all_years_and_one_instrument_per_batch():
    corpus=AnnualCorpus(rows(),2)
    batches=list(corpus.batches(seed=7,epoch=1))
    assert sum(map(len,batches))==12
    for batch in batches:
        assert len({r['symbol'] for r in batch})==len(batch)
    for symbol in ['X','Y','Z']:
        assert [r['date'] for b in batches for r in b if r['symbol']==symbol]==[f'{y}-12-31' for y in range(2020,2024)]
    assert corpus.batch_count(seed=7,epoch=1)==len(batches)


def model():
    torch.manual_seed(42)
    return MultiRateTransformer({r:2 for r in ['annual','quarterly','daily','sparse']},
        config=MultiRateTransformerConfig(d_model=8,num_heads=2,layers=1,dropout=0),
        tasks=[MultiRateTaskSpec('return','token',source='daily')])


def inputs(year=2023, length=5):
    result={}
    dates=torch.tensor([[int((datetime(year,1,1)-datetime(1970,1,1)).total_seconds()*1e9)-1]+[int((datetime(year,i,2)-datetime(1970,1,1)).total_seconds()*1e9) for i in range(1,length)]])
    for rate in ['annual','quarterly','daily','sparse']:
        v=torch.arange(length*2,dtype=torch.float32).view(1,length,2)/10
        v[:,0]=0
        result[rate+'_values']=v
        result[rate+'_dates']=dates
        result[rate+'_padding_mask']=torch.zeros(1,length,dtype=torch.bool)
    return result


def test_prior_year_changes_next_year_but_future_year_updates_cannot_change_prefix():
    m=model().eval(); state=AnnualMemory()
    r23={'symbol':'X','date':'2023-12-31','document_start':'2023-01-01'}
    r24={'symbol':'X','date':'2024-12-31','document_start':'2024-01-01'}
    with torch.no_grad():
        prev=m(**inputs(),annual_memory=state.inputs([r23],m),compute_document_outputs=False)
        state.update([r23],prev)
        memory=state.inputs([r24],m)
        full=m(**inputs(2024),annual_memory=memory,compute_document_outputs=False)
        short={k:v[:,:3] for k,v in inputs(2024).items()}
        prefix=m(**short,annual_memory=memory,compute_document_outputs=False)
        torch.testing.assert_close(full['token_outputs']['return'][:,:3],prefix['token_outputs']['return'],atol=2e-6,rtol=2e-5)
        cold=m(**inputs(2024),annual_memory=AnnualMemory().inputs([r24],m),compute_document_outputs=False)
        assert not torch.allclose(full['token_outputs']['return'],cold['token_outputs']['return'])
    with pytest.raises(ValueError,match='precede'):
        state.inputs([r23],m)
    assert not any(t.requires_grad for p in state.states['X']['rates'].values() for t in p.values())
    state.begin_epoch(1)
    assert state.states=={}


def test_model_optimizer_and_memory_resume_match_uninterrupted_training(tmp_path):
    def step(m,opt,mem,year):
        row={'symbol':'X','date':f'{year}-12-31','document_start':f'{year}-01-01'}
        opt.zero_grad()
        out=m(**inputs(year),annual_memory=mem.inputs([row],m),compute_document_outputs=False)
        loss=out['token_outputs']['return'][:,1:].square().mean()
        mem.update([row],out);loss.backward();opt.step()
        return loss.detach()
    m=model();opt=torch.optim.AdamW(m.parameters(),lr=.001);mem=AnnualMemory();mem.begin_epoch(0)
    step(m,opt,mem,2022)
    torch.save({'model':m.state_dict(),'optimizer':opt.state_dict(),'memory':mem.state_dict()},tmp_path/'checkpoint.pt')
    expected=step(m,opt,mem,2023)
    saved=torch.load(tmp_path/'checkpoint.pt',weights_only=True)
    restored=model();restored.load_state_dict(saved['model']);other=torch.optim.AdamW(restored.parameters(),lr=.001);other.load_state_dict(saved['optimizer'])
    memory=AnnualMemory();memory.load_state_dict(saved['memory']);memory.begin_epoch(0)
    actual=step(restored,other,memory,2023)
    torch.testing.assert_close(expected,actual,atol=0,rtol=0)
    for key,value in m.state_dict().items():
        torch.testing.assert_close(value,restored.state_dict()[key],atol=0,rtol=0)
