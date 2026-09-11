from datetime import datetime
import copy
import polars as pl
import pytest
import torch
from quant_orchestrator.research_tools.document_sequences import document_anchors, document_window, prediction_positions
from quant_orchestrator.research_tools.streaming_context import StreamingContext
from quant_orchestrator.research_tools.multirate_batch import BatchTensors
from quant_orchestrator.platforms.ml_frameworks.torch.models.transformers.multirate import MultiRateTransformer, MultiRateTransformerConfig, MultiRateTaskSpec


def test_calendar_ownership_covers_every_date_once_and_is_prefix_stable():
    dates = [datetime(2023,12,29),datetime(2024,1,2),datetime(2024,2,1),datetime(2024,3,29),datetime(2024,4,1)]
    scan = pl.DataFrame({'symbol':['X']*len(dates), 'date':dates}).lazy()
    full = document_anchors([scan], start='2024-01-01',end='2024-04-01')
    live = document_anchors([scan], start='2024-01-01',end='2024-02-01')
    assert full['document_start'].to_list() == [datetime(2024,1,1),datetime(2024,4,1)]
    assert live['document_start'][0] == full['document_start'][0]
    assert live['date'][0] == datetime(2024,2,1)
    owners=[]
    for row in full.iter_rows(named=True):
        item={'sequence_mode':'documents', 'document_start':str(row['document_start'].date()),
              'date':str(row['date'].date()), 'daily_dates':[str(d.date()) for d in dates if d<=row['date']]}
        owners.extend(date for _,date in prediction_positions(item,start='2024-01-01'))
    assert owners == [str(d.date()) for d in dates[1:]]
    train=document_anchors([scan],cutoff='2024-01-01')
    assert train['date'].max() < datetime(2024,1,1)


def test_document_window_keeps_history_and_all_updates_without_moving_prefix():
    dates=[datetime(2023,12,1),datetime(2023,12,29),datetime(2024,1,2),datetime(2024,2,1)]
    index=StreamingContext(pl.DataFrame({'symbol':['X']*4,'date':dates,'x':[1.,2.,3.,4.]}).lazy(),['x'])
    prefix=document_window(index,'X',datetime(2024,1,1),dates[2],2)
    full=document_window(index,'X',datetime(2024,1,1),dates[3],2)
    torch.testing.assert_close(prefix[0],full[0][:len(prefix[0])],equal_nan=True)
    assert prefix[3].tolist()==full[3][:len(prefix[3])].tolist()
    assert full[0][~full[1]].flatten().tolist()==[1.,2.,3.,4.]
    batch=[{'sequence_mode':'documents','daily':p[0],'daily_padding':p[1],'daily_timestamps':p[3]} for p in (prefix,full)]
    tensors=BatchTensors(batch,'cpu')
    assert tensors('daily_timestamps')[0,-1] == torch.iinfo(torch.long).max
    assert tensors('daily_padding')[0,-1]
    torch.testing.assert_close(tensors('daily')[0,:len(prefix[0])],prefix[0],equal_nan=True)


def model_inputs(length):
    data={}
    for rate in ('annual','quarterly','daily','sparse'):
        values=torch.randn(1,length,2)
        values[:,0]=0
        dates=torch.tensor([[torch.iinfo(torch.long).min,*[1704067200000000000+i*86400000000000 for i in range(length-1)]]])
        padding=torch.zeros(1,length,dtype=torch.bool)
        data[f'{rate}_values']=values
        data[f'{rate}_dates']=dates
        data[f'{rate}_padding_mask']=padding
    data['issuer_streams']={rate:{'values':data[f'{rate}_values'].clone(), 'dates':data[f'{rate}_dates'].clone(),
                                  'padding':data[f'{rate}_padding_mask'].clone()} for rate in ('daily','sparse')}
    return data


def test_full_document_matches_live_prefix_including_issuer_updates():
    torch.manual_seed(3)
    model=MultiRateTransformer({r:2 for r in ('annual','quarterly','daily','sparse')},
        config=MultiRateTransformerConfig(d_model=8,num_heads=2,layers=1,dropout=0),
        tasks=[MultiRateTaskSpec('return','token',source='daily')]).eval()
    full=model_inputs(7)
    prefix={key:(value[:,:4] if torch.is_tensor(value) else {r:{k:t[:,:4] for k,t in payload.items()} for r,payload in value.items()}) for key,value in full.items()}
    with torch.inference_mode():
        a=model(**full,compute_document_outputs=False)['token_outputs']['return'][:,:4]
        b=model(**prefix,compute_document_outputs=False)['token_outputs']['return']
        changed=copy.deepcopy(full)
        for key,value in changed.items():
            if key.endswith('_values'): value[:,4:]+=1000
        for payload in changed['issuer_streams'].values(): payload['values'][:,4:]+=1000
        c=model(**changed,compute_document_outputs=False)['token_outputs']['return'][:,:4]
    torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5)
    torch.testing.assert_close(a,c,atol=2e-6,rtol=2e-5)


def test_document_labels_use_each_event_date_and_exclude_context():
    from types import SimpleNamespace
    from quant_orchestrator.research_tools.sequence_training import window_supervision
    dates=pl.Series([datetime(2023,12,29),datetime(2024,1,2),datetime(2024,1,3)]).dt.epoch('ns').to_torch()
    store=SimpleNamespace(scan=pl.DataFrame({'symbol':['X']*3,'date':[datetime(2023,12,29),datetime(2024,1,2),datetime(2024,1,3)],'buy':[1.,1.,None],'sell':[None,None,1.]}).lazy())
    targets,valid=window_supervision(store,'X',dates,length=4,tasks=('buy','sell'),start=datetime(2024,1,1),end=datetime(2024,1,3),positions=slice(1,4))
    assert not valid[:2].any()
    assert valid[2].tolist()==[True,False]
    assert valid[3].tolist()==[False,True]
    assert targets[2,0]==targets[3,1]==1


def test_score_coverage_rejects_missing_duplicate_and_nonfinite_rows():
    from quant_orchestrator.research_tools.document_sequences import validate_document_predictions
    expected=pl.DataFrame({'symbol':['X','X'],'date':[datetime(2024,1,2),datetime(2024,1,3)]})
    scores=expected.with_columns(pl.Series('score',[.2,.3]))
    assert validate_document_predictions(scores.lazy(),expected.lazy())['prediction_rows']==2
    for broken in [scores.head(1),pl.concat([scores,scores.head(1)]),scores.with_columns(pl.lit(float('nan')).alias('score'))]:
        with pytest.raises(ValueError,match='coverage failed'):
            validate_document_predictions(broken.lazy(),expected.lazy())
