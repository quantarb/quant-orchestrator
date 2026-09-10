from datetime import datetime,timedelta
from types import SimpleNamespace
import polars as pl
import torch
from quant_orchestrator.research_tools.sequence_training import sequence_anchors,window_supervision


def test_sequences_own_every_event_once_without_synthetic_labels():
    dates=[datetime(2020,1,1)+timedelta(days=i) for i in range(21)]
    daily=pl.DataFrame({'symbol':['A']*21,'date':dates}).lazy()
    events=pl.DataFrame({'symbol':['A']*6,'date':[dates[i] for i in [0,3,4,8,15,20]]})
    owners,report=sequence_anchors(daily,events,cutoff='2020-01-21',stride=4,window=8)
    assert report['matched_event_dates']==5
    assert report['unmatched_event_dates']==1
    labels=events.with_columns(pl.lit(1.).alias('buy')).lazy()
    store=SimpleNamespace(scan=labels)
    observed=[]
    for row in owners.iter_rows(named=True):
        history=[d for d in dates if d<=row['date']][-8:]
        ns=pl.Series(history).dt.epoch('ns').to_torch()
        targets,valid=window_supervision(store,'A',ns,length=8,tasks=('buy',),start=row['supervision_start'],end=row['date'])
        observed.extend([d for d,keep in zip(history,valid[-len(history):,0].tolist()) if keep])
        assert targets[valid].tolist()==[1.]*int(valid.sum())
    assert sorted(observed)==[dates[i] for i in [0,3,4,8,15]]
    assert len(set(observed))==len(observed)


def test_sequence_boundaries_keep_adjacent_temporal_pairs():
    dates=[datetime(2020,1,1)+timedelta(days=i) for i in range(20)]
    events=pl.DataFrame({'symbol':['A']*20,'date':dates})
    owners,_=sequence_anchors(events.lazy(),events,cutoff='2021-01-01',stride=4,window=8)
    pairs=set()
    for end in owners['date']:
        history=[d for d in dates if d<=end][-8:]
        pairs.update(zip(history,history[1:]))
    assert pairs==set(zip(dates,dates[1:]))


def test_sequence_labels_match_exact_dates_and_keep_missing_tasks_masked():
    dates=torch.tensor([0,86_400_000_000_000,172_800_000_000_000])
    store=SimpleNamespace(scan=pl.DataFrame({'symbol':['A','A','A'],
        'date':[datetime(1970,1,1),datetime(1970,1,2),datetime(1970,1,4)],
        'buy':[1.,0.,1.],'rank':[None,.2,.8]}).lazy())
    targets,valid=window_supervision(store,'A',dates,length=4,tasks=('buy','rank'),
        start=datetime(1970,1,2),end=datetime(1970,1,3))
    assert valid.tolist()==[[False,False],[False,False],[True,True],[False,False]]
    torch.testing.assert_close(targets[2],torch.tensor([0.,.2]))


def test_sequence_context_retains_start_history_and_all_intermediate_updates():
    from quant_orchestrator.research_tools.streaming_context import StreamingContext
    dates=[datetime(2020,1,1)+timedelta(days=i) for i in range(30)]
    data=pl.DataFrame({'symbol':['A']*30,'date':dates,'x':list(range(30))})
    context=StreamingContext(data.lazy(),['x'])
    values,padding,ns,_=context.sequence_window('A',dates[10],dates[20],4)
    assert values[~padding,0].tolist()==list(range(7,21))
    # Every supervised position can access its own prior four observations.
    for day in range(10,21):
        visible=values[~padding][ns<=pl.Series([dates[day]]).dt.epoch('ns')[0]]
        assert visible[-4:,0].tolist()==list(range(day-3,day+1))


def test_variable_sequence_context_padding_preserves_values_dates_and_missingness():
    from quant_orchestrator.research_tools.multirate_batch import BatchTensors
    batch=[{'annual':torch.tensor([[1.],[2.]]),'annual_padding':torch.tensor([False,False]),'annual_timestamps':torch.tensor([1,2])},
           {'annual':torch.tensor([[3.],[4.],[5.]]),'annual_padding':torch.tensor([False]*3),'annual_timestamps':torch.tensor([3,4,5])}]
    tensors=BatchTensors(batch,torch.device('cpu'))
    assert tensors('annual').shape==(2,3,1)
    assert torch.isnan(tensors('annual')[0,0]).all()
    assert tensors('annual_padding').tolist()==[[True,False,False],[False]*3]
    assert tensors('annual_timestamps')[0].tolist()==[torch.iinfo(torch.long).min,1,2]
