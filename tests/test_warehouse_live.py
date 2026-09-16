from datetime import datetime
from types import SimpleNamespace

import polars as pl
import pytest
import torch

from quant_orchestrator.research_tools import warehouse_live as live
from quant_orchestrator.research_tools import warehouse_multirate_training as training


def test_latest_date_comes_from_stored_prices_not_wall_clock():
    frames = {
        'A':pl.DataFrame({'date':[datetime(2026,6,1),datetime(2026,6,2)],'close':[1.,2.]}),
        'B':pl.DataFrame({'date':[datetime(2026,5,1),datetime(2026,6,3)],'close':[2.,float('nan')]}),
    }
    reads=[]
    def read(symbol, **kwargs):
        reads.append(kwargs)
        return frames[symbol]
    warehouse=SimpleNamespace(catalog=SimpleNamespace(query_symbol_profiles=lambda **kwargs:[SimpleNamespace(symbol=s) for s in frames]),read_prices=read)
    assert live.latest_warehouse_equity_date(1e10,warehouse=warehouse)=='2026-06-02'
    assert all(r['start']=='1900-01-01' and r['provider']=='fmp' for r in reads)


def test_latest_training_includes_partial_year_and_disables_backtests(tmp_path,monkeypatch):
    monkeypatch.setattr(live,'latest_warehouse_equity_date',lambda *a,**k:'2026-06-02')
    def run(args, **kwargs):
        assert args.train_end_date=='2026-06-03'
        assert args.prediction_start_date==args.prediction_end_date=='2026-06-02'
        assert args.warehouse_start_date=='1900-01-01' and args.options_per_side==0
        assert args.self_supervision=='both' and kwargs['latest_only'] is True
        assert args.checkpoint is None and not args.resume_training
        return {'stage':'complete'}
    monkeypatch.setattr(training,'run_warehouse_training',run)
    assert live.train_latest_warehouse_model(tmp_path/'new',warehouse=object())=={'stage':'complete'}
    with pytest.raises(FileExistsError):
        live.train_latest_warehouse_model(tmp_path,warehouse=object())


def test_scheduler_trains_partial_current_year_but_not_post_cutoff():
    dates=[datetime(2023,1,1),datetime(2024,1,1),datetime(2024,6,1),datetime(2025,1,1)]
    seen=[]
    stream=SimpleNamespace(prices={'A':pl.DataFrame({'date':dates})},cutoff=datetime(2024,7,1),sources={},
        prepare_equity_sources=lambda symbols:None,
        sample=lambda symbol,year,**kwargs:seen.append(year) or {'symbol':symbol,'year':year})
    batches=list(training.equity_training_batches(stream,64,0))
    assert [row['year'] for batch in batches for row in batch]==[2023,2024]
    assert seen==[2023,2024]


def test_latest_scoring_keeps_year_context_and_emits_no_stale_dates(tmp_path,monkeypatch):
    date=datetime(2026,6,2)
    prices={'A':pl.DataFrame({'date':[datetime(2026,1,2),date],'close':[1.,2.]}),
            'STALE':pl.DataFrame({'date':[datetime(2026,1,2)],'close':[3.]})}
    stream=SimpleNamespace(prices=prices,layout={})
    def batches(actual,size,start,end,*,symbols):
        assert actual is stream and start==datetime(2026,1,1) and end==date and symbols==['A']
        yield [{'symbol':'A'}]
    def predict(model,batch,memory,layout,*,score_date):
        assert score_date==date
        return [dict(symbol='A',date=date,**{k:.5 for k in training.SUPERVISED_TARGET_TASK_NAMES})]
    monkeypatch.setattr(training,'equity_inference_batches',batches)
    monkeypatch.setattr(training,'predict_batch',predict)
    result=live.score_latest_equities(torch.nn.Linear(1,1),stream,
        SimpleNamespace(prediction_end_date='2026-06-02',batch_size=64,output_dir=tmp_path))
    assert result['missing_latest_prices']==['STALE'] and result['predictions']==1
    assert result['evaluation_mode']=='latest_date_in_sample'
    assert pl.read_parquet(result['prediction_path'])['date'].to_list()==[date]
