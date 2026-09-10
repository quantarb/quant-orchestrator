import pytest
from quant_orchestrator.research_tools.epoch_evaluation import (
    evaluation_command, last_epoch_batches, trend_report, format_epoch_report,
)


def report(skill, mse=2.):
    return {'metrics': [dict(rate='daily', level='subtoken', family='price',
        values=10, unique_pairs=2, persistence_mse=4., model_mse=mse,
        skill=skill, beats_persistence=mse<4.)]}


def test_trends_compare_fixed_targets_and_show_improvement():
    first=trend_report(1, report(-.5,6.))
    second=trend_report(2, report(.5,2.), first)
    assert second['metrics'][0]['delta_skill']==1.
    assert second['groups_beating_persistence']==1
    assert 'delta_skill' in format_epoch_report(second)
    changed=report(.5);changed['metrics'][0]['values']=11
    with pytest.raises(ValueError, match='changed'):
        trend_report(3, changed, second)


def test_constant_baseline_does_not_produce_false_skill_or_trend():
    data=report(None,1.);data['metrics'][0]['persistence_mse']=0.
    data['metrics'][0]['beats_persistence']=False
    first=trend_report(1,data)
    second=trend_report(2,data,first)
    assert second['zero_error_baseline_groups']==1
    assert second['metrics'][0]['delta_skill'] is None
    assert 'null' in format_epoch_report(second)


def test_evaluation_command_never_trains_or_changes_training_dates():
    base=['python','train.py','--train-end-date','2024-01-01','--output-dir','train',
        '--skip-predictions','--max-samples','2048']
    command=evaluation_command(base,'snapshot.pt','validation','2024-01-02','2024-12-31',256)
    assert '--skip-predictions' not in command
    assert command.count('--inference-only')==1
    assert command[command.index('--train-end-date')+1]=='2024-01-01'
    assert command[command.index('--max-samples')+1]=='256'
    assert command[command.index('--checkpoint')+1]=='snapshot.pt'
    assert '--inference-only' not in base


def test_completed_epoch_size_is_read_from_training_progress(tmp_path):
    log=tmp_path/'train.log'
    log.write_text('[multirate-train] epoch=1/12 batch=762/762 samples=86081\n'
                   '[multirate-train] epoch=2/12 batch=10/763 samples=1280\n')
    assert last_epoch_batches(log)=={0:762,1:763}


def test_epoch_backtest_freezes_prices_and_reports_return_changes(tmp_path, monkeypatch):
    from datetime import date
    import polars as pl
    from quant_orchestrator.research_tools.epoch_evaluation import anchored_epoch_backtest, format_backtest_report
    corpus=tmp_path/'corpus';corpus.mkdir()
    pl.DataFrame({'symbol':['A','DELISTED'],'asset_class':['equity','equity']}).write_csv(corpus/'taxonomy.csv')
    dates=[date(2024,1,d) for d in (5,8,9)]
    calls=[]
    class Warehouse:
        def read_prices(self,symbol,**kwargs):
            calls.append(kwargs)
            return pl.DataFrame({'date':dates,'close':[100.,200.,220.]})
    monkeypatch.setattr('quant_warehouse.Warehouse',Warehouse)
    previous=None
    for epoch in (1,2):
        directory=tmp_path/f'epoch_{epoch}';directory.mkdir()
        pl.DataFrame({'date':dates,'symbol':['A']*3,
            **{f'hits_{side}_return_{role}':[.1]*3 for side in ('long','short') for role in ('hub','authority')}}).write_csv(directory/'supervised_predictions.csv')
        reports=anchored_epoch_backtest(['python','train','--corpus',str(corpus)],directory,'2024-01-05','2024-01-09',previous)
        assert len(reports)==2
        if previous:
            assert all(r['return_change_vs_previous_epoch']==0 for r in reports)
        previous=reports
    assert len(calls)==1 and calls[0]['adjustment']=='splits_and_dividends'
    assert reports[0]['capital_return']>0 and reports[1]['capital_return']==0
    assert 'backtest[2]' in format_backtest_report(reports)


def test_yearly_backtests_isolate_dates_capital_and_price_snapshots(tmp_path, monkeypatch):
    from datetime import date
    import polars as pl
    from quant_orchestrator.research_tools.epoch_evaluation import yearly_epoch_backtests, format_backtest_report
    corpus=tmp_path/'corpus';corpus.mkdir()
    pl.DataFrame({'symbol':['A'],'asset_class':['equity']}).write_csv(corpus/'taxonomy.csv')
    calls=[]
    class Warehouse:
        def read_prices(self, symbol, **kwargs):
            calls.append(kwargs)
            year=int(kwargs['start'][:4])
            return pl.DataFrame({'date':[date(year,1,5),date(year,1,6)],
                'close':[100.,200. if year==2024 else 50.]})
    monkeypatch.setattr('quant_warehouse.Warehouse',Warehouse)
    previous=None
    for epoch in (1,2):
        directory=tmp_path/f'epoch_{epoch}';directory.mkdir()
        dates=[date(y,1,d) for y in (2024,2025,2026) for d in (5,6)]
        pl.DataFrame({'date':dates,'symbol':['A']*6,
            **{f'hits_{side}_return_{role}':[.1]*6 for side in ('long','short') for role in ('hub','authority')}}).write_csv(directory/'supervised_predictions.csv')
        reports=yearly_epoch_backtests(['python','train','--corpus',str(corpus)],directory,'2024-01-01','2026-09-09',previous)
        assert len(reports)==6
        longs=[r for r in reports if r['side']=='long']
        assert [r['period'] for r in longs]==['2024','2025','2026']
        assert longs[0]['capital_return']>0
        assert longs[1]['capital_return']<0 and longs[2]['capital_return']<0
        assert all(r['initial_cash']==100000 for r in reports)
        if previous:
            assert all(r['return_change_vs_previous_epoch']==0 for r in reports)
        previous=reports
    assert len(calls)==3
    assert calls[-1]['end']=='2026-09-09'
    assert 'period,start,end' in format_backtest_report(reports)


def test_monitor_drains_immutable_epochs_after_trainer_exits(tmp_path,monkeypatch):
    import json,sys,torch
    from scripts import monitor_multirate_epochs as monitor
    training=tmp_path/'training';snapshots=training/'epoch_checkpoints';snapshots.mkdir(parents=True)
    (training/'multirate_mtl_model.pt').touch()
    for epoch in (0,1):
        torch.save({'metrics':{'epoch':epoch,'batch':3,'epoch_complete':True}},snapshots/f'epoch_{epoch+1:04d}.pt')
    command=tmp_path/'command.json';command.write_text(json.dumps(['python','train','--output-dir',str(training),'--train-end-date','2024-01-01']))
    log=tmp_path/'train.log';log.write_text('')
    out=tmp_path/'evaluation';calls=[]
    def inference(invocation,**kwargs):
        directory=__import__('pathlib').Path(invocation[invocation.index('--output-dir')+1])
        (directory/'ntp_evaluation.json').write_text(json.dumps({'metrics':[]}))
        calls.append(directory.name)
        return __import__('types').SimpleNamespace(returncode=0)
    monkeypatch.setattr(monitor.subprocess,'run',inference)
    def exited(*args):raise ProcessLookupError
    monkeypatch.setattr(monitor.os,'kill',exited)
    monkeypatch.setattr(sys,'argv',['monitor','--command-file',str(command),'--training-log',str(log),
        '--output-dir',str(out),'--validation-start','2024-01-01','--validation-end','2024-12-31','--training-pid','123'])
    monitor.main()
    assert calls==['epoch_0001','epoch_0002']
    assert json.loads((out/'status.json').read_text())['completed_epoch']==2
