from datetime import date
import polars as pl
import pytest
from quant_orchestrator.platforms.backtesting_frameworks.anchored_hits_replay import ranked_day, update_book, replay_anchored_hits


def test_daily_average_tie_percentiles_and_strict_threshold():
    frame=pl.DataFrame({'symbol':list('ABCDE'), 'hits_long_return_hub':[1.,2.,2.,3.,4.],
                        'hits_long_return_authority':[5.,4.,3.,2.,1.]})
    ranks=ranked_day(frame,'long')
    assert ranks['hub'].to_list()==[.2,.5,.5,.8,1.]
    records={r['symbol']:r for r in ranks.iter_rows(named=True)}
    held,events=update_book({'A'},records,threshold=.8,top_k=20)
    assert held=={'E'}
    assert [(s,a) for s,a,_ in events]==[('A','exit'),('E','enter')]


@pytest.mark.parametrize('capacity',[1,20])
@pytest.mark.parametrize('side,expected',[('long',1100),('short',900)])
def test_next_session_close_does_not_earn_entry_day_return(tmp_path,side,expected,capacity):
    dates=[date(2024,1,d) for d in [5,8,9]]
    prices=pl.DataFrame({'date':dates,'symbol':['A']*3,'close':[100.,200.,220.]}).lazy()
    scores=pl.DataFrame({'date':dates,'symbol':['A']*3,
        f'hits_{side}_return_hub':[.01]*3,f'hits_{side}_return_authority':[.02]*3}).lazy()
    result=replay_anchored_hits(scores,prices,tmp_path/side,start='2024-01-05',end='2024-01-09',
        side=side,top_k=capacity,cost_bps=0,initial_cash=1000)
    assert result['allocation_slots']==1
    assert result['final_equity']==pytest.approx(expected)
    actions=pl.read_parquet(tmp_path/side/'action_tape.parquet')
    assert actions['date'][0]==dates[1]
    assert pl.read_parquet(tmp_path/side/'equity_curve.parquet')['equity'][1]==1000
