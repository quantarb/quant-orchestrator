from datetime import datetime,timedelta
import polars as pl
import pytest
from quant_orchestrator.platforms.backtesting_frameworks.existing_multirate_backtest import apply_oracle_gate,run_existing_multirate_backtest


def test_oracle_gate_separates_entry_threshold_from_hold_and_exit():
    scores=pl.DataFrame({'long_score':[.9]*4,'short_score':[.8]*4,
        'long_agree_count':[1]*4,'short_agree_count':[0]*4,
        'oracle_is_buy':[.8,.4,.8,.3],'oracle_is_short':[.1,.1,.1,.7],
        'oracle_is_sell':[.1,.1,.6,.1],'oracle_is_cover':[.1]*4}).to_pandas()
    result=apply_oracle_gate(scores)
    assert result.long_score.tolist()==[.9,0.,.9,0.]
    assert result.long_agree_count.tolist()==[1,1,0,0]
    scores.loc[0,'oracle_is_buy']=float('nan')
    with pytest.raises(ValueError,match='finite'):apply_oracle_gate(scores)


def test_gate_can_leave_cash_without_reranking_or_changing_capacity(tmp_path):
    dates=[datetime(2024,1,2)+timedelta(days=i) for i in range(3)]
    rows=[]
    for date in dates:
        for symbol,hub in [('A',.9),('B',.1)]:
            rows.append(dict(date=date,symbol=symbol,hits_long_return_hub=hub,
                hits_short_return_hub=hub,hits_long_return_authority=.1,hits_short_return_authority=.1,
                oracle_is_buy=.3,oracle_is_short=.2,oracle_is_sell=.1,oracle_is_cover=.1))
    prices=pl.DataFrame([dict(date=d,symbol=s,close=100.+i*10) for i,d in enumerate(dates) for s in ['A','B']]).lazy()
    predictions=pl.DataFrame(rows).lazy()
    baseline=run_existing_multirate_backtest(predictions,prices,tmp_path/'baseline')
    gated=run_existing_multirate_backtest(predictions,prices,tmp_path/'gated',oracle_gate=True)
    assert baseline[0]['entries']>0
    assert all(r['entries']==0 and r['capital_return']==0 and r['mean_gross_exposure']==0 for r in gated)
    assert all(r['top_k']==2 for r in gated)
    directional=run_existing_multirate_backtest(predictions,prices,tmp_path/'directional',
        oracle_gate=True,oracle_gate_mode='directional')
    assert directional[0]['capital_return']==baseline[0]['capital_return']
    assert directional[0]['entries']==baseline[0]['entries']
    assert directional[0]['oracle_gate_mode']=='directional'


def test_short_gate_mirrors_entry_hold_and_cover_veto():
    scores=pl.DataFrame({'long_score':[.8]*4,'short_score':[.9]*4,
        'long_agree_count':[0]*4,'short_agree_count':[1]*4,
        'oracle_is_buy':[.1,.1,.1,.7],'oracle_is_short':[.8,.4,.8,.3],
        'oracle_is_sell':[.1]*4,'oracle_is_cover':[.1,.1,.6,.1]}).to_pandas()
    result=apply_oracle_gate(scores)
    assert result.short_score.tolist()==[.9,0.,.9,0.]
    assert result.short_agree_count.tolist()==[1,1,0,0]


def test_directional_gate_uses_only_relative_buy_short_and_preserves_ranks():
    # Low probabilities still permit entry; ties abstain; original HITS veto remains.
    scores=pl.DataFrame({'long_score':[.9]*4,'short_score':[.8]*4,
        'long_agree_count':[1,1,1,0],'short_agree_count':[1,1,1,0],
        'oracle_is_buy':[.3,.1,.2,.3],'oracle_is_short':[.1,.3,.2,.1]}).to_pandas()
    result=apply_oracle_gate(scores,mode='directional')
    assert result.long_agree_count.tolist()==[1,0,0,0]
    assert result.short_agree_count.tolist()==[0,1,0,0]
    assert result.long_score.tolist()==scores.long_score.tolist()
    assert result.short_score.tolist()==scores.short_score.tolist()
    # Sell/cover predictions have no permission to veto directional entries.
    scores['oracle_is_sell']=1.
    scores['oracle_is_cover']=1.
    assert apply_oracle_gate(scores,mode='directional').long_agree_count.tolist()==[1,0,0,0]
