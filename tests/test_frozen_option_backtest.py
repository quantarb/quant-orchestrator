from datetime import datetime

import polars as pl
import pytest

from quant_orchestrator.platforms.backtesting_frameworks.frozen_option_backtest import run_frozen_option_backtest


def test_frozen_basket_replay_uses_prior_equity_signal_and_pays_spread(tmp_path):
    dates=[datetime(2024,1,d) for d in (2,3,4)]
    predictions=[];quotes=[];equities=[]
    for date in dates:
        for underlying in ('A','B'):
            equities.append(dict(symbol=underlying,date=date,
                hits_long_return_hub=2. if underlying=='A' else 1.,hits_long_return_authority=2. if underlying=='A' else 1.,
                hits_short_return_hub=1. if underlying=='A' else 2.,hits_short_return_authority=1. if underlying=='A' else 2.))
            for right in ('CALL','PUT'):
                symbol=f'OPT_{underlying}_2024_{right}_DTE_365'
                predictions.append(dict(symbol=symbol,date=date,underlying_symbol=underlying))
                quotes.append(dict(symbol=symbol,date=date,open=11.,close=11.,low=10.,high=12.,volume=10.))
    results=run_frozen_option_backtest(pl.DataFrame(predictions),pl.DataFrame(quotes),tmp_path/'bt',equity_predictions=pl.DataFrame(equities))
    for side in ('call','put'):
        actions=pl.read_parquet(tmp_path/'bt'/f'{side}_actions.parquet')
        assert actions['date'].min()==dates[1]
        assert actions.filter(pl.col('action')=='enter_long').height==1
    for result in results:
        assert result['capital_return']==pytest.approx(-(.5/11+.5*5.5/10000))
        assert result['max_drawdown']==pytest.approx(result['capital_return'])
        assert result['stale_valuation_position_days']==0


def test_missing_basket_quote_defers_entry_instead_of_inventing_execution(tmp_path):
    dates=[datetime(2024,1,d) for d in (2,3,4)]
    eq=pl.DataFrame([dict(symbol='A',date=d,hits_long_return_hub=1.,hits_long_return_authority=1.,
                         hits_short_return_hub=0.,hits_short_return_authority=0.) for d in dates])
    predictions=pl.DataFrame([dict(symbol=f'OPT_A_2024_{r}_DTE_365',underlying_symbol='A',date=d)
                             for r in ('CALL','PUT') for d in dates])
    prices=pl.DataFrame([dict(symbol=f'OPT_A_2024_{r}_DTE_365',date=d,open=11.,close=11.,low=10.,high=12.,volume=1.)
                        for r in ('CALL','PUT') for d in (dates[0],dates[2])])
    run_frozen_option_backtest(predictions,prices,tmp_path/'bt',equity_predictions=eq)
    actions=pl.read_parquet(tmp_path/'bt'/'call_actions.parquet')
    assert actions['date'].min()==dates[2]
