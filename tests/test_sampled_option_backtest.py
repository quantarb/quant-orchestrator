from datetime import datetime

import pandas as pd
import polars as pl
import pytest

from quant_orchestrator.platforms.backtesting_frameworks.sampled_option_backtest import run_sampled_option_backtest


def test_sampled_contract_replay_uses_prior_equity_signal_and_pays_spread(tmp_path):
    dates=[datetime(2024,1,d) for d in (2,3,4)]
    predictions=[];quotes=[];equities=[]
    for date in dates:
        for underlying in ('A','B'):
            equities.append(dict(symbol=underlying,date=date,
                hits_long_return_hub=2. if underlying=='A' else 1.,hits_long_return_authority=2. if underlying=='A' else 1.,
                hits_short_return_hub=1. if underlying=='A' else 2.,hits_short_return_authority=1. if underlying=='A' else 2.))
            for right in ('CALL','PUT'):
                symbol=f'{underlying}250101{right[0]}00100000'
                predictions.append(dict(symbol=symbol,date=date,underlying_symbol=underlying,option_type=right.lower(),expiration=datetime(2025,1,1),settlement=datetime(2024,12,31)))
                quotes.append(dict(symbol=symbol,date=date,open=11.,close=11.,low=10.,high=12.,volume=10.))
    results=run_sampled_option_backtest(pl.DataFrame(predictions),pl.DataFrame(quotes),tmp_path/'bt',equity_predictions=pl.DataFrame(equities))
    for side in ('call','put'):
        actions=pl.read_parquet(tmp_path/'bt'/f'{side}_actions.parquet')
        assert actions['date'].min()==dates[1]
        assert actions.filter(pl.col('action')=='enter_long').height==1
    for result in results:
        assert result['capital_return']==pytest.approx(-(.5/11+.5*5.5/10000))
        assert result['max_drawdown']==pytest.approx(result['capital_return'])
        assert result['stale_valuation_position_days']==0


def test_missing_contract_quote_defers_entry_instead_of_inventing_execution(tmp_path):
    dates=[datetime(2024,1,d) for d in (2,3,4)]
    eq=pl.DataFrame([dict(symbol='A',date=d,hits_long_return_hub=1.,hits_long_return_authority=1.,
                         hits_short_return_hub=0.,hits_short_return_authority=0.) for d in dates])
    predictions=pl.DataFrame([dict(symbol=f'A250101{r[0]}00100000',underlying_symbol='A',date=d,option_type=r.lower(),expiration=datetime(2025,1,1),settlement=datetime(2024,12,31))
                             for r in ('CALL','PUT') for d in dates])
    prices=pl.DataFrame([dict(symbol=f'A250101{r[0]}00100000',date=d,open=11.,close=11.,low=10.,high=12.,volume=1.)
                        for r in ('CALL','PUT') for d in (dates[0],dates[2])])
    run_sampled_option_backtest(predictions,prices,tmp_path/'bt',equity_predictions=eq)
    actions=pl.read_parquet(tmp_path/'bt'/'call_actions.parquet')
    assert actions['date'].min()==dates[2]


def test_empty_fixed_universe_is_reported_as_cash_not_a_missing_report(tmp_path):
    result=run_sampled_option_backtest(pl.DataFrame(),pl.DataFrame(),tmp_path/'empty',equity_predictions=pl.DataFrame())
    assert len(result)==2
    assert all(r['status']=='no_sampled_contracts' and r['capital_return']==0 for r in result)


def test_only_one_fixed_contract_per_underlying_in_each_book(tmp_path):
    dates=[datetime(2024,1,d) for d in (2,3,4)]
    equities=pl.DataFrame([dict(symbol=s,date=d,hits_long_return_hub=2. if s=='A' else 1.,
        hits_long_return_authority=2. if s=='A' else 1.,hits_short_return_hub=1. if s=='A' else 2.,
        hits_short_return_authority=1. if s=='A' else 2.) for s in ('A','B') for d in dates])
    rows=[dict(symbol=f'{s}250101{r[0].upper()}{i:08}',underlying_symbol=s,option_type=r,
        expiration=datetime(2025,1,1),settlement=datetime(2024,12,31),date=d)
        for s in ('A','B') for r in ('call','put') for i in range(5) for d in dates]
    predictions=pl.DataFrame(rows)
    prices=predictions.select('symbol','date').with_columns(pl.lit(11.).alias('open'),pl.lit(11.).alias('close'),
        pl.lit(10.).alias('low'),pl.lit(12.).alias('high'),pl.lit(1.).alias('volume'))
    results=run_sampled_option_backtest(predictions,prices,tmp_path/'first',equity_predictions=equities)
    selected=pl.read_parquet(tmp_path/'first/backtest_contracts.parquet')
    assert selected.height==4
    assert selected.group_by('underlying_symbol','option_type').len()['len'].to_list()==[1]*4
    assert all(r['contract_count']==2 and r['contracts_per_underlying_per_book']==1 for r in results)
    for right in ('call','put'):
        weights=pd.read_parquet(tmp_path/'first'/f'{right}_weights.parquet')
        assert set(weights.columns)==set(selected.filter(pl.col('option_type')==right)['symbol'])
    run_sampled_option_backtest(predictions.reverse(),prices.reverse(),tmp_path/'second',equity_predictions=equities)
    assert set(selected['symbol'])==set(pl.read_parquet(tmp_path/'second/backtest_contracts.parquet')['symbol'])
