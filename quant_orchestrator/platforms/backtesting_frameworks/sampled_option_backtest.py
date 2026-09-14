"""Replay equity signals on fixed sampled option contracts through the shared planner."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from .shared_book import build_shared_book_weights, run_shared_book_backtest
from quant_orchestrator.research_tools.sampled_options import sample_options_per_side


def run_sampled_option_backtest(predictions, prices, output, *, equity_predictions, initial_cash=100000.):
    """Long calls and long puts in separate books; executable quotes gate orders.

    Select one fixed call and one fixed put per underlying from the training sample.
    Equity signals decide direction. Known contract expirations force settlement.
    Missing contract quotes defer execution; they never change contract membership.
    The shared planner preserves capacity while a position awaits an exit quote.
    """
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    if predictions.is_empty() and prices.is_empty():
        results=[dict(side=side, status='no_sampled_contracts', contract_count=0,
            capital_return=0., initial_cash=initial_cash, final_equity=initial_cash,
            hindsight_selection=True, fixed_universe=True) for side in ('long_calls','long_puts')]
        (output/'results.json').write_text(json.dumps(results,indent=2))
        return results
    policy=Path(__file__).resolve().parents[3].parent/'optimal_trader/scripts/multirate_transformer/trading_policy.py'
    spec=importlib.util.spec_from_file_location('sampled_option_equity_policy',policy)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    raw=equity_predictions.select('symbol','date',*[pl.col(f'hits_{side}_return_{role}').alias(f'{side}_{role}')
        for side in ('long','short') for role in ('hub','authority')]).to_pandas()
    equity_scores=module.build_legacy_compatible_scores(raw)
    dates=pd.DatetimeIndex(sorted(equity_scores.date.unique()))
    # Shift decisions onto the next equity trading session before matching
    # actual contract quotes. No same-close execution of EOD predictions.
    next_date=dict(zip(dates[:-1],dates[1:]))
    equity_scores['date']=equity_scores['date'].map(next_date)
    equity_scores=equity_scores.dropna(subset=['date']).rename(columns={'symbol':'underlying_symbol'})
    metadata=predictions.select('symbol','underlying_symbol','option_type','expiration','settlement').unique()
    # One real contract per underlying/right/year. Selection is deterministic,
    # independent of row order, and never replaced after expiry or missing quotes.
    metadata=sample_options_per_side(metadata.rename({'symbol':'contract_symbol'}),count=1,seed=0).rename({'contract_symbol':'symbol'})
    metadata.write_parquet(output/'backtest_contracts.parquet')
    metadata=metadata.to_pandas()
    quotes=prices.to_pandas();quotes['date']=pd.to_datetime(quotes['date'])
    quotes=quotes.merge(metadata,on='symbol',how='inner',validate='many_to_one')
    quotes['settlement']=pd.to_datetime(quotes['settlement'])
    for symbol,group in quotes.groupby('symbol'):
        settlement=group.settlement.iloc[0]
        if settlement<=dates[-1] and not group.date.eq(settlement).any():
            raise ValueError(f'{symbol}: missing expiration-session contract value; cannot settle this backtest')
    results=[]
    for right in ('CALL','PUT'):
        q=quotes.loc[quotes.option_type.eq(right.lower())].copy()
        symbols=sorted(q.symbol.unique())
        if not symbols:
            results.append(dict(side='long_calls' if right=='CALL' else 'long_puts', status='no_sampled_contracts',
                contract_count=0, capital_return=0., initial_cash=initial_cash, final_equity=initial_cash, hindsight_selection=True))
            continue
        frame=q.merge(equity_scores,on=['underlying_symbol','date'],how='left')
        if right=='PUT':
            for a,b in [('long_score','short_score'),('long_exit_score','short_exit_score'),('long_agree_count','short_agree_count')]:
                frame[a],frame[b]=frame[b].copy(),frame[a].copy()
        expired=frame.date>=frame.settlement
        frame.loc[expired,'long_agree_count']=0
        frame.loc[expired,'long_score']=0.
        frame.loc[expired,'model_count']=1
        frame=frame.loc[frame.model_count.notna()]
        capacity=min(20,len(symbols))
        weights,actions=build_shared_book_weights(frame,symbols,dates,top_k=capacity,variant='long_only',entry_threshold=.5,exit_threshold=.5)
        def panel(name):return q.pivot(index='date',columns='symbol',values=name).reindex(index=dates,columns=symbols)
        mid=panel('close');bid=panel('low');ask=panel('high')
        marked=mid.ffill()
        returns=marked.pct_change(fill_method=None).shift(-1).replace([np.inf,-np.inf],np.nan).fillna(0.)
        net,_,turnover=run_shared_book_backtest(weights,returns,cost_bps=5.5,capital_base=initial_cash)
        changes=weights.diff().fillna(weights)
        buying=changes.clip(lower=0);selling=(-changes.clip(upper=0))
        spread=((buying*(ask-mid)/mid)+(selling*(mid-bid)/mid)).replace([np.inf,-np.inf],np.nan).fillna(0.).sum(axis=1)
        net=net-spread
        equity=initial_cash*(1+net).cumprod()
        if not np.isfinite(equity).all() or (equity<0).any():raise ValueError('Invalid sampled-contract portfolio equity')
        # Include initial cash in the drawdown high-water mark.
        peak=equity.cummax().clip(lower=initial_cash)
        stale=int(((weights>0)&mid.isna()).sum().sum())
        outstanding=int((weights.iloc[-1]>0).sum())
        result=dict(hindsight_selection=True, fixed_universe=True, contracts_per_underlying_per_book=1, selection_seed=0, side='long_calls' if right=='CALL' else 'long_puts',capital_return=float(equity.iloc[-1]/initial_cash-1),
            sharpe=float(net.mean()/net.std()*np.sqrt(252)) if net.std()>0 else 0.,max_drawdown=float((equity/peak-1).min()),
            entries=int(actions.action.eq('enter_long').sum()),exits=int(actions.action.eq('exit_long').sum()),
            initial_cash=initial_cash,final_equity=float(equity.iloc[-1]),contract_count=len(symbols),capacity=capacity,
            stale_valuation_position_days=stale,open_positions_at_period_end=outstanding,
            execution='prior-session equity signals; contract bid/ask spreads plus 5.5 bps per turnover',
            valuation='midpoint; last observed mark on missing-quote dates; intrinsic settlement when present')
        results.append(result)
        stem=right.lower()
        actions.to_parquet(output/f'{stem}_actions.parquet',index=False)
        weights.to_parquet(output/f'{stem}_weights.parquet')
        pd.DataFrame(dict(equity=equity,net_return=net,spread_cost=spread,turnover=turnover)).to_parquet(output/f'{stem}_equity.parquet')
    (output/'results.json').write_text(json.dumps(results,indent=2))
    return results
