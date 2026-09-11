"""Adapt model outputs to the existing transformer policy and shared-book engine.

No entry, exit, sizing or return loop is implemented here. A bounded annual
Polars panel crosses into the original engine's pandas interface.
"""
from pathlib import Path
import importlib.util
import hashlib
import json
import polars as pl
from . import shared_book


def apply_oracle_gate(scores):
    """Additional predicted-Oracle entry permission and hold/exit vetoes."""
    columns=['oracle_is_buy','oracle_is_sell','oracle_is_short','oracle_is_cover']
    frame=pl.from_pandas(scores)
    if not frame.select(pl.all_horizontal([
        pl.col(c).is_not_null() & pl.col(c).is_finite() & pl.col(c).is_between(0,1)
        for c in columns]).all()).item():
        raise ValueError('Oracle gate requires finite predicted probabilities in [0,1]')
    buy,sell,short,cover=[pl.col(c) for c in columns]
    return frame.with_columns(
        ((pl.col('long_agree_count')==1) & (buy>short) & (sell<.5)).cast(pl.Int64).alias('long_agree_count'),
        ((pl.col('short_agree_count')==1) & (short>buy) & (cover<.5)).cast(pl.Int64).alias('short_agree_count'),
        pl.when(buy>=.5).then(pl.col('long_score')).otherwise(0.).alias('long_score'),
        pl.when(short>=.5).then(pl.col('short_score')).otherwise(0.).alias('short_score'),
    ).to_pandas()


def run_existing_multirate_backtest(predictions, prices, output, *, initial_cash=100000., oracle_gate=False):
    source = Path(__file__).resolve().parents[3].parent/'optimal_trader/scripts/multirate_transformer/trading_policy.py'
    spec = importlib.util.spec_from_file_location('existing_transformer_trading_policy', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    oracle_columns=['oracle_is_buy','oracle_is_sell','oracle_is_short','oracle_is_cover'] if oracle_gate else []
    raw = predictions.select('symbol','date',*oracle_columns,*[
        pl.col(f'hits_{side}_return_{role}').alias(f'{side}_{role}')
        for side in ('long','short') for role in ('hub','authority')
    ]).sort('date','symbol').collect(engine='streaming')
    scores = module.build_legacy_compatible_scores(raw.to_pandas())
    if oracle_gate:
        scores = apply_oracle_gate(scores)
    close = prices.collect(engine='streaming').to_pandas().pivot(index='date',columns='symbol',values='close').sort_index().ffill()
    capacity = min(20,len(close.columns))
    next_returns = close.pct_change().shift(-1)
    summary,actions,weights = shared_book.run_shared_book_framework_comparison(
        scores=scores,next_returns=next_returns,symbols=tuple(close.columns),dates=close.index,
        variants=('long_only','short_only'),top_k_values=(capacity,),entry_threshold=.5,
        exit_threshold=.5,cost_models={'family_common':shared_book.SharedBookCostModel(.5,5.)},capital_base=initial_cash)
    pl.from_pandas(scores).write_parquet(output/'strategy_scores.parquet')
    pl.from_pandas(actions).write_parquet(output/'action_tape.parquet')
    for (variant,_),frame in weights.items():
        returns,equity,turnover = shared_book.run_shared_book_backtest(frame,next_returns,cost_bps=5.5,capital_base=initial_cash)
        pl.from_pandas(frame.reset_index()).write_parquet(output/f'{variant}_weights.parquet')
        pl.from_pandas(equity.to_frame().assign(net_return=returns,turnover=turnover).reset_index()).write_parquet(output/f'{variant}_equity.parquet')
    reports = []
    for row in summary.to_dict('records'):
        events = actions.loc[actions.variant.eq(row['variant']),'action']
        reports.append({**row,'side':row['variant'].removesuffix('_only'),'oracle_gate':oracle_gate,
            'capital_return':row['final_equity']/initial_cash-1,
            'initial_cash':initial_cash,'entries':int(events.str.startswith('enter').sum()),
            'exits':int(events.str.startswith('exit').sum()),'mean_gross_exposure':row['avg_gross_exposure']})
    (output/'results.json').write_text(json.dumps(reports,indent=2))
    (output/'provenance.json').write_text(json.dumps(dict(
        score_policy=str(source),engine=str(Path(shared_book.__file__).resolve()),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        engine_sha256=hashlib.sha256(Path(shared_book.__file__).read_bytes()).hexdigest(),
        timing='original signal-date weights times next close-to-close returns',
        total_return='original metric excludes the first recorded daily return; capital_return uses initial_cash',
        price_adjustment='splits_and_dividends',capacity=capacity,oracle_gate=oracle_gate,
        oracle_gate_rules='Additional to HITS: enter with side probability >= .5 and above opposite; hold only while above opposite and exit-action probability < .5; no entry during exit veto' if oracle_gate else None),indent=2))
    return reports
