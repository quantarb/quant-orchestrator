"""Compare an Oracle gate using completed epoch scores and identical frozen prices."""
import json
from pathlib import Path
import polars as pl
from quant_orchestrator.platforms.backtesting_frameworks.existing_multirate_backtest import run_existing_multirate_backtest


def compare_epoch_oracle_gate(directory: Path):
    directory=Path(directory)
    baseline=json.loads((directory/'backtest_metrics.json').read_text())
    comparisons=[]
    for period in sorted({r['period'] for r in baseline}):
        original=[r for r in baseline if r['period']==period]
        start,end=original[0]['start'],original[0]['end']
        symbols=json.loads((directory/period/'backtest_universe.json').read_text())['included']
        prices=[directory.parent/'backtest_prices'/f'{start}_{end}'/f'{symbol}.parquet' for symbol in symbols]
        if not all(p.exists() for p in prices):raise ValueError('Missing frozen baseline price snapshot')
        output=directory/'oracle_gate'/period;output.parent.mkdir(exist_ok=True)
        if (output/'results.json').exists():
            gated=json.loads((output/'results.json').read_text())
        else:
            scores=pl.scan_csv(directory/'supervised_predictions.csv',try_parse_dates=True).filter(
                pl.col('symbol').is_in(symbols) & pl.col('date').cast(pl.Date).is_between(pl.lit(start).str.to_date(),pl.lit(end).str.to_date()))
            gated=run_existing_multirate_backtest(scores,pl.scan_parquet(prices),output,oracle_gate=True)
        for result in gated:
            before=next(r for r in original if r['side']==result['side'])
            comparisons.append(dict(period=period,side=result['side'],
                baseline_return=before['capital_return'],gated_return=result['capital_return'],
                return_difference=result['capital_return']-before['capital_return'],
                baseline_sharpe=before['sharpe'],gated_sharpe=result['sharpe'],
                baseline_drawdown=before['max_drawdown'],gated_drawdown=result['max_drawdown'],
                baseline_exposure=before['mean_gross_exposure'],gated_exposure=result['mean_gross_exposure'],
                baseline_entries=before['entries'],gated_entries=result['entries']))
    temporary=directory/'oracle_gate_comparison.tmp'
    temporary.write_text(json.dumps(comparisons,indent=2));temporary.replace(directory/'oracle_gate_comparison.json')
    return comparisons
