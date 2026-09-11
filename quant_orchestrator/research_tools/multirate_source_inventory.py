"""Snapshot broad warehouse inputs one issuer/source at a time using Polars.

An inventory is not a training corpus: missing histories and ambiguous dates
remain explicit until the family adapters can consume the corresponding source.
"""
from datetime import datetime
import json
from pathlib import Path
import polars as pl

PERIOD_SECTIONS = ('income','balance','cash','ratios','metrics','income_growth',
                   'balance_growth','cash_growth','estimates_historical')
EVENT_SECTIONS = ('employee_count','ownership_institutional','ratings_historical',
                  'esg_score','historical_market_cap','ownership_government_trades',
                  'ownership_insider_trading','estimates_price_target','dividends','historical_splits')


def describe_source(frame: pl.DataFrame, *, cutoff='2024-01-01'):
    numeric=[c for c,t in frame.schema.items() if t.is_numeric() and
             frame[c].cast(pl.Float64).is_finite().any()]
    dates={}
    for column,dtype in frame.schema.items():
        if not dtype.is_temporal():
            continue
        values=frame[column].cast(pl.Datetime('ns')).drop_nulls()
        dates[column]={'first':str(values.min()),'last':str(values.max()),
                       'pre_cutoff_rows':int((values<datetime.fromisoformat(cutoff)).sum()),
                       'unique_dates':values.n_unique()}
    return {'rows':frame.height,'numeric_fields':numeric,'date_columns':dates,
            'schema':{k:str(v) for k,v in frame.schema.items()},
            'status':'observed' if frame.height else 'missing'}


def snapshot_sources(output, symbols, *, warehouse, start='1900-01-01',end='2026-09-09'):
    """Resume immutable per-source snapshots; never collect a multi-issuer panel."""
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    config={'symbols':sorted(set(symbols)),'start':start,'end':end}
    manifest=output/'configuration.json'
    if manifest.exists() and json.loads(manifest.read_text())!=config:
        raise ValueError('Source snapshot configuration differs from existing inventory')
    manifest.write_text(json.dumps(config,indent=2))
    requests=[(s,p) for s in PERIOD_SECTIONS for p in (None,'annual','quarter')]
    requests += [(s,None) for s in EVENT_SECTIONS]+[('company_news',None)]
    records=[]
    for index,symbol in enumerate(config['symbols']):
        directory=output/symbol;directory.mkdir(exist_ok=True)
        for section,period in requests:
            name=section+('__'+period if period else '')
            report=directory/f'{name}.json';path=directory/f'{name}.parquet'
            if report.exists() and json.loads(report.read_text())['status']!='error':
                row=json.loads(report.read_text())
                if row['status']=='observed' and not path.exists():
                    raise ValueError(f'Missing recorded snapshot {path}')
            else:
                try:
                    frame=(warehouse.read_news(symbol,provider='fmp',start=datetime.fromisoformat(start),end=datetime.fromisoformat(end))
                           if section=='company_news' else warehouse.read_fundamentals(symbol,section=section,period=period,start=start,end=end))
                    if not isinstance(frame,pl.DataFrame):
                        raise TypeError('Source reader must return Polars')
                    row={'symbol':symbol,'section':section,'period':period,**describe_source(frame)}
                    if frame.height:
                        tmp=path.with_suffix('.tmp');frame.write_parquet(tmp);tmp.replace(path)
                except Exception as exc:
                    row={'symbol':symbol,'section':section,'period':period,'status':'error',
                         'error':f'{type(exc).__name__}: {exc}'}
                temporary=report.with_suffix('.tmp');temporary.write_text(json.dumps(row,indent=2));temporary.replace(report)
            records.append(row)
        state={'stage':'snapshotting','completed_issuers':index+1,'total_issuers':len(config['symbols']),'last_symbol':symbol}
        (output/'status.json').write_text(json.dumps(state))
        print(json.dumps(state),flush=True)
    (output/'inventory.json').write_text(json.dumps(records,indent=2))
    (output/'status.json').write_text(json.dumps({'stage':'complete','issuers':len(config['symbols']),
        'observed_sources':sum(r['status']=='observed' for r in records),
        'missing_sources':sum(r['status']=='missing' for r in records),
        'errors':sum(r['status']=='error' for r in records)}))
    return records
