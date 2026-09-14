from datetime import datetime
from types import SimpleNamespace
import polars as pl
import exchange_calendars as xcals

from quant_orchestrator.research_tools.sampled_options import select_contracts, contract_candidates


def test_filters_then_separate_right_medians_then_fixed_balanced_sample():
    rows=[]
    for symbol in ('A','B'):
        for right in ('call','put'):
            for i in range(20):
                rows.append(dict(underlying_symbol=symbol,contract_symbol=f'{symbol}{right}{i:02}',option_type=right,
                    moneyness=1.,profit_pct=float(i+1)*(100 if right=='call' else 1),valid_quote_days=20,quote_coverage=.8))
    audit=pl.DataFrame(rows)
    picked=select_contracts(audit)
    assert picked.height==20
    assert picked.group_by('underlying_symbol','option_type').len()['len'].to_list()==[5]*4
    assert picked.equals(select_contracts(audit.reverse()))
    assert picked.filter(pl.col('option_type')=='call')['profit_pct'].min()>=1050
    assert picked.filter(pl.col('option_type')=='put')['profit_pct'].min()>=10.5
    assert select_contracts(audit.with_columns(pl.lit(19).alias('valid_quote_days'))).is_empty()
    assert select_contracts(audit.with_columns(pl.lit(.79).alias('quote_coverage'))).is_empty()
    assert select_contracts(audit.with_columns(pl.lit(0.).alias('moneyness'))).is_empty()
    assert select_contracts(audit.with_columns(pl.lit(-1.).alias('profit_pct'))).is_empty()
    small=audit.filter((pl.col('underlying_symbol')=='A')&(pl.col('option_type')=='put')).head(4)
    assert select_contracts(small).height==2  # Never fill absent call slots with puts.


def test_native_paths_settlement_and_last_invalid_bid_not_previous_bid():
    cal=xcals.get_calendar('XNYS',start='2023-12-01',end='2025-12-31')
    days=cal.sessions_in_range('2024-01-02','2024-02-16').tz_localize(None).to_pydatetime().tolist()
    rows=[]
    for day in days:
        for name,strike in [('good',90.),('invalid_last',95.),('nextyear',80.)]:
            rows.append(dict(snapshot_date=day,underlying_symbol='A',contract_symbol=name,option_type='call',
                expiration=datetime(2025 if name=='nextyear' else 2024,2,16),strike=strike,
                bid=None if name=='invalid_last' and day==days[-1] else 3.,ask=4. if day==days[0] else 5.,
                volume=1.,underlying_price=110.))
    # Positive bid history and positive first-ask/last-bid return for good.
    q=pl.DataFrame(rows).with_columns(pl.when(pl.col('snapshot_date')==days[0]).then(1.).otherwise(pl.col('ask')).alias('ask'),
        pl.when(pl.col('snapshot_date')==days[0]).then(.5).otherwise(pl.col('bid')).alias('bid'))
    read=lambda a,b:q.filter(pl.col('snapshot_date').is_between(a,b))
    w=SimpleNamespace(read_fundamentals=lambda *a,**k:pl.DataFrame())
    audit,paths,status=contract_candidates(w,'A',2024,datetime(2024,12,31),read,cal)
    assert status=='audited'
    assert set(audit['contract_symbol'])=={'good','invalid_last'}
    assert audit.filter(pl.col('contract_symbol')=='invalid_last')['profit_pct'][0] is None
    good=audit.filter(pl.col('contract_symbol')=='good').row(0,named=True)
    assert good['profit_pct']==200 and good['quote_coverage']==1 and good['moneyness']==20
    assert paths.filter((pl.col('symbol')=='good')&(pl.col('date')==days[-1]))['close'][0]==20
    assert paths.filter((pl.col('symbol')=='good')&(pl.col('date')==days[1]))['close'][0]==4
    assert select_contracts(audit)['contract_symbol'].to_list()==['good']
