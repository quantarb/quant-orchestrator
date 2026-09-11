from datetime import datetime
import polars as pl
from quant_orchestrator.research_tools.multirate_supervision import StreamingSupervision


def test_activity_sizes_are_inputs_and_event_pairs_are_binary_targets():
    rows=[]
    for day,family,value in [(1,'holder_activity.buy',1000000.),(2,'holder_activity.reduce',20000.),(3,'fund_activity.etf_buy',.0008)]:
        rows.append(dict(symbol='A',date=datetime(2023,1,day),event_date=datetime(2023,1,day),target_family=family,
                         signal_value=value,**{f'text_{i}':None for i in range(7)}))
    store=StreamingSupervision(pl.DataFrame(rows).lazy(),cutoff=datetime(2024,1,1))
    first=store.get(('A',datetime(2023,1,1)))
    second=store.get(('A',datetime(2023,1,2)))
    assert first['holder_activity_buy']==1 and second['holder_activity_buy']==0
    assert first['holder_activity_reduce']==0 and second['holder_activity_reduce']==1
    assert 'fund_activity_etf_buy' in store.disabled_activity_tasks
    assert 'fund_activity_etf_buy' not in store.get(('A',datetime(2023,1,3)))


def test_post_cutoff_activity_does_not_enable_training_head():
    rows=[]
    for year,family in [(2023,'holder_activity.buy'),(2025,'holder_activity.reduce')]:
        rows.append(dict(symbol='A',date=datetime(year,1,1),event_date=datetime(year,1,1),target_family=family,
                         signal_value=100.,**{f'text_{i}':None for i in range(7)}))
    store=StreamingSupervision(pl.DataFrame(rows).lazy(),cutoff=datetime(2024,1,1))
    assert 'holder_activity_buy' in store.disabled_activity_tasks


def test_trade_targets_use_transaction_date_and_only_actual_buy_sell_events():
    rows=[]
    for family in ['equity.ownership.government_trades', 'equity.ownership.insider_trading']:
        for day,buy,sell in [(2,1.,0.),(3,0.,1.),(4,0.,0.)]:
            rows.append(dict(symbol='A', date=datetime(2023,2,1), event_date=datetime(2023,1,day),
                target_family=family,signal_value=buy,text_0=sell,**{f'text_{i}':None for i in range(1,7)}))
    events=pl.DataFrame(rows)
    store=StreamingSupervision(events.lazy(),cutoff=datetime(2024,1,1))
    for prefix in ['government','insider']:
        assert store.get(('A',datetime(2023,1,2)))[prefix+'_is_buy']==1
        assert store.get(('A',datetime(2023,1,3)))[prefix+'_is_buy']==0
        assert store.get(('A',datetime(2023,1,3)))[prefix+'_is_sell']==1
    assert store.get(('A',datetime(2023,1,4)))=={}
    assert store.get(('A',datetime(2023,2,1)))=={}
    assert events['date'].min()==datetime(2023,2,1)
