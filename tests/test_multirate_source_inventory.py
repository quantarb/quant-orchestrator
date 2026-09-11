from datetime import datetime
import json
import polars as pl
import pytest
from quant_orchestrator.research_tools.multirate_source_inventory import describe_source,snapshot_sources


def test_inventory_distinguishes_unknown_cadence_history_from_snapshot():
    historical=pl.DataFrame({'__index__':[datetime(1985,9,30),datetime(2025,9,30)],'fiscal_period':[float('nan')]*2,'ratio':[1.,2.]})
    r=describe_source(historical)
    assert r['numeric_fields']==['ratio']
    assert r['date_columns']['__index__']['pre_cutoff_rows']==1
    assert r['date_columns']['__index__']['unique_dates']==2
    assert describe_source(historical.tail(1))['date_columns']['__index__']['pre_cutoff_rows']==0


def test_snapshot_retains_each_source_and_retries_failed_reads(tmp_path):
    class Warehouse:
        fail=True
        calls=0
        def read_fundamentals(self,symbol,*,section,period,start,end):
            self.calls+=1
            if section=='ratios' and period is None:
                if self.fail:raise RuntimeError('temporary failure')
                return pl.DataFrame({'date':[datetime(2000,1,1)],'ratio':[2.]})
            return pl.DataFrame()
        def read_news(self,*args,**kwargs):return pl.DataFrame()
    warehouse=Warehouse()
    first=snapshot_sources(tmp_path,['A'],warehouse=warehouse)
    assert sum(r['status']=='error' for r in first)==1
    calls=warehouse.calls;warehouse.fail=False
    second=snapshot_sources(tmp_path,['A'],warehouse=warehouse)
    assert warehouse.calls==calls+1
    assert sum(r['status']=='observed' for r in second)==1
    assert pl.read_parquet(tmp_path/'A/ratios.parquet').height==1
    with pytest.raises(ValueError,match='configuration'):
        snapshot_sources(tmp_path,['B'],warehouse=warehouse)
