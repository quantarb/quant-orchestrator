from datetime import datetime, timedelta

import polars as pl
import pytest

from quant_orchestrator.research_tools.option_training import frozen_option_paths, frozen_ten_option_members


def test_frozen_baskets_keep_strikes_and_reject_missing_constituents():
    first = datetime(2024, 1, 2)
    chain = pl.DataFrame([
        dict(snapshot_date=day, expiration=first+timedelta(days=dte),
             contract_symbol=f'{right}_{dte}_{strike}', underlying_symbol='A', option_type=right,
             bid=float(strike), ask=float(strike+2), volume=10)
        for day in (first, first+timedelta(days=1))
        for right in ('call','put') for dte in (10,20,30,40,50,60,70)
        for strike in (1,3)
    ])
    members = frozen_ten_option_members(chain, first_session=first)
    assert members['document_symbol'].n_unique() == 10
    assert members.height == 20
    assert members['weight'].to_list() == [0.5]*20
    paths = frozen_option_paths(chain, members)
    assert paths.height == 20
    assert paths['close'].to_list() == [3.0]*20
    missing = chain.filter(~((pl.col('contract_symbol')=='call_10_1') & (pl.col('snapshot_date')>first)))
    incomplete = frozen_option_paths(missing, members)
    assert incomplete.height == 19
    assert incomplete.filter(pl.col('date')==first).height == 10
    with pytest.raises(ValueError, match='five required'):
        frozen_ten_option_members(chain, first_session=first-timedelta(days=1))
