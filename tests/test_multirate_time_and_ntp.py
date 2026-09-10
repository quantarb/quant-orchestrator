from datetime import datetime
import math
import polars as pl
import torch
from quant_orchestrator.platforms.ml_frameworks.torch.models.transformers.multirate.time_features import family_clock, DAY_NS
from quant_orchestrator.research_tools.ntp_evaluation import NTPPersistenceAudit
from quant_orchestrator.research_tools.multirate_objectives import reconstruction_targets
from quant_orchestrator.research_tools.multirate_supervision import input_event_families, StreamingSupervision


def test_clock_uses_actual_gaps_and_observation_age_without_future():
    dates = torch.tensor([[0, 3, 10, 40]]) * DAY_NS
    presence = torch.tensor([[[True, False], [True, True], [False, False], [True, True]]])
    clock = family_clock(dates, presence)
    assert clock[0, 1, 0, 0].item() == torch.tensor(math.log1p(3)).item()
    assert clock[0, 2, 0, 1].item() == torch.tensor(math.log1p(7)).item()
    assert not clock[0, 0, 1].any()
    changed = dates.clone(); changed[:, -1] = 400 * DAY_NS
    torch.testing.assert_close(clock[:, :3], family_clock(changed, presence)[:, :3])
    query = torch.tensor([[2, 5, 12]]) * DAY_NS
    age = family_clock(dates, presence, query)
    assert not age[0, 0, 1].any()
    assert age[0, 2, 0, 0].item() == torch.tensor(math.log1p(9)).item()
    torch.testing.assert_close(age, family_clock(changed, presence, query))


def test_clock_padding_and_empty_history_are_finite():
    dates = torch.tensor([[torch.iinfo(torch.long).min, 3 * DAY_NS, 8 * DAY_NS]])
    presence = torch.tensor([[[False], [True], [False]]])
    timing = family_clock(dates, presence)
    assert torch.isfinite(timing).all()
    assert not timing[:, 0].any()
    assert timing[0, -1, 0, 1].item() == torch.tensor(math.log1p(5)).item()
    assert not family_clock(dates, torch.zeros_like(presence)).any()


def test_persistence_is_matched_deduplicated_and_heldout(tmp_path):
    values = torch.tensor([[[1., float('nan')], [3., 5.], [6., 7.]]])
    dates = torch.tensor([[1, 2, 3]]) * DAY_NS
    padding = torch.zeros(1, 3, dtype=torch.bool)
    targets = reconstruction_targets(values, padding, dates, torch.zeros_like(values, dtype=torch.bool), (2,))
    predictions = {'next_daily_'+level: targets['next_'+level][0].clone() for level in ('token','subtoken')}
    audit = NTPPersistenceAudit(tmp_path/'pairs.sqlite', start_ns=3*DAY_NS)
    for _ in range(2):
        audit.update(['X'], 'daily', ['price'], (2,), values, padding, dates, predictions)
    report = audit.report(); audit.close()
    assert len(report['metrics']) == 2
    for row in report['metrics']:
        assert row['unique_pairs'] == 1 and row['values'] == 2
        assert row['model_mse'] == 0 and row['persistence_mse'] == 6.5
        assert row['skill'] == 1 and row['beats_persistence']


def test_supervised_outcomes_never_enter_inputs_even_after_available_date():
    families = ['equity.strategy.hits_graph', 'equity.strategy.oracle_trades', 'fmp.ownership_insider_trading']
    frame = pl.DataFrame({'symbol':['X']*3, 'date':[datetime(2023,12,31)]*3,
        'event_date':[datetime(2023,1,2)]*3, 'target_family':families,
        'signal_value':[2.,1.,9.], **{f'text_{i}':[1.,0.,3.] for i in range(7)}})
    labels = StreamingSupervision(frame.lazy()).get(('X',datetime(2023,1,2)))
    assert labels['oracle_is_buy'] == 1 and labels['hits_long_return_hub'] == 2
    inputs,names = input_event_families(frame.lazy(),families)
    assert names == ['fmp.ownership_insider_trading']
    assert inputs.collect()['signal_value'].to_list() == [9.]
    changed = frame.with_columns(pl.when(pl.col('target_family').str.starts_with('equity.strategy')).then(1e9).otherwise(pl.col('signal_value')).alias('signal_value'))
    altered,_ = input_event_families(changed.lazy(),families)
    assert inputs.collect().equals(altered.collect())
    empty,names = input_event_families(frame.head(2).lazy(),families[:2])
    assert empty.collect().is_empty() and names == ['__empty_sparse_family__']


def test_elapsed_time_changes_family_representation_and_receives_gradients():
    from quant_orchestrator.platforms.ml_frameworks.torch.models.transformers.multirate.auto_features import AutoFeatureEngineer
    torch.manual_seed(1)
    module = AutoFeatureEngineer(8, num_heads=2).eval()
    values = torch.randn(1, 3, 8)
    daily = torch.tensor([[0, 1, 2]]) * DAY_NS
    irregular = torch.tensor([[0, 30, 90]]) * DAY_NS
    first = module(values, mode='temporal', dates=daily)
    second = module(values, mode='temporal', dates=irregular)
    assert not torch.allclose(first[:, 1:], second[:, 1:])
    second[..., 0].sum().backward()
    assert module.elapsed_time[0].weight.grad.abs().sum() > 0


def test_zero_error_persistence_has_no_invented_skill(tmp_path):
    values = torch.ones(1, 3, 1)
    dates = torch.tensor([[1, 2, 3]]) * DAY_NS
    padding = torch.zeros(1, 3, dtype=torch.bool)
    audit = NTPPersistenceAudit(tmp_path/'zero.sqlite')
    audit.update(['X'], 'annual', ['constant'], (1,), values, padding, dates,
        {'next_annual_token': values.clone(), 'next_annual_subtoken': values.unsqueeze(-1)})
    for row in audit.report()['metrics']:
        assert row['persistence_mse'] == 0 and row['skill'] is None
        assert not row['beats_persistence']
    audit.close()


def test_empty_family_coverage_is_reported_instead_of_silently_omitted(tmp_path):
    values = torch.full((1, 3, 1), float('nan'))
    audit = NTPPersistenceAudit(tmp_path/'empty.sqlite')
    audit.update(['X'], 'annual', ['absent'], (1,), values, torch.ones(1, 3, dtype=torch.bool),
        torch.tensor([[1, 2, 3]]) * DAY_NS,
        {'next_annual_token': torch.zeros_like(values), 'next_annual_subtoken': torch.zeros(1, 3, 1, 1)})
    rows = audit.report()['metrics']
    assert len(rows) == 2
    assert all(row['values'] == 0 and row['model_mse'] is None for row in rows)
    audit.close()
