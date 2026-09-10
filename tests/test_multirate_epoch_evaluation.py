import pytest
from quant_orchestrator.research_tools.epoch_evaluation import (
    evaluation_command, last_epoch_batches, trend_report, format_epoch_report,
)


def report(skill, mse=2.):
    return {'metrics': [dict(rate='daily', level='subtoken', family='price',
        values=10, unique_pairs=2, persistence_mse=4., model_mse=mse,
        skill=skill, beats_persistence=mse<4.)]}


def test_trends_compare_fixed_targets_and_show_improvement():
    first=trend_report(1, report(-.5,6.))
    second=trend_report(2, report(.5,2.), first)
    assert second['metrics'][0]['delta_skill']==1.
    assert second['groups_beating_persistence']==1
    assert 'delta_skill' in format_epoch_report(second)
    changed=report(.5);changed['metrics'][0]['values']=11
    with pytest.raises(ValueError, match='changed'):
        trend_report(3, changed, second)


def test_constant_baseline_does_not_produce_false_skill_or_trend():
    data=report(None,1.);data['metrics'][0]['persistence_mse']=0.
    data['metrics'][0]['beats_persistence']=False
    first=trend_report(1,data)
    second=trend_report(2,data,first)
    assert second['zero_error_baseline_groups']==1
    assert second['metrics'][0]['delta_skill'] is None
    assert 'null' in format_epoch_report(second)


def test_evaluation_command_never_trains_or_changes_training_dates():
    base=['python','train.py','--train-end-date','2024-01-01','--output-dir','train',
        '--skip-predictions','--max-samples','2048']
    command=evaluation_command(base,'snapshot.pt','validation','2024-01-02','2024-12-31',256)
    assert '--skip-predictions' not in command
    assert command.count('--inference-only')==1
    assert command[command.index('--train-end-date')+1]=='2024-01-01'
    assert command[command.index('--max-samples')+1]=='256'
    assert command[command.index('--checkpoint')+1]=='snapshot.pt'
    assert '--inference-only' not in base


def test_completed_epoch_size_is_read_from_training_progress(tmp_path):
    log=tmp_path/'train.log'
    log.write_text('[multirate-train] epoch=1/12 batch=762/762 samples=86081\n'
                   '[multirate-train] epoch=2/12 batch=10/763 samples=1280\n')
    assert last_epoch_batches(log)=={0:762,1:763}
