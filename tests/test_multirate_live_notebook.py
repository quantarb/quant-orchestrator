"""Notebook result state must belong to the currently selected live run."""
import ast
import json
from pathlib import Path

import nbformat
import pytest

NOTEBOOK = Path(__file__).resolve().parents[1]/'notebooks/multirate_warehouse_training.ipynb'


@pytest.mark.parametrize('returncode', [0, 1])
def test_failed_run_exception_includes_current_process_error(returncode):
    notebook = nbformat.read(NOTEBOOK, 4)
    failure = next(node for node in ast.parse(notebook.cells[31].source).body
                   if isinstance(node, ast.If) and ast.unparse(node.test) == 'returncode or not run_complete')
    namespace = dict(returncode=returncode, run_complete=False, universe_label=lambda: '10B',
                     log_path=Path('/tmp/current-run.log'),
                     live_output_tail=['error: ABC/ratios/annual: missing fiscal period'])
    with pytest.raises(RuntimeError, match='ABC/ratios/annual: missing fiscal period') as error:
        exec(compile(ast.Module(body=[failure], type_ignores=[]), '<live-failure>', 'exec'), namespace)
    assert '/tmp/current-run.log' in str(error.value)
    assert '10B' in str(error.value)


def state():
    notebook=nbformat.read(NOTEBOOK,4)
    namespace={}
    exec(notebook.cells[1].source,namespace)
    exec(notebook.cells[5].source,namespace)
    functions=[node for node in ast.parse(notebook.cells[31].source).body if isinstance(node,ast.FunctionDef)]
    exec(compile(ast.Module(body=functions,type_ignores=[]),'<live-results>','exec'),namespace)
    namespace['run_settings']=namespace['current_run_settings']().copy()
    return notebook,namespace


@pytest.mark.parametrize('market_cap,universe',[(1e12,'1T'),(1e11,'100B'),(1e10,'10B'),(5e10,'50B')])
def test_command_and_reports_use_only_selected_universe(market_cap,universe):
    notebook,s=state()
    s['MIN_MARKET_CAP']=market_cap
    s['run_settings']=s['current_run_settings']().copy()
    exec(notebook.cells[29].source,s)
    command=s['training_command'](Path('/tmp/fresh-output'))
    assert command[command.index('--min-market-cap')+1]==str(market_cap)
    assert command[command.index('--progress-updates-per-epoch')+1]=='10'
    assert '--progress-every-batches' not in command
    assert command[command.index('--warehouse-start-date')+1]=='1900-01-01'
    assert command[command.index('--warehouse-option-start-date')+1]=='2021-01-01'
    report=dict(epoch=1,year=2024,asset_class='option',side='long_calls',capital_return=.12)
    assert s['consume_live_line']('[warehouse-backtest-book] '+json.dumps(report))
    frame=s['result_frame']()
    assert frame.universe.tolist()==[universe]
    assert frame.book.tolist()==['long_calls']
    # The final epoch event updates the same book, rather than duplicating it.
    epoch=dict(epoch=1,reports=[{k:v for k,v in report.items() if k!='epoch'}],inference_seconds=2.)
    s['consume_live_line']('[warehouse-backtest] '+json.dumps(epoch))
    assert len(s['live_reports'])==1
    s['MIN_MARKET_CAP']=market_cap*2
    with pytest.raises(RuntimeError,match='Settings changed'):
        s['result_frame']()


def test_new_configuration_resets_results_and_saved_notebook_has_no_stale_outputs():
    notebook,s=state()
    s['live_reports']={'old':{'universe':'100B'}}
    s['run_complete']=True
    exec(notebook.cells[5].source,s)
    assert s['live_reports']=={} and not s['run_complete'] and s['run_settings'] is None
    assert all(not c.get('outputs') for c in notebook.cells)
    source='\n'.join(c.source for c in notebook.cells if c.cell_type=='code')
    assert 'REVIEW_RUNS' not in source and 'read_json' not in source
    assert '.glob(' not in source and '.read_text(' not in source


def test_three_years_four_books_are_visible_before_and_during_backtesting():
    notebook,s=state()
    table=s['live_return_table']()
    assert table.index.get_level_values('year').tolist()==[2024,2025,2026]
    assert table.shape==(3,4) and table.isna().all().all()
    for year in (2024,2025,2026):
        for asset,sides in [('equity',('long','short')),('option',('long_calls','long_puts'))]:
            for side in sides:
                event=dict(epoch=1,year=year,asset_class=asset,side=side,capital_return=.1)
                s['consume_live_line']('[warehouse-backtest-book] '+json.dumps(event))
                assert s['live_return_table']().shape==(3,4)
    assert len(s['live_reports'])==12
    assert s['live_return_table']().notna().all().all()
    s['run_complete']=True
    exec(notebook.cells[38].source,s)
