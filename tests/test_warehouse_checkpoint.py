"""Saved warehouse weights must reach evaluation without any optimizer steps."""
import json
from types import SimpleNamespace

import pytest
import torch

from quant_orchestrator.research_tools import warehouse_multirate_training as training


@pytest.fixture
def saved_epoch(tmp_path, monkeypatch):
    source = tmp_path/'source'
    source.mkdir()
    (source/'universe.json').write_text(json.dumps({'equity_symbols':['A']}))
    model = torch.nn.Linear(2, 1)
    with torch.no_grad():
        model.weight.fill_(3)
        model.bias.fill_(7)
    configuration = dict(document_contract=training.ANNUAL_CONTRACT, options_per_side=0,
        min_market_cap=1e10, warehouse_start_date='1900-01-01', train_end_date='2024-01-01',
        prediction_start_date='2024-01-01', prediction_end_date='2026-09-09', warehouse_option_start_date=None)
    payload = dict(metrics={'epoch_complete':True, 'epoch':1}, configuration=configuration,
        feature_schema=['close'], normalization='signed_log1p_div10', asset_classes=['equity'],
        state_dict=model.state_dict())
    checkpoint = source/'epoch_0001.pt'
    torch.save(payload, checkpoint)
    stream = SimpleNamespace(features=['close'], prices={'A':None})
    monkeypatch.setattr(training, 'WarehouseAnnualStream', lambda **kwargs:stream)
    monkeypatch.setattr(training, 'warehouse_model', lambda *args:(torch.nn.Linear(2,1),None,None,None))
    def optimizer_forbidden(*args, **kwargs):
        raise AssertionError('Evaluation must not create an optimizer')
    monkeypatch.setattr(torch.optim, 'AdamW', optimizer_forbidden)
    return checkpoint, payload, stream


def test_saved_weights_reach_evaluation(saved_epoch, tmp_path, monkeypatch):
    checkpoint, payload, stream = saved_epoch
    def evaluate(model, actual_stream, args, epoch):
        assert actual_stream is stream and epoch == 1
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, payload['state_dict'][name], rtol=0, atol=0)
        return [{'side':'long', 'capital_return':0.1}]
    monkeypatch.setattr(training, 'evaluate_epoch', evaluate)
    output = tmp_path/'evaluation'
    reports = training.evaluate_warehouse_checkpoint(checkpoint, output, device='cpu')
    status = json.loads((output/'status.json').read_text())
    assert reports == status['reports']
    assert status['stage'] == 'complete' and status['optimizer_steps'] == 0
    assert len(status['checkpoint_sha256']) == 64
    with pytest.raises(FileExistsError):
        training.evaluate_warehouse_checkpoint(checkpoint, output, device='cpu')


@pytest.mark.parametrize('invalid', ['incomplete','normalization','contract','schema','universe','assets','weights'])
def test_mismatches_fail_before_evaluation(saved_epoch, tmp_path, monkeypatch, invalid):
    checkpoint, payload, stream = saved_epoch
    if invalid == 'incomplete': payload['metrics']['epoch_complete'] = False
    if invalid == 'normalization': payload['normalization'] = 'different'
    if invalid == 'contract': payload['configuration']['document_contract'] = 'different'
    if invalid == 'schema': stream.features = ['changed']
    if invalid == 'universe': stream.prices = {'B':None}
    if invalid == 'assets': payload['asset_classes'] = ['equity','option']
    if invalid == 'weights': payload['state_dict'] = {}
    torch.save(payload, checkpoint)
    def forbidden(*args): raise AssertionError('Invalid checkpoint must not be evaluated')
    monkeypatch.setattr(training, 'evaluate_epoch', forbidden)
    with pytest.raises((ValueError, RuntimeError)):
        training.evaluate_warehouse_checkpoint(checkpoint, tmp_path/'evaluation', device='cpu')


def test_inference_blocks_reuse_sources_and_keep_year_order():
    from datetime import datetime
    import polars as pl
    warmed, calls = [], []
    years = {'A':[2024,2025,2026], 'B':[2024,2026], 'C':[2025,2026]}
    def sample(symbol, year, **kwargs):
        assert kwargs['training'] is False
        assert symbol in warmed[-1]
        calls.append((symbol,year))
        return {'symbol':symbol, 'year':year}
    stream = SimpleNamespace(prices={s:pl.DataFrame({'date':[datetime(y,1,2) for y in ys]}) for s,ys in years.items()},
        prepare_equity_sources=lambda block:warmed.append(block), sample=sample)
    batches = list(training.equity_inference_batches(stream, 2, datetime(2024,1,1), datetime(2026,9,9)))
    assert warmed == [['A','B'],['C']]
    assert sorted(calls) == sorted((s,y) for s,ys in years.items() for y in ys)
    for batch in batches:
        assert len({row['symbol'] for row in batch}) == len(batch) <= 2
    for symbol, expected in years.items():
        assert [row['year'] for batch in batches for row in batch if row['symbol']==symbol] == expected
