import torch

from quant_orchestrator.platforms.ml_frameworks.torch.models.transformers.multirate.instrument_attention import InstrumentAttention


def test_attention_connects_instruments_without_crossing_dates_or_issuers():
    torch.manual_seed(7)
    layer = InstrumentAttention(8, 2).eval()
    states = torch.randn(3, 3, 8, requires_grad=True)
    dates = torch.tensor([[1, 2, 3]] * 3)
    ids = torch.tensor([0, 0, 1])
    padding = torch.zeros(3, 3, dtype=torch.bool)
    out = layer(states, dates, padding, ids)
    out[0, 1, 0].backward()
    assert states.grad[1, 1].abs().sum() > 0
    assert states.grad[1, 2].abs().sum() == 0
    assert states.grad[2].abs().sum() == 0
    changed = states.detach().clone()
    changed[1, 2] += torch.arange(8) * 100
    changed[2] += torch.arange(8) * 100
    after = layer(changed, dates, padding, ids)
    torch.testing.assert_close(out[0, :2], after[0, :2])


def test_missing_quotes_are_excluded_and_instrument_order_is_equivariant():
    torch.manual_seed(8)
    layer = InstrumentAttention(8, 2).eval()
    states = torch.randn(3, 3, 8)
    dates = torch.tensor([[1, 2, 3]] * 3)
    ids = torch.zeros(3, dtype=torch.long)
    padding = torch.zeros(3, 3, dtype=torch.bool)
    padding[1, 1] = True
    before = layer(states, dates, padding, ids)
    changed = states.clone()
    changed[1, 1] += 1000
    after = layer(changed, dates, padding, ids)
    torch.testing.assert_close(before[0], after[0])
    permutation = torch.tensor([2, 0, 1])
    reordered = layer(states[permutation], dates[permutation], padding[permutation], ids[permutation])
    torch.testing.assert_close(before[permutation], reordered)


def test_supervised_model_head_uses_other_instruments():
    from quant_orchestrator.platforms.ml_frameworks.torch.models.transformers.multirate import (
        MultiRateTransformer, MultiRateTransformerConfig, MultiRateTaskSpec,
    )
    torch.manual_seed(9)
    model = MultiRateTransformer({rate: 2 for rate in ('annual', 'quarterly', 'daily', 'sparse')},
        config=MultiRateTransformerConfig(d_model=8, num_heads=2, layers=1, dropout=0., cross_instrument_attention=True),
        tasks=[MultiRateTaskSpec('return', 'token', source='daily')])
    data = {rate + '_values': torch.randn(2, 3, 2, requires_grad=True)
            for rate in ('annual', 'quarterly', 'daily', 'sparse')}
    out = model(**data, instrument_group_ids=torch.tensor([0, 0]))
    out['token_outputs']['return'][0, 1].sum().backward()
    assert data['daily_values'].grad[1, :2].abs().sum() > 0
    assert data['daily_values'].grad[1, 2].abs().sum() == 0
