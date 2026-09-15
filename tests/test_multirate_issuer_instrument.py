import copy

import pytest
import torch

from quant_orchestrator.platforms.ml_frameworks.torch.models.transformers.multirate import (
    MultiRateTransformer, MultiRateTransformerConfig, MultiRateTaskSpec,
)


def model():
    return MultiRateTransformer(
        {rate: 2 for rate in ('annual', 'quarterly', 'daily', 'sparse')},
        config=MultiRateTransformerConfig(d_model=8, num_heads=2, layers=1, dropout=0.),
        tasks=[MultiRateTaskSpec('return', 'token', source='daily')],
        modalities=('equity', 'option'),
    )


def inputs():
    return {f'{rate}_values': torch.randn(2, 3, 2) for rate in ('annual', 'quarterly', 'daily', 'sparse')}


def test_supervised_instrument_loss_reaches_issuer_encoders():
    torch.manual_seed(3)
    net = model().train()
    out = net(**inputs())
    out['token_outputs']['return'][:, -1].square().sum().backward()
    for encoder in (net.annual_encoder, net.quarterly_encoder, net.encoders['sparse'], net.encoders['daily']):
        grads = [p.grad for p in encoder.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert sum(g.abs().sum().item() for g in grads) > 0


def test_grouped_projection_matches_outputs_and_accumulated_gradients():
    torch.manual_seed(4)
    net = model().eval()
    reference = copy.deepcopy(net)
    data = inputs()
    for rate in ('annual', 'quarterly'):
        data[f'{rate}_values'][1] = data[f'{rate}_values'][0]
    calls = []
    hook = net.coverage_inputs['annual'].register_forward_pre_hook(lambda _, args: calls.append(args[0].shape[0]))
    grouped = net(**data, rate_context_ids={'annual': torch.tensor([0, 0]), 'quarterly': torch.tensor([0, 0])})
    plain = reference(**data)
    hook.remove()
    assert calls[0] == 1
    torch.testing.assert_close(grouped['token_outputs']['return'], plain['token_outputs']['return'])
    grouped['token_outputs']['return'].sum().backward()
    plain['token_outputs']['return'].sum().backward()
    for (name, p), (_, q) in zip(net.named_parameters(), reference.named_parameters()):
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, atol=2e-5, rtol=2e-4, msg=name)


def test_future_observations_do_not_change_earlier_instrument_predictions():
    net = model().eval()
    data = inputs()
    dates = torch.tensor([[1, 2, 4], [10, 20, 40]])
    data.update({f'{rate}_dates': dates for rate in ('annual', 'quarterly', 'daily', 'sparse')})
    with torch.no_grad():
        before = net(**data)['token_outputs']['return']
        for rate in ('annual', 'quarterly', 'daily', 'sparse'):
            data[f'{rate}_values'][:, -1] += 500
        after = net(**data)['token_outputs']['return']
    torch.testing.assert_close(before[:, :2], after[:, :2])


def test_grouped_context_rejects_different_corruptions():
    net = model()
    with pytest.raises(ValueError, match='different inputs or masks'):
        net(**inputs(), rate_context_ids={'annual': torch.tensor([0, 0])})


def test_cached_inference_preserves_subtoken_and_instrument_outputs():
    net = model().eval()
    data = inputs()
    with torch.no_grad():
        plain = net(**data)
        cached = net(**data, rate_cache=net.rate_cache_from_output(plain))
    torch.testing.assert_close(plain['token_outputs']['return'], cached['token_outputs']['return'])
    net.train()
    with pytest.raises(ValueError, match='inference-only'):
        net(**data, rate_cache=net.rate_cache_from_output(plain))


def test_cache_rejects_updated_weights_and_changed_windows():
    net = model().eval()
    data = inputs()
    with torch.no_grad():
        cached = net(**data)['rate_cache']
        data['annual_values'][0, 0, 0] += 1
        with pytest.raises(ValueError, match='input/window changed'):
            net(**data, rate_cache=cached)
        next(net.parameters()).add_(1)
        with pytest.raises(ValueError, match='updated model weights'):
            net(**data, rate_cache=cached)


def test_separate_issuer_daily_context_is_shared_and_date_causal():
    net = model().eval()
    data = inputs()
    values = torch.randn(1, 3, 2).expand(2, -1, -1).clone()
    dates = torch.tensor([[0, 1, 5], [0, 1, 5]])
    payload = {'daily': {'values': values, 'dates': dates,
                        'padding': torch.zeros(2, 3, dtype=torch.bool),
                        'context_ids': torch.tensor([0, 0])}}
    before = net(**data, issuer_streams=payload)
    assert before['issuer_reuse']['daily'] == {'requested': 2, 'encoded': 1}
    values[:, -1] += 1000
    after = net(**data, issuer_streams=payload)
    torch.testing.assert_close(before['token_outputs']['return'], after['token_outputs']['return'])
    before['token_outputs']['return'].square().sum().backward()
    assert net.issuer_context_fusion.weight.grad.abs().sum() > 0


def test_model_checkpoint_round_trip(tmp_path):
    net = model().eval()
    data = inputs()
    path = tmp_path / 'model.pt'
    torch.save(net.state_dict(), path)
    restored = model().eval()
    restored.load_state_dict(torch.load(path, weights_only=True))
    with torch.no_grad():
        torch.testing.assert_close(net(**data)['token_outputs']['return'], restored(**data)['token_outputs']['return'])


def test_final_context_order_is_annual_quarterly_issuer_daily_sparse_instrument():
    net=model().eval()
    with torch.no_grad():
        for parameter in net.information_age.parameters():
            parameter.zero_()
    captured=[];instrument=[]
    fusion_hook=net.instrument_fusion.register_forward_pre_hook(lambda _,args:captured.append(args[0]))
    price_hook=net.encoders['daily'].register_forward_hook(lambda _,args,value:instrument.append(value))
    data=inputs()
    dates=torch.tensor([[1,2,3],[1,2,3]])
    data.update({f'{r}_dates':dates for r in ('annual','quarterly','daily','sparse')})
    with torch.no_grad():
        out=net(**data)
    fusion_hook.remove();price_hook.remove()
    parts=captured[0].chunk(5,dim=-1)
    for index,rate in [(0,'annual'),(1,'quarterly'),(3,'sparse')]:
        states=out['rate_states'][rate]
        expected=states.cumsum(1)/torch.arange(1,4)[None,:,None]
        torch.testing.assert_close(parts[index],expected)
    assert torch.count_nonzero(parts[2])==0  # No issuer-daily payload in this test.
    torch.testing.assert_close(parts[4],instrument[0])


@pytest.mark.parametrize('difference', [None, 'memory', 'dates', 'values', 'padding'])
def test_recurrent_issuer_sharing_preserves_outputs_gradients_and_memory(difference):
    from dataclasses import replace
    from quant_orchestrator.research_tools.annual_memory import AnnualMemory
    torch.manual_seed(42)
    reference = model().train()
    shared = copy.deepcopy(reference)
    shared.config = replace(shared.config, share_recurrent_issuer_context=True)
    data = inputs()
    for rate in ('annual', 'quarterly'):
        data[rate + '_values'][1] = data[rate + '_values'][0]
    batch = [dict(symbol=s, date='2023-12-31', document_start='2023-01-01') for s in ['A', 'A_CALL']]
    memory = AnnualMemory().inputs(batch, reference)
    for rate in reference.config.rates:
        data[rate + '_padding_mask'] = torch.zeros(2, 3, dtype=torch.bool)
    payload = dict(values=torch.randn(1, 3, 2).expand(2, -1, -1).clone(),
                   dates=torch.tensor([[0, 1, 2], [0, 1, 2]]),
                   padding=torch.zeros(2, 3, dtype=torch.bool), context_ids=torch.tensor([0, 0]))
    if difference == 'memory':
        memory['issuer_daily']['states'][1, 0] = 1.
        memory['annual']['states'][1, 0] = 1.
    elif difference == 'dates':
        payload['dates'][1, -1] += 1
    elif difference == 'values':
        payload['values'][1, -1, 0] += 1
    elif difference == 'padding':
        payload['padding'][1, -1] = True
    plain = reference(**data, issuer_streams={'daily': payload}, annual_memory=memory)
    grouped = shared(**data, issuer_streams={'daily': payload}, annual_memory=memory,
                     rate_context_ids={rate: torch.tensor([0, 0]) for rate in ('annual', 'quarterly')})
    assert grouped['issuer_reuse']['annual']['encoded'] == (2 if difference == 'memory' else 1)
    assert grouped['issuer_reuse']['daily']['encoded'] == (1 if difference is None else 2)
    torch.testing.assert_close(grouped['token_outputs']['return'], plain['token_outputs']['return'])
    for rate, values in plain['annual_memory'].items():
        for key, value in values.items():
            torch.testing.assert_close(grouped['annual_memory'][rate][key], value)
    for out in [plain, grouped]:
        out['token_outputs']['return'].square().sum().backward()
    for (name, p), (_, q) in zip(shared.named_parameters(), reference.named_parameters()):
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, atol=2e-5, rtol=2e-4, msg=name)
