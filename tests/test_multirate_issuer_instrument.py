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
