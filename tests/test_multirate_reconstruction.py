import pytest
import torch

from quant_orchestrator.research_tools.multirate_objectives import (
    family_channels, reconstruction_mask, reconstruction_targets,
)
from quant_orchestrator.platforms.ml_frameworks.torch.models.transformers.multirate import (
    MultiRateTransformer, MultiRateTransformerConfig, MultiRatePredictionTaskSpec,
)


def test_reconstruction_preserves_values_even_when_channel_means_match():
    values = torch.tensor([[[1., 9., 4.], [5., 5., float('nan')], [7., 3., 6.]]])
    padding = torch.zeros(1, 3, dtype=torch.bool)
    dates = torch.tensor([[10, 20, 30]])
    targets = reconstruction_targets(values, padding, dates, torch.ones_like(values, dtype=torch.bool), (2, 1))
    target, valid = targets['masked_subtoken']
    assert torch.equal(target[0, 0, 0], torch.tensor([1., 9.]))
    assert torch.equal(target[0, 1, 0], torch.tensor([5., 5.]))
    assert not valid[0, 1, 1].any()  # Missing value and family-width padding.
    prediction = target.clone().requires_grad_()
    with torch.no_grad():
        prediction[0, 0, 0] = torch.tensor([5., 5.])
    loss = (prediction[valid] - target[valid]).square().mean()
    assert loss > 0  # These observations had identical scalar-average targets.
    loss.backward()
    assert prediction.grad[0, 0, 0].abs().min() > 0


def test_token_and_family_successors_preserve_coherent_future_observations():
    values = torch.tensor([[[1., 2.], [99., 99.], [3., float('nan')], [4., 8.]]])
    targets = reconstruction_targets(values, torch.zeros(1, 4, dtype=torch.bool),
        torch.tensor([[1, 1, 2, 3]]), torch.zeros_like(values, dtype=torch.bool), (2,))
    token, token_valid = targets['next_token']
    family, family_valid = targets['next_subtoken']
    assert token[0, 0, 0] == 3 and not token_valid[0, 0, 1]
    assert family[0, 0, 0, 0] == 3
    assert not family_valid[0, 0, 0, 1]
    assert not token_valid[0, -1].any() and not family_valid[0, -1].any()


def test_separate_masks_retain_family_context_and_ignore_missing_padding():
    values = torch.ones(4, 100, 8)
    values[:, :, 0] = float('nan')
    padding = torch.zeros(4, 100, dtype=torch.bool)
    padding[:, :10] = True
    kwargs = dict(widths=(2, 3, 3), family_probability=.5, feature_probability=1.)
    masks = reconstruction_mask(values, padding, generator=torch.Generator().manual_seed(4), **kwargs)
    again = reconstruction_mask(values, padding, generator=torch.Generator().manual_seed(4), **kwargs)
    features, families = masks
    assert all(torch.equal(a, b) for a, b in zip(masks, again))
    assert features.any() and families.any()
    for mask in masks:
        assert not mask[:, :10].any() and not mask[:, :, 0].any()
    assert not features[..., :2].any()  # Only one observed sibling.
    assert (features[:, 10:, 2:5].sum(-1) == 2).all()
    assert torch.equal(families[..., 2], families[..., 3])
    assert torch.equal(families[..., 3], families[..., 4])
    targets = reconstruction_targets(values, padding, torch.arange(100).expand(4, -1),
                                     features, (2, 3, 3), family_selected=families)
    assert torch.equal(targets['masked_subtoken'][1], family_channels(features, (2, 3, 3)))
    assert torch.equal(targets['masked_token'][1], families)


@pytest.mark.parametrize('rate', ['annual', 'quarterly', 'daily', 'sparse'])
def test_hidden_observation_uses_past_context_and_cannot_see_future(rate):
    torch.manual_seed(9)
    model = MultiRateTransformer(
        {r: 3 for r in ('annual', 'quarterly', 'daily', 'sparse')},
        config=MultiRateTransformerConfig(d_model=8, num_heads=2, layers=1, dropout=0.),
        feature_families={r: {'first': 2, 'second': 1} for r in ('annual', 'quarterly', 'daily', 'sparse')},
        tasks=(), prediction_tasks=[
            MultiRatePredictionTaskSpec('masked', 'masked_token', level='token', output_dim=3, source=rate),
            MultiRatePredictionTaskSpec('subtoken', 'masked_token', level='subtoken', output_dim=2, source=rate),
        ],
    ).eval()
    data = {}
    for r in ('annual', 'quarterly', 'daily', 'sparse'):
        data[f'{r}_values'] = torch.randn(1, 3, 3)
        data[f'{r}_dates'] = torch.tensor([[1, 2, 3]])
        data[f'{r}_padding_mask'] = torch.zeros(1, 3, dtype=torch.bool)
        data[f'{r}_family_presence'] = torch.ones(1, 3, 2, dtype=torch.bool)
    data[f'{rate}_values'][:, 1] = float('nan')
    base = model(**data)['prediction_outputs']
    data[f'{rate}_values'][:, 2] += 100
    future = model(**data)['prediction_outputs']
    for key in base:
        torch.testing.assert_close(base[key][:, 1], future[key][:, 1])
    data[f'{rate}_values'][:, 0] *= -10
    past = model(**data)['prediction_outputs']
    for key in base:
        assert not torch.allclose(base[key][:, 1], past[key][:, 1])
    sum(out[:, 1].square().sum() for out in past.values()).backward()
    grads = [p.grad for p in model.prediction_heads.parameters()]
    assert all(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)


def test_family_widths_cannot_drop_values():
    with pytest.raises(ValueError, match='partition'):
        family_channels(torch.ones(1, 2, 3), (1, 1))


def test_source_families_group_contiguous_fields_without_averaging_them():
    from quant_orchestrator.research_tools.multirate_objectives import feature_family_layout
    assert feature_family_layout(['fmp.income.profit', 'fmp.income.revenue', 'price.close', 'price.open']) == {
        'fmp.income': 2, 'price': 2}
    with pytest.raises(ValueError, match='not contiguous'):
        feature_family_layout(['price.close', 'fmp.income.revenue', 'price.open'])


def test_family_adapter_knows_which_feature_is_missing_when_values_are_zero():
    from quant_orchestrator.platforms.ml_frameworks.torch.models.transformers.multirate.coverage import CoverageAwareInput
    torch.manual_seed(3)
    adapter = CoverageAwareInput({'family': 2}, 8)
    values = torch.tensor([[[float('nan'), 0.], [0., float('nan')]]])
    encoded = adapter(values)
    assert not torch.allclose(encoded[:, 0], encoded[:, 1])
    encoded.square().sum().backward()
    assert adapter.feature_missingness_embeddings.grad.abs().sum() > 0


def test_family_heads_use_own_history_and_token_heads_use_combined_history():
    torch.manual_seed(12)
    rates = ('annual', 'quarterly', 'daily', 'sparse')
    model = MultiRateTransformer(
        {r: 3 for r in rates},
        config=MultiRateTransformerConfig(d_model=8, num_heads=2, layers=1, dropout=0.),
        feature_families={r: {'family_a': 2, 'family_b': 1} for r in rates},
        tasks=(), prediction_tasks=[
            MultiRatePredictionTaskSpec('next', 'next_token', level='subtoken', output_dim=2, source='daily'),
            MultiRatePredictionTaskSpec('masked', 'masked_token', level='subtoken', output_dim=2, source='daily'),
            MultiRatePredictionTaskSpec('next_joint', 'next_token', level='token', output_dim=3, source='daily'),
            MultiRatePredictionTaskSpec('masked_joint', 'masked_token', level='token', output_dim=3, source='daily'),
        ],
    ).eval()
    data = {f'{r}_values': torch.randn(1, 3, 3) for r in rates}
    data.update({f'{r}_dates': torch.tensor([[1, 2, 3]]) for r in rates})
    data['daily_values'][:, 1, :2] = float('nan')
    data['daily_family_presence'] = torch.ones(1, 3, 2, dtype=torch.bool)
    base = model(**data)['prediction_outputs']
    data['daily_values'][:, 1, 2] += 100  # Only the other family, same date.
    changed = model(**data)['prediction_outputs']
    torch.testing.assert_close(base['next'][:, 1, 0], changed['next'][:, 1, 0])
    torch.testing.assert_close(base['masked'][:, 1, 0], changed['masked'][:, 1, 0])
    for name in ('next_joint', 'masked_joint'):
        assert not torch.allclose(base[name][:, 1], changed[name][:, 1])
    data['daily_values'][:, 0, :2] *= -10  # Own family's past must matter.
    past = model(**data)['prediction_outputs']
    assert not torch.allclose(changed['next'][:, 1, 0], past['next'][:, 1, 0])

    assert not torch.allclose(changed['masked'][:, 1, 0], past['masked'][:, 1, 0])
    data['daily_values'][:, 2] += 1000
    future = model(**data)['prediction_outputs']
    for name in past:
        torch.testing.assert_close(past[name][:, 1], future[name][:, 1])


def test_family_successor_skips_absent_family_but_token_keeps_next_joint_date():
    values = torch.tensor([[[1., 2., 10.], [float('nan'), float('nan'), 20.], [3., 4., 30.]]])
    targets = reconstruction_targets(values, torch.zeros(1, 3, dtype=torch.bool),
        torch.tensor([[1, 2, 3]]), torch.zeros_like(values, dtype=torch.bool), (2, 1))
    family, fv = targets['next_subtoken']
    token, tv = targets['next_token']
    torch.testing.assert_close(family[0, 0, 0], torch.tensor([3., 4.]))
    assert fv[0, 0, 0].all()
    assert token[0, 0, 2] == 20 and not tv[0, 0, :2].any()
