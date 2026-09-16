"""Compact masks preserve native attention, gradients and isolation."""
import copy

import pytest
import torch

from quant_orchestrator.platforms.ml_frameworks.torch.models.transformers.multirate.auto_features import family_temporal_attention


@pytest.mark.parametrize('batched_dates', [False, True])
@pytest.mark.parametrize('training', [False, True])
@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required'))])
def test_compact_family_attention_matches_native(batched_dates, training, device):
    torch.manual_seed(17)
    batch, families, length, width = 3, 2, 7, 12
    dtype = torch.float64 if device == 'cpu' else torch.float32
    tolerance = dict(atol=1e-12, rtol=1e-10) if device == 'cpu' else dict(atol=2e-5, rtol=1e-4)
    native = torch.nn.MultiheadAttention(width, 3, batch_first=True).to(device=device, dtype=dtype).train(training)
    compact = copy.deepcopy(native)
    x = torch.randn(batch * families, length, width, dtype=dtype, device=device, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    dates = torch.tensor([0, 0, 2, 3, 3, 5, 6], device=device)
    if batched_dates:
        dates = dates.expand(batch, -1).clone()
        dates[1, 3] = 2
    padding = torch.rand(batch * families, length, device=device) < .3
    padding[0] = True
    padding[:, 0] = False
    mask = dates.unsqueeze(-2) > dates.unsqueeze(-1)
    if batched_dates:
        mask = mask.repeat_interleave(families * native.num_heads, 0)
    expected = native(x, x, x, attn_mask=mask, key_padding_mask=padding, need_weights=False)[0]
    actual = family_temporal_attention(compact, y, dates, padding, batch, families)
    torch.testing.assert_close(actual, expected, **tolerance)
    weight = torch.randn_like(expected)
    (expected * weight).sum().backward()
    (actual * weight).sum().backward()
    torch.testing.assert_close(x.grad, y.grad, **tolerance)
    for a, b in zip(native.parameters(), compact.parameters()):
        torch.testing.assert_close(a.grad, b.grad, **tolerance)
    # Instrument 0's outputs must not depend on any other instrument, or on future dates.
    y = y.detach().requires_grad_()
    out = family_temporal_attention(compact, y, dates, padding, batch, families)
    grad = torch.autograd.grad(out[0, 2].square().sum(), y)[0]
    assert torch.count_nonzero(grad[1:]) == 0
    assert torch.count_nonzero(grad[0, 3:]) == 0
