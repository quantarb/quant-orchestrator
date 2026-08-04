import torch

from quant_orchestrator.platforms.ml_frameworks.torch.models.transformers.multirate import FactorizationPooling


def test_factorization_pooling_returns_one_prototype_per_matrix():
    pooling = FactorizationPooling(input_dim=5, rank=3)
    output = pooling(torch.randn(7, 5))
    assert output.shape == (1, 5)


def test_factorization_pooling_supports_batches_and_row_masks():
    pooling = FactorizationPooling(input_dim=4, rank=2)
    matrix = torch.randn(2, 6, 4)
    mask = torch.zeros(2, 6, dtype=torch.bool)
    mask[:, -1] = True
    output = pooling(matrix, mask)
    assert output.shape == (2, 1, 4)
    assert torch.isfinite(output).all()
