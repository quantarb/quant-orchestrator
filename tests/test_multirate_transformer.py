import pytest
import torch

from quant_orchestrator.platforms.ml_frameworks.torch import (
    AutoFeatureEngineer,
    MultiRatePredictionTaskSpec,
    MultiRateTaskSpec,
    MultiRateTransformer,
    MultiRateTransformerConfig,
    build_attention_mask,
)
from quant_orchestrator.platforms.ml_frameworks.torch.models.transformers.multirate import (
    CoverageAwareInput,
    MultiRateTransformer as NestedMultiRateTransformer,
)


def _tasks():
    return (
        MultiRateTaskSpec("daily_hits", "token", output_dim=6),
        MultiRateTaskSpec("sector", "document", output_dim=4, source="fused"),
        MultiRateTaskSpec("annual_context", "document", output_dim=2, source="annual"),
    )


def test_temporal_mask_blocks_future_and_cross_sectional_mask_shares_dates():
    temporal = build_attention_mask(torch.tensor([0, 1, 2]), mode="temporal")
    assert torch.isfinite(temporal[2, 0])
    assert torch.isneginf(temporal[0, 1])

    cross_sectional = build_attention_mask(torch.tensor([10, 10, 11]), mode="cross_sectional")
    assert torch.isfinite(cross_sectional[0, 1])
    assert torch.isneginf(cross_sectional[0, 2])
    assert torch.isfinite(cross_sectional[2, 0])


def test_multirate_model_is_owned_by_nested_transformer_module():
    assert MultiRateTransformer is NestedMultiRateTransformer


@pytest.mark.parametrize("backbone", ["encoder_only", "decoder_only", "encoder_decoder"])
@pytest.mark.parametrize("attention_mode", ["temporal", "cross_sectional"])
def test_all_backbones_return_token_and_document_tasks(backbone, attention_mode):
    config = MultiRateTransformerConfig(
        backbone=backbone,
        d_model=16,
        num_heads=4,
        layers=1,
        document_pool="mean",
    )
    model = MultiRateTransformer(
        {"annual": 3, "quarterly": 4, "daily": 5},
        config=config,
        tasks=_tasks(),
    )
    output = model(
        torch.randn(2, 5, 5),
        torch.randn(2, 2, 3),
        torch.randn(2, 3, 4),
        attention_mode=attention_mode,
        daily_dates=torch.tensor([0, 0, 1, 1, 2]),
    )
    assert output["token_states"].shape == (2, 5, 16)
    assert output["document_state"].shape == (2, 16)
    assert output["token_outputs"]["daily_hits"].shape == (2, 5, 6)
    assert output["document_outputs"]["sector"].shape == (2, 4)
    assert output["document_outputs"]["annual_context"].shape == (2, 2)


def test_padding_mask_is_respected_by_document_pooling():
    config = MultiRateTransformerConfig(d_model=16, num_heads=4, layers=1, document_pool="last")
    model = MultiRateTransformer(
        {"annual": 2, "quarterly": 2, "daily": 2}, config=config, tasks=_tasks()[:1]
    )
    output = model(
        torch.randn(1, 3, 2), torch.randn(1, 1, 2), torch.randn(1, 1, 2),
        daily_padding_mask=torch.tensor([[False, True, True]]),
    )
    assert torch.equal(output["document_state"], output["token_states"][:, 0])


def test_coverage_input_distinguishes_missing_from_observed_zero():
    torch.manual_seed(7)
    layer = CoverageAwareInput({"prices": 2, "fundamentals": 1}, 8)
    values = torch.tensor([[[0.0, 0.0, 0.0]]])
    observed = layer(values, family_presence=torch.tensor([[[1.0, 0.0]]]))
    missing = layer(values, family_presence=torch.tensor([[[0.0, 1.0]]]))
    assert not torch.equal(observed, missing)


def test_model_accepts_family_presence_and_nan_imputation_path():
    model = MultiRateTransformer(
        {"annual": 3, "quarterly": 3, "daily": 3},
        config=MultiRateTransformerConfig(d_model=12, num_heads=3, layers=1),
        feature_families={
            "annual": {"fundamentals": 2, "prices": 1},
            "quarterly": {"fundamentals": 2, "prices": 1},
            "daily": {"fundamentals": 2, "prices": 1},
        },
        tasks=_tasks()[:1],
    )
    output = model(
        torch.tensor([[[float("nan"), 1.0, 0.0]]]),
        torch.zeros(1, 1, 3),
        torch.zeros(1, 1, 3),
        daily_family_presence=torch.tensor([[[0.0, 1.0]]]),
    )
    assert torch.isfinite(output["token_states"]).all()


def test_auto_feature_engineer_has_temporal_and_cross_sectional_paths():
    torch.manual_seed(3)
    block = AutoFeatureEngineer(12)
    values = torch.randn(2, 4, 12)
    families = torch.randn(2, 4, 2, 12)
    presence = torch.ones(2, 4, 2)
    temporal = block(values, mode="temporal", family_states=families, family_presence=presence)
    cross = block(
        values,
        mode="cross_sectional",
        dates=torch.tensor([0, 0, 1, 1]),
        family_states=families,
        family_presence=presence,
    )
    assert temporal.shape == values.shape
    assert cross.shape == values.shape
    assert not torch.equal(temporal, cross)


def test_auto_feature_engineer_has_cross_rate_path():
    block = AutoFeatureEngineer(12)
    rates = tuple(torch.randn(2, 12) for _ in range(3))
    output = block.cross_rate_features(rates)
    assert output.shape == (2, 12)


def test_prediction_heads_support_token_and_subtoken_objectives():
    model = MultiRateTransformer(
        {"annual": 3, "quarterly": 3, "daily": 3},
        config=MultiRateTransformerConfig(d_model=12, num_heads=3, layers=1),
        feature_families={
            "annual": {"fundamentals": 2, "prices": 1},
            "quarterly": {"fundamentals": 2, "prices": 1},
            "daily": {"fundamentals": 2, "prices": 1},
        },
        prediction_tasks=(
            MultiRatePredictionTaskSpec("next_daily_token", "next_token", "token", output_dim=3),
            MultiRatePredictionTaskSpec("masked_daily_family", "masked_token", "subtoken", output_dim=2),
        ),
    )
    output = model(
        torch.randn(2, 4, 3), torch.randn(2, 2, 3), torch.randn(2, 3, 3),
    )
    assert output["prediction_outputs"]["next_daily_token"].shape == (2, 4, 3)
    assert output["prediction_outputs"]["masked_daily_family"].shape == (2, 4, 2, 2)
