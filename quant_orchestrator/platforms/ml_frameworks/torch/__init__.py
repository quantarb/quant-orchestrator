"""PyTorch ML framework provider."""

from quant_orchestrator.platforms.ml_frameworks.torch.provider import TorchFramework, torch_provider
from quant_orchestrator.platforms.ml_frameworks.torch.runtime import (
    TorchRuntimeInfo,
    configure_torch_runtime,
)

__all__ = [
    "MultiRateTaskSpec",
    "MultiRatePredictionTaskSpec",
    "MultiRateTransformer",
    "MultiRateTransformerConfig",
    "CoverageAwareInput",
    "AutoFeatureEngineer",
    "SubtokenDocumentCorpus",
    "build_1t_subtoken_documents",
    "mean_document_embeddings",
    "TorchFramework",
    "TorchRuntimeInfo",
    "build_attention_mask",
    "configure_torch_runtime",
    "torch_provider",
]


def __getattr__(name: str):
    """Load Torch model classes lazily because Torch is an optional extra."""
    if name in {"MultiRateTaskSpec", "MultiRatePredictionTaskSpec", "MultiRateTransformer", "MultiRateTransformerConfig", "build_attention_mask", "CoverageAwareInput", "AutoFeatureEngineer", "SubtokenDocumentCorpus", "build_1t_subtoken_documents", "mean_document_embeddings"}:
        from quant_orchestrator.platforms.ml_frameworks.torch.models.transformers import multirate

        return getattr(multirate, name)
    raise AttributeError(name)
