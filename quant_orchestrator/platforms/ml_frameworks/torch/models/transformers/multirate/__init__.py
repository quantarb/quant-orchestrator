"""Multi-rate Transformer model family."""

from quant_orchestrator.platforms.ml_frameworks.torch.models.transformers.multirate.model import (
    MultiRatePredictionTaskSpec,
    MultiRateTaskSpec,
    MultiRateTransformer,
    MultiRateTransformerConfig,
    build_attention_mask,
)
from quant_orchestrator.platforms.ml_frameworks.torch.models.transformers.multirate.coverage import (
    CoverageAwareInput,
)
from quant_orchestrator.platforms.ml_frameworks.torch.models.transformers.multirate.auto_features import (
    AutoFeatureEngineer,
)
from quant_orchestrator.platforms.ml_frameworks.torch.models.transformers.multirate.documents import (
    SubtokenDocumentCorpus,
    build_1t_subtoken_documents,
    mean_document_embeddings,
)

__all__ = [
    "MultiRateTaskSpec",
    "MultiRatePredictionTaskSpec",
    "MultiRateTransformer",
    "MultiRateTransformerConfig",
    "build_attention_mask",
    "CoverageAwareInput",
    "AutoFeatureEngineer",
    "SubtokenDocumentCorpus",
    "build_1t_subtoken_documents",
    "mean_document_embeddings",
]
