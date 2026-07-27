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
from quant_orchestrator.platforms.ml_frameworks.torch.models.transformers.multirate.target_documents import (
    TargetSubtokenDocumentCorpus,
    build_endpoint_target_subtoken_documents,
    build_event_target_family_subtoken_documents,
    build_earnings_report_subtoken_documents,
    build_analyst_rating_subtoken_documents,
    build_ownership_insider_trading_subtoken_documents,
)
from quant_orchestrator.platforms.ml_frameworks.torch.models.transformers.multirate.entity_embeddings import (
    GlobalEntityEmbedding,
    GlobalEntityVocabulary,
    canonical_entity,
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
    "TargetSubtokenDocumentCorpus",
    "build_endpoint_target_subtoken_documents",
    "build_event_target_family_subtoken_documents",
    "build_earnings_report_subtoken_documents",
    "build_analyst_rating_subtoken_documents",
    "build_ownership_insider_trading_subtoken_documents",
    "GlobalEntityEmbedding",
    "GlobalEntityVocabulary",
    "canonical_entity",
]
