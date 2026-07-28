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
from quant_orchestrator.platforms.ml_frameworks.torch.models.transformers.multirate.text_embeddings import (
    canonical_row_text,
    encode_frozen_text_rows,
)
from quant_orchestrator.platforms.ml_frameworks.torch.models.transformers.multirate.document_axes import (
    CrossSectionalDocument,
    TemporalDocument,
    build_cross_sectional_documents,
    build_temporal_documents,
)
from quant_orchestrator.platforms.ml_frameworks.torch.models.transformers.multirate.temporal_tasks import (
    DOCUMENT_TASK_NAMES,
    FUND_ACTIVITY_SUPERVISED_TASK_NAMES,
    HITS_SUPERVISED_TASK_NAMES,
    ORACLE_SUPERVISED_TASK_NAMES,
    PREDICTION_TASK_NAMES,
    SUPERVISED_TARGET_TASK_NAMES,
    SUBTOKEN_PREDICTION_TASK_NAMES,
    TOKEN_PREDICTION_TASK_NAMES,
    TEMPORAL_MTL_TASK_NAMES,
    TemporalMTLTaskBundle,
    add_subtoken_temporal_tasks,
)
from quant_orchestrator.platforms.ml_frameworks.torch.models.transformers.multirate.multitask import (
    Corpus,
    CorpusTaskGroup,
    Task,
    Trainer,
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
    "canonical_row_text",
    "encode_frozen_text_rows",
    "TemporalDocument",
    "CrossSectionalDocument",
    "build_temporal_documents",
    "build_cross_sectional_documents",
    "DOCUMENT_TASK_NAMES",
    "FUND_ACTIVITY_SUPERVISED_TASK_NAMES",
    "ORACLE_SUPERVISED_TASK_NAMES",
    "HITS_SUPERVISED_TASK_NAMES",
    "SUPERVISED_TARGET_TASK_NAMES",
    "SUBTOKEN_PREDICTION_TASK_NAMES",
    "TOKEN_PREDICTION_TASK_NAMES",
    "PREDICTION_TASK_NAMES",
    "TEMPORAL_MTL_TASK_NAMES",
    "TemporalMTLTaskBundle",
    "add_subtoken_temporal_tasks",
    "Corpus",
    "CorpusTaskGroup",
    "Task",
    "Trainer",
]
