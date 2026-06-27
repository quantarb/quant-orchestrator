"""Compatibility import for the scikit-learn ML framework provider."""

from quant_orchestrator.platforms.ml_frameworks.sklearn.provider import (
    SklearnFramework,
    sklearn_provider,
)

__all__ = ["SklearnFramework", "sklearn_provider"]
