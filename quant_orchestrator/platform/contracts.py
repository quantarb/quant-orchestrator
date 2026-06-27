from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class ProviderManifest:
    """Metadata for an installable Quant Orchestrator provider."""

    name: str
    category: str
    display_name: str
    description: str = ""
    website: str | None = None
    version: str | None = None
    capabilities: tuple[str, ...] = ()
    adapters: dict[str, type[Any]] = field(default_factory=dict)


@runtime_checkable
class MLFramework(Protocol):
    """Contract implemented by ML framework adapters."""

    name: str

    def fit(self, dataset: Any, **kwargs: Any) -> Any:
        """Train a model or pipeline from an orchestrator dataset."""

    def predict(self, model: Any, dataset: Any, **kwargs: Any) -> Any:
        """Return predictions for an orchestrator dataset."""


@runtime_checkable
class BacktestingFramework(Protocol):
    """Contract implemented by backtesting framework adapters."""

    name: str

    def run(self, strategy: Any, data: Any, **kwargs: Any) -> Any:
        """Run a strategy against data and return an engine-specific result."""


# Backward-compatible name for older research code.
BacktestEngine = BacktestingFramework


