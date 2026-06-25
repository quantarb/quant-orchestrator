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
class BacktestEngine(Protocol):
    """Contract implemented by backtest engine adapters."""

    name: str

    def run(self, strategy: Any, data: Any, **kwargs: Any) -> Any:
        """Run a strategy against data and return an engine-specific result."""


@runtime_checkable
class BrokerAdapter(Protocol):
    """Contract implemented by broker adapters."""

    name: str

    def get_account(self, **kwargs: Any) -> Any:
        """Return account metadata and balances."""

    def get_positions(self, **kwargs: Any) -> Any:
        """Return current broker positions."""

    def submit_order(self, order: Any, **kwargs: Any) -> Any:
        """Submit an order to the broker."""


@runtime_checkable
class ExperimentTracker(Protocol):
    """Contract implemented by experiment and model registry trackers."""

    name: str

    def start_run(
        self,
        *,
        name: str | None = None,
        experiment: str | None = None,
        tags: dict[str, str] | None = None,
        nested: bool = False,
    ) -> Any:
        """Start a tracker run and return the provider-specific run context."""

    def log_params(self, params: dict[str, Any]) -> None:
        """Log run parameters."""

    def log_metrics(self, metrics: dict[str, float], *, step: int | None = None) -> None:
        """Log numeric run metrics."""

    def log_artifact(self, path: str, *, artifact_path: str | None = None) -> None:
        """Log a local artifact file or directory."""

    def register_model(self, model_uri: str, name: str, **kwargs: Any) -> Any:
        """Register a model artifact in the provider-specific model registry."""
