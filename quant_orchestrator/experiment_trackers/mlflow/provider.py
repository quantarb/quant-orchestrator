from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from quant_orchestrator.platform.contracts import ProviderManifest

DEFAULT_TRACKING_URI = "sqlite:///artifacts/mlflow/mlflow.db"


class MLflowExperimentTracker:
    name = "mlflow"

    def __init__(
        self,
        *,
        tracking_uri: str | None = None,
        registry_uri: str | None = None,
        artifact_location: str | None = None,
    ) -> None:
        self._mlflow = _import_mlflow()
        self.artifact_location = artifact_location
        resolved_tracking_uri = (
            tracking_uri
            or os.getenv("QUANT_ORCHESTRATOR_MLFLOW_TRACKING_URI")
            or os.getenv("MLFLOW_TRACKING_URI")
            or DEFAULT_TRACKING_URI
        )
        _ensure_sqlite_parent(resolved_tracking_uri)
        self._mlflow.set_tracking_uri(resolved_tracking_uri)
        if registry_uri:
            self._mlflow.set_registry_uri(registry_uri)

    def start_run(
        self,
        *,
        name: str | None = None,
        experiment: str | None = None,
        tags: dict[str, str] | None = None,
        nested: bool = False,
    ) -> Any:
        if experiment:
            existing = self._mlflow.get_experiment_by_name(experiment)
            if existing is None and self.artifact_location:
                self._mlflow.create_experiment(
                    experiment,
                    artifact_location=self.artifact_location,
                )
            self._mlflow.set_experiment(experiment)
        return self._mlflow.start_run(run_name=name, tags=tags, nested=nested)

    def log_params(self, params: dict[str, Any]) -> None:
        cleaned = {
            key: _serialize_param(value)
            for key, value in params.items()
            if value is not None
        }
        if cleaned:
            self._mlflow.log_params(cleaned)

    def log_metrics(self, metrics: dict[str, float], *, step: int | None = None) -> None:
        cleaned = {
            key: float(value)
            for key, value in metrics.items()
            if value is not None
        }
        if cleaned:
            self._mlflow.log_metrics(cleaned, step=step)

    def log_artifact(self, path: str, *, artifact_path: str | None = None) -> None:
        artifact = Path(path)
        if artifact.is_dir():
            self._mlflow.log_artifacts(str(artifact), artifact_path=artifact_path)
        else:
            self._mlflow.log_artifact(str(artifact), artifact_path=artifact_path)

    def register_model(self, model_uri: str, name: str, **kwargs: Any) -> Any:
        return self._mlflow.register_model(model_uri=model_uri, name=name, **kwargs)


def _import_mlflow() -> Any:
    try:
        import mlflow
    except ImportError as exc:
        raise ImportError(
            "MLflow tracking requires the tracking extra: "
            "pip install 'quant-orchestrator[tracking]'",
        ) from exc
    return mlflow


def _serialize_param(value: Any) -> str | int | float | bool:
    if isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Path):
        return str(value)
    return repr(value)


def _ensure_sqlite_parent(tracking_uri: str) -> None:
    if not tracking_uri.startswith("sqlite:///"):
        return
    db_path = Path(tracking_uri.removeprefix("sqlite:///")).expanduser()
    db_path.parent.mkdir(parents=True, exist_ok=True)


mlflow_provider = ProviderManifest(
    name="mlflow",
    category="experiment_tracker",
    display_name="MLflow",
    description="Experiment tracking, artifact logging, backtest tracking, and model registry.",
    website="https://mlflow.org/",
    capabilities=("experiments", "artifacts", "metrics", "model_registry"),
    adapters={"default": MLflowExperimentTracker},
)
