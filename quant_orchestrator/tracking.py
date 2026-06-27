from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractContextManager
import os
from pathlib import Path
from typing import Any

DEFAULT_TRACKING_URI = "sqlite:///artifacts/mlflow/mlflow.db"


class MLflowTracker:
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


def get_tracker(
    provider: str = "mlflow",
    *,
    adapter: str = "default",
    **kwargs: Any,
) -> MLflowTracker:
    _validate_provider(provider)
    if str(adapter).strip().lower() != "default":
        raise ValueError("quant-orchestrator uses the default MLflow tracking adapter")
    return MLflowTracker(**kwargs)


def start_tracked_run(
    *,
    run_name: str,
    experiment: str,
    provider: str = "mlflow",
    tags: Mapping[str, str] | None = None,
    params: Mapping[str, Any] | None = None,
    metrics: Mapping[str, float] | None = None,
    artifacts: Mapping[str, str | Path] | None = None,
    nested: bool = False,
    **tracker_kwargs: Any,
) -> AbstractContextManager[Any]:
    tracker = get_tracker(provider=provider, **tracker_kwargs)
    run_context = tracker.start_run(
        name=run_name,
        experiment=experiment,
        tags=dict(tags or {}),
        nested=nested,
    )
    return _TrackedRunContext(
        tracker=tracker,
        run_context=run_context,
        params=dict(params or {}),
        metrics=dict(metrics or {}),
        artifacts=dict(artifacts or {}),
    )


def log_backtest_run(
    *,
    run_name: str,
    engine: str,
    strategy: str,
    data_source: str,
    params: Mapping[str, Any] | None = None,
    metrics: Mapping[str, float] | None = None,
    artifacts: Mapping[str, str | Path] | None = None,
    experiment: str = "backtests",
    provider: str = "mlflow",
    **tracker_kwargs: Any,
) -> None:
    tags = {
        "quant_orchestrator.kind": "backtest",
        "quant_orchestrator.engine": engine,
        "quant_orchestrator.strategy": strategy,
        "quant_orchestrator.data_source": data_source,
    }
    with start_tracked_run(
        run_name=run_name,
        experiment=experiment,
        provider=provider,
        tags=tags,
        params=params,
        metrics=metrics,
        artifacts=artifacts,
        **tracker_kwargs,
    ):
        pass


def log_model_run(
    *,
    run_name: str,
    framework: str,
    dataset: str,
    params: Mapping[str, Any] | None = None,
    metrics: Mapping[str, float] | None = None,
    artifacts: Mapping[str, str | Path] | None = None,
    model_uri: str | None = None,
    registered_model_name: str | None = None,
    experiment: str = "models",
    provider: str = "mlflow",
    **tracker_kwargs: Any,
) -> Any:
    tags = {
        "quant_orchestrator.kind": "model",
        "quant_orchestrator.framework": framework,
        "quant_orchestrator.dataset": dataset,
    }
    tracker = get_tracker(provider=provider, **tracker_kwargs)
    with tracker.start_run(name=run_name, experiment=experiment, tags=tags):
        tracker.log_params(dict(params or {}))
        tracker.log_metrics(dict(metrics or {}))
        _log_artifacts(tracker, dict(artifacts or {}))
        if model_uri and registered_model_name:
            return tracker.register_model(model_uri, registered_model_name)
    return None


class _TrackedRunContext:
    def __init__(
        self,
        *,
        tracker: MLflowTracker,
        run_context: AbstractContextManager[Any],
        params: dict[str, Any],
        metrics: dict[str, float],
        artifacts: dict[str, str | Path],
    ) -> None:
        self.tracker = tracker
        self.run_context = run_context
        self.params = params
        self.metrics = metrics
        self.artifacts = artifacts
        self.run: Any = None

    def __enter__(self) -> Any:
        self.run = self.run_context.__enter__()
        self.tracker.log_params(self.params)
        self.tracker.log_metrics(self.metrics)
        _log_artifacts(self.tracker, self.artifacts)
        return self.run

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool | None:
        return self.run_context.__exit__(exc_type, exc, traceback)


def _log_artifacts(tracker: MLflowTracker, artifacts: Mapping[str, str | Path]) -> None:
    for artifact_path, local_path in artifacts.items():
        tracker.log_artifact(str(local_path), artifact_path=artifact_path)


def _validate_provider(provider: str) -> None:
    if str(provider).strip().lower() != "mlflow":
        raise ValueError("quant-orchestrator uses MLflow for experiment tracking")


def _import_mlflow() -> Any:
    try:
        import mlflow
    except ImportError as exc:
        raise ImportError("quant-orchestrator requires MLflow for experiment tracking") from exc
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
