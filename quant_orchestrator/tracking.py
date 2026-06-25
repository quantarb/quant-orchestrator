from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

from quant_orchestrator.platform import registry
from quant_orchestrator.platform.contracts import ExperimentTracker


def get_tracker(
    provider: str = "mlflow",
    *,
    adapter: str = "default",
    **kwargs: Any,
) -> ExperimentTracker:
    tracker_cls = registry.adapter("experiment_tracker", provider, adapter)
    return tracker_cls(**kwargs)


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
    tracker = get_tracker(provider, **tracker_kwargs)
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
    tracker = get_tracker(provider, **tracker_kwargs)
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
        tracker: ExperimentTracker,
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


def _log_artifacts(tracker: ExperimentTracker, artifacts: Mapping[str, str | Path]) -> None:
    for artifact_path, local_path in artifacts.items():
        tracker.log_artifact(str(local_path), artifact_path=artifact_path)
