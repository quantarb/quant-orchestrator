from __future__ import annotations

from collections.abc import Callable, Iterable, MutableMapping
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Protocol, runtime_checkable


ArtifactName = str


class MissingArtifactError(KeyError):
    """Raised when a pipeline stage cannot find required context artifacts."""


@dataclass
class PipelineContext:
    """Mutable artifact store shared by research pipeline stages.

    The context is intentionally lightweight. It keeps native objects in memory
    and lets callers decide when an artifact is worth persisting to MLflow,
    the artifact store, or another external system.
    """

    artifacts: dict[ArtifactName, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def set(self, name: ArtifactName, value: Any) -> PipelineContext:
        self.artifacts[name] = value
        return self

    def update(self, values: MutableMapping[ArtifactName, Any] | None = None, **kwargs: Any) -> PipelineContext:
        if values:
            self.artifacts.update(values)
        if kwargs:
            self.artifacts.update(kwargs)
        return self

    def get(self, name: ArtifactName, default: Any = None) -> Any:
        return self.artifacts.get(name, default)

    def require(self, name: ArtifactName) -> Any:
        try:
            return self.artifacts[name]
        except KeyError as exc:
            available = ", ".join(sorted(self.artifacts))
            raise MissingArtifactError(f"Missing required artifact '{name}'. Available: {available}") from exc

    def require_all(self, names: Iterable[ArtifactName]) -> dict[ArtifactName, Any]:
        return {name: self.require(name) for name in names}

    def has(self, name: ArtifactName) -> bool:
        return name in self.artifacts

    def child(self, **metadata: Any) -> PipelineContext:
        merged_metadata = {**self.metadata, **metadata}
        return PipelineContext(artifacts=dict(self.artifacts), metadata=merged_metadata)


@runtime_checkable
class PipelineStage(Protocol):
    """Contract for an executable research pipeline stage."""

    name: str
    required_inputs: tuple[ArtifactName, ...]
    produced_outputs: tuple[ArtifactName, ...]

    def run(self, context: PipelineContext) -> PipelineContext | None:
        """Read required artifacts, write produced artifacts, and return context."""


@dataclass(frozen=True)
class FunctionStage:
    """Small adapter that turns a Python callable into a pipeline stage."""

    name: str
    function: Callable[[PipelineContext], PipelineContext | MutableMapping[ArtifactName, Any] | None]
    required_inputs: tuple[ArtifactName, ...] = ()
    produced_outputs: tuple[ArtifactName, ...] = ()

    def run(self, context: PipelineContext) -> PipelineContext:
        result = self.function(context)
        if isinstance(result, PipelineContext):
            return result
        if isinstance(result, MutableMapping):
            context.update(result)
        return context


@dataclass(frozen=True)
class StageRun:
    stage: str
    elapsed_seconds: float
    required_inputs: tuple[ArtifactName, ...]
    produced_outputs: tuple[ArtifactName, ...]


@dataclass(frozen=True)
class PipelineResult:
    context: PipelineContext
    stage_runs: tuple[StageRun, ...]


@dataclass(frozen=True)
class Pipeline:
    """Sequential in-memory pipeline for composable research workflows."""

    stages: tuple[PipelineStage, ...]
    name: str = "research_pipeline"

    def __init__(self, stages: Iterable[PipelineStage], *, name: str = "research_pipeline") -> None:
        object.__setattr__(self, "stages", tuple(stages))
        object.__setattr__(self, "name", name)

    def run(self, context: PipelineContext | None = None) -> PipelineResult:
        current = context or PipelineContext()
        stage_runs: list[StageRun] = []
        for stage in self.stages:
            self._validate_inputs(stage, current)
            started = perf_counter()
            next_context = stage.run(current)
            if next_context is not None:
                current = next_context
            elapsed = perf_counter() - started
            self._validate_outputs(stage, current)
            stage_runs.append(
                StageRun(
                    stage=stage.name,
                    elapsed_seconds=elapsed,
                    required_inputs=stage.required_inputs,
                    produced_outputs=stage.produced_outputs,
                )
            )
        current.metadata.setdefault("pipeline_name", self.name)
        current.metadata["pipeline_stage_runs"] = stage_runs
        return PipelineResult(context=current, stage_runs=tuple(stage_runs))

    @staticmethod
    def _validate_inputs(stage: PipelineStage, context: PipelineContext) -> None:
        missing = [name for name in stage.required_inputs if not context.has(name)]
        if missing:
            available = ", ".join(sorted(context.artifacts))
            raise MissingArtifactError(
                f"Stage '{stage.name}' missing required artifacts {missing}. Available: {available}",
            )

    @staticmethod
    def _validate_outputs(stage: PipelineStage, context: PipelineContext) -> None:
        missing = [name for name in stage.produced_outputs if not context.has(name)]
        if missing:
            raise MissingArtifactError(
                f"Stage '{stage.name}' declared outputs {missing} but did not produce them",
            )

