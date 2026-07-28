"""Generic corpus/task routing and shared-gradient training primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class Task:
    """A named task attached to a model head or another task implementation."""

    name: str
    spec: Any
    loss_weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("task name must not be empty")
        if self.loss_weight < 0:
            raise ValueError("task loss weight must be non-negative")


@dataclass(frozen=True)
class Corpus:
    """A reusable sequence of rows that can be assigned to many task lists."""

    rows: Sequence[Any]
    name: str = "corpus"
    batch_size: int = 1

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("corpus batch_size must be at least one")

    def __len__(self) -> int:
        return len(self.rows)

    def batches(self, *, seed: int, epoch: int) -> Iterator[list[Any]]:
        order = np.random.default_rng(seed + epoch).permutation(len(self.rows))
        for start in range(0, len(order), self.batch_size):
            yield [self.rows[index] for index in order[start:start + self.batch_size]]


CorpusTaskGroup = tuple[Corpus, tuple[Task, ...]]
LossStep = Callable[[nn.Module, list[Any], tuple[Task, ...]], Mapping[str, torch.Tensor]]
EpochEnd = Callable[[int, float], bool]


class Trainer:
    """Train one model with task activation scoped to each corpus group.

    ``step`` computes one loss per active task.  The trainer sums those losses
    and performs exactly one backward/optimizer update for the whole group.
    A task may appear in multiple corpus groups without changing activation of
    tasks in any other group.
    """

    def __init__(
        self,
        model: nn.Module,
        corpus_tasks: Sequence[CorpusTaskGroup],
        optimizer: torch.optim.Optimizer,
        *,
        grad_accumulation_steps: int = 1,
        seed: int = 0,
    ) -> None:
        if not corpus_tasks:
            raise ValueError("at least one corpus/task group is required")
        if grad_accumulation_steps < 1:
            raise ValueError("grad_accumulation_steps must be at least one")
        self.model = model
        self.corpus_tasks = tuple(corpus_tasks)
        self.optimizer = optimizer
        self.grad_accumulation_steps = grad_accumulation_steps
        self.seed = seed
        self.current_epoch = 0
        self.current_step = 0
        self._validate_groups()

    def _validate_groups(self) -> None:
        for corpus, tasks in self.corpus_tasks:
            if not tasks:
                raise ValueError(f"corpus {corpus.name!r} has no tasks")
            names = [task.name for task in tasks]
            if len(names) != len(set(names)):
                raise ValueError(f"corpus {corpus.name!r} contains duplicate tasks")

    def batches(self, epoch: int) -> Iterator[tuple[list[Any], tuple[Task, ...]]]:
        grouped = [
            [(batch, tasks) for batch in corpus.batches(seed=self.seed, epoch=epoch)]
            for corpus, tasks in self.corpus_tasks
        ]
        pending = [item for group in grouped for item in group]
        order = np.random.default_rng(self.seed + epoch).permutation(len(pending))
        for index in order:
            yield pending[index]

    def fit(self, epochs: int, step: LossStep, *, on_epoch_end: EpochEnd | None = None) -> list[float]:
        """Run shared-gradient training and return mean loss per epoch."""
        losses: list[float] = []
        for epoch in range(epochs):
            self.current_epoch = epoch
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            total = 0.0
            count = 0
            epoch_batches = list(self.batches(epoch))
            for step_index, (batch, tasks) in enumerate(epoch_batches):
                self.current_step = step_index
                task_losses = step(self.model, batch, tasks)
                missing = {task.name for task in tasks} - set(task_losses)
                if missing:
                    raise ValueError(f"step did not return losses for tasks: {sorted(missing)}")
                loss = self.backward_step(task_losses, tasks, step_index, step_index + 1 == len(epoch_batches))
                total += float(loss.detach())
                count += 1
            losses.append(total / max(1, count))
            if on_epoch_end is not None and on_epoch_end(epoch, losses[-1]):
                break
        return losses

    def backward_step(
        self,
        task_losses: Mapping[str, torch.Tensor],
        tasks: Sequence[Task],
        step_index: int,
        last: bool,
    ) -> torch.Tensor:
        """Sum local task losses and perform one shared optimizer update."""
        missing = {task.name for task in tasks} - set(task_losses)
        if missing:
            raise ValueError(f"missing task losses: {sorted(missing)}")
        loss = sum(task.loss_weight * task_losses[task.name] for task in tasks)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss for tasks {[task.name for task in tasks]}")
        (loss / self.grad_accumulation_steps).backward()
        if (step_index + 1) % self.grad_accumulation_steps == 0 or last:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
        return loss


__all__ = ["Corpus", "CorpusTaskGroup", "Task", "Trainer"]
