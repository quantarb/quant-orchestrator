import torch
from torch import nn

from quant_orchestrator.platforms.ml_frameworks.torch.models.transformers.multirate.multitask import (
    Corpus,
    Task,
    Trainer,
)


def test_tasks_are_local_to_each_corpus_and_share_one_update():
    model = nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    task_a = Task("a", spec=None)
    task_b = Task("b", spec=None)
    corpus_one = Corpus((torch.tensor([[1.0]]),), name="one")
    corpus_two = Corpus((torch.tensor([[2.0]]),), name="two")
    seen: list[tuple[str, ...]] = []

    trainer = Trainer(
        model,
        [(corpus_one, (task_a, task_b)), (corpus_two, (task_b,))],
        optimizer,
    )

    def step(module, batch, tasks):
        seen.append(tuple(task.name for task in tasks))
        prediction = module(batch[0])
        return {task.name: (prediction - 1.0).square().mean() for task in tasks}

    trainer.fit(epochs=1, step=step)

    assert sorted(seen) == [("a", "b"), ("b",)]
