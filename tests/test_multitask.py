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


def test_resume_keeps_batch_order_and_optimizer_updates(tmp_path):
    import copy
    import torch
    from quant_orchestrator.platforms.ml_frameworks.torch.models.transformers.multirate.multitask import Corpus,Task,Trainer
    rows=list(range(8));task=Task('loss',spec=None);seen=[]
    initial=torch.nn.Linear(1,1)
    def setup(model):
        optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
        return Trainer(model,[(Corpus(rows,batch_size=2),(task,))],optimizer,seed=5),optimizer
    reference,refopt=setup(copy.deepcopy(initial))
    def step(model,batch,tasks):
        x=torch.tensor(batch,dtype=torch.float32).reshape(-1,1)
        return {'loss':model(x).square().mean()}
    reference.fit(2,step)
    interrupted,opt=setup(copy.deepcopy(initial));snapshot={}
    class Stop(Exception):pass
    def capture(epoch,batch,total,loss):
        if epoch==1 and batch==2:
            snapshot.update(model=copy.deepcopy(interrupted.model.state_dict()),optimizer=copy.deepcopy(opt.state_dict()))
            raise Stop
    import pytest
    with pytest.raises(Stop):interrupted.fit(2,step,on_batch_end=capture)
    resumed,resopt=setup(copy.deepcopy(initial));resumed.model.load_state_dict(snapshot['model']);resopt.load_state_dict(snapshot['optimizer'])
    resumed.fit(2,step,start_epoch=1,start_batch=2,on_batch_end=lambda e,b,*_:seen.append((e,b)))
    assert seen==[(1,3),(1,4)]
    for a,b in zip(reference.model.parameters(),resumed.model.parameters()):torch.testing.assert_close(a,b)
