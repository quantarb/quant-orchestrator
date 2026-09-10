"""Bounded, per-step tensor staging and batched diagnostic transfers."""
from collections import Counter
import torch


class BatchTensors:
    """Stage each raw field once. Callers must clone before mutating a field."""
    def __init__(self, batch, device):
        self.batch = batch
        self.device = device
        self.tensors = {}

    def __call__(self, name):
        if name not in self.tensors:
            self.tensors[name] = torch.stack([torch.as_tensor(item[name]) for item in self.batch]).to(self.device)
        return self.tensors[name]


def supervision_counts(valid, batch):
    """One GPU-to-CPU transfer for a task, instead of one per instrument."""
    counts = valid.sum(dim=1).detach().cpu().tolist()
    by_asset = Counter()
    for item, count in zip(batch, counts):
        by_asset[item['asset_class']] += count
    return sum(counts), by_asset
