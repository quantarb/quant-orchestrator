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
            rows = [torch.as_tensor(item[name]) for item in self.batch]
            if len({row.shape for row in rows}) > 1:
                length=max(row.shape[0] for row in rows)
                fill = True if name.endswith('_padding') else torch.iinfo(torch.long).min if name.endswith('_timestamps') else float('nan')
                padded=[]
                for row in rows:
                    prefix=torch.full((length-row.shape[0],*row.shape[1:]),fill,dtype=row.dtype)
                    padded.append(torch.cat([prefix,row]))
                rows=padded
            self.tensors[name] = torch.stack(rows).to(self.device)
        return self.tensors[name]


def supervision_counts(valid, batch):
    """One GPU-to-CPU transfer for a task, instead of one per instrument."""
    counts = valid.sum(dim=1).detach().cpu().tolist()
    by_asset = Counter()
    for item, count in zip(batch, counts):
        by_asset[item['asset_class']] += count
    return sum(counts), by_asset
