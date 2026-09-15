"""Cross-instrument attention bounded to one issuer and observation timestamp."""
import torch
from torch import nn


class InstrumentAttention(nn.Module):
    def __init__(self, width, heads, dropout=0.):
        super().__init__()
        self.attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(width)

    def forward(self, states, dates, padding, issuer_ids):
        if issuer_ids is None or issuer_ids.shape != (states.shape[0],):
            raise ValueError('Instrument attention requires one issuer ID per instrument')
        if dates.ndim == 1:
            dates = dates.expand(states.shape[0], -1)
        valid = ~padding if padding is not None else torch.ones(states.shape[:2], device=states.device, dtype=torch.bool)
        indices = valid.flatten().nonzero().flatten()
        if not len(indices):
            return states
        keys = torch.stack((issuer_ids[:, None].expand_as(dates)[valid], dates[valid]), dim=1)
        _, groups, counts = torch.unique(keys, dim=0, return_inverse=True, return_counts=True)
        order = torch.argsort(groups, stable=True)
        ordered_groups = groups[order]
        offsets = counts.cumsum(0) - counts
        slots = torch.arange(len(indices), device=states.device) - torch.repeat_interleave(offsets, counts)
        width = int(counts.max())
        packed = states.new_zeros((len(counts), width, states.shape[-1]))
        blocked = torch.ones((len(counts), width), device=states.device, dtype=torch.bool)
        values = states.reshape(-1, states.shape[-1])[indices[order]]
        packed[ordered_groups, slots] = values
        blocked[ordered_groups, slots] = False
        attended, _ = self.attention(packed, packed, packed, key_padding_mask=blocked, need_weights=False)
        mixed = self.norm(values + attended[ordered_groups, slots])
        return states.reshape(-1, states.shape[-1]).index_copy(0, indices[order], mixed).view_as(states)
