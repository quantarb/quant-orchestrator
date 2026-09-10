"""Targets for the next observed value in each native-frequency family."""
import torch


def next_observation_targets(values, valid, dates):
    """Skip absent families and same-date rows without exposing the target."""
    batch, length, families = values.shape
    following = torch.full((batch, length + 1, families), length, device=values.device, dtype=torch.long)
    for index in range(length - 1, -1, -1):
        following[:, index] = torch.where(valid[:, index], index, following[:, index + 1])
    later = torch.searchsorted(dates.contiguous(), dates.contiguous(), right=True)
    indices = following.gather(1, later.unsqueeze(-1).expand(-1, -1, families))
    exists = (indices < length) & valid
    target = values.gather(1, indices.clamp_max(length - 1))
    return target, exists
