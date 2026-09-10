"""Causal elapsed-time signals. Explicit dates are Unix nanoseconds."""
import torch

DAY_NS = 86_400_000_000_000


def family_clock(dates, presence, query_dates=None):
    """Log days since previous observation and age, with known-history flags.

    With query_dates, return age of each family's latest available observation.
    Never derive inputs from the next observation date.
    """
    batch, length, families = presence.shape
    dates = dates.expand(batch, -1) if dates.ndim == 1 else dates
    rows = torch.arange(length, device=dates.device).view(1, length, 1)
    latest = torch.where(presence, rows, -1).cummax(dim=1).values
    if query_dates is not None:
        query_dates = query_dates.expand(batch, -1) if query_dates.ndim == 1 else query_dates
        stops = torch.searchsorted(dates.contiguous(), query_dates.contiguous(), right=True) - 1
        indices = latest.gather(1, stops.clamp_min(0).unsqueeze(-1).expand(-1, -1, families))
        known = (indices >= 0) & (stops >= 0).unsqueeze(-1)
        prior_dates = dates.gather(1, indices.clamp_min(0).flatten(1)).reshape_as(indices)
        age = ((query_dates.double().unsqueeze(-1) - prior_dates.double()) / DAY_NS).clamp_min(0)
        return torch.stack((torch.where(known, age.log1p(), 0), known), -1).float()
    previous = torch.cat((torch.full_like(latest[:, :1], -1), latest[:, :-1]), dim=1)
    def elapsed(indices):
        known = indices >= 0
        prior_dates = dates.gather(1, indices.clamp_min(0).flatten(1)).reshape_as(indices)
        days = ((dates.double().unsqueeze(-1) - prior_dates.double()) / DAY_NS).clamp_min(0)
        return torch.where(known, days.log1p(), 0), known
    gap, has_previous = elapsed(previous)
    age, has_observation = elapsed(latest)
    return torch.stack((gap, age, has_previous, has_observation), -1).float()
