"""Targets for the next observed value in each native-frequency family."""
import torch


RECONSTRUCTION_CONTRACT = "family_temporal_cross_feature_v2"


def reconstruction_mask(values, padding, *, widths, generator, observation_probability=0.10,
                        family_probability=0.10, feature_probability=0.10):
    """Hide observations, families, and individual cells in disjoint selections.

    Returns disjoint masks so both masking modes can be audited separately.
    Original missing values and padding never become targets.
    """
    if any(not 0 <= p <= 1 for p in (observation_probability, family_probability, feature_probability)):
        raise ValueError("Mask probabilities must lie in [0, 1]")
    if not widths or min(widths) < 1 or sum(widths) != values.shape[-1]:
        raise ValueError("Family widths must partition the feature vector")
    valid = torch.isfinite(values) & ~padding.unsqueeze(-1)
    whole = torch.rand((*values.shape[:2], 1), device=values.device, generator=generator) < observation_probability
    observations = valid & whole
    family_draws = torch.rand((*values.shape[:2], len(widths)), device=values.device, generator=generator) < family_probability
    family_draws = torch.cat([family_draws[..., i:i+1].expand(*values.shape[:2], width)
                             for i, width in enumerate(widths)], dim=-1)
    families = valid & ~whole & family_draws
    features = valid & ~whole & ~family_draws & (torch.rand(values.shape, device=values.device, generator=generator) < feature_probability)
    return observations | families | features, observations, families, features


def family_channels(values, widths, *, fill=0):
    """Pack every original channel into [batch, date, family, channel]."""
    if not widths or min(widths) < 1 or sum(widths) != values.shape[-1]:
        raise ValueError("Family widths must partition the feature vector")
    return torch.stack([
        torch.nn.functional.pad(part, (0, max(widths) - width), value=fill)
        for part, width in zip(values.split(widths, dim=-1), widths)
    ], dim=-2)


def reconstruction_targets(values, padding, dates, selected, widths):
    """Individual-value targets; token successors preserve a single future row."""
    valid = torch.isfinite(values) & ~padding.unsqueeze(-1)
    clean = torch.nan_to_num(values)
    following, following_valid = next_observation_targets(clean, valid, dates)
    # Find the next strictly later observation, then retain exactly its vector
    # and missingness. Do not combine channels from different future dates.
    rows = torch.arange(values.shape[1], device=values.device).view(1, -1, 1).expand(values.shape[0], -1, -1)
    next_rows, row_valid = next_observation_targets(rows, valid.any(-1, keepdim=True), dates)
    indices = next_rows.expand_as(values)
    token_next = clean.gather(1, indices)
    token_valid = valid.gather(1, indices) & row_valid
    masked_valid = selected & valid
    return {
        "next_subtoken": (family_channels(following, widths), family_channels(following_valid, widths)),
        "next_token": (token_next, token_valid),
        "masked_subtoken": (family_channels(clean, widths), family_channels(masked_valid, widths)),
        "masked_token": (clean, masked_valid),
    }


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
