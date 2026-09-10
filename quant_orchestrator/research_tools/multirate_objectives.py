"""Targets for the next observed value in each native-frequency family."""
import torch


RECONSTRUCTION_CONTRACT = "features_only_time_v6"


def feature_family_layout(columns):
    """Keep contiguous raw endpoint fields together in one family adapter."""
    layout = {}
    previous = None
    for column in columns:
        family = column.rsplit('.', 1)[0] if '.' in column else column
        if family != previous and family in layout:
            raise ValueError(f"Feature family {family} is not contiguous")
        layout[family] = layout.get(family, 0) + 1
        previous = family
    if not layout:
        raise ValueError("At least one input feature is required")
    return layout


def reconstruction_mask(values, padding, *, widths, generator,
                        family_probability=0.15, feature_probability=0.15):
    """Independent corruption views: features within families, or whole families.

    Feature corruption always retains an observed sibling in that family.
    Naturally absent values are never targets in either view.
    """
    if any(not 0 <= p <= 1 for p in (family_probability, feature_probability)):
        raise ValueError("Mask probabilities must lie in [0, 1]")
    if not widths or min(widths) < 1 or sum(widths) != values.shape[-1]:
        raise ValueError("Family widths must partition the feature vector")
    valid = torch.isfinite(values) & ~padding.unsqueeze(-1)
    feature_parts, family_parts = [], []
    for local in valid.split(widths, dim=-1):
        selected = local & (torch.rand(local.shape, device=values.device, generator=generator) < feature_probability)
        # Never turn within-family masking into whole-family masking.
        all_hidden = selected.sum(-1) == local.sum(-1)
        first_visible = local.to(torch.int64).argmax(-1, keepdim=True)
        keep = torch.zeros_like(local).scatter_(-1, first_visible, True) & all_hidden.unsqueeze(-1)
        feature_parts.append(selected & ~keep)
        entire = torch.rand((*values.shape[:2], 1), device=values.device, generator=generator) < family_probability
        family_parts.append(local & entire)
    return torch.cat(feature_parts, -1), torch.cat(family_parts, -1)


def family_channels(values, widths, *, fill=0):
    """Pack every original channel into [batch, date, family, channel]."""
    if not widths or min(widths) < 1 or sum(widths) != values.shape[-1]:
        raise ValueError("Family widths must partition the feature vector")
    return torch.stack([
        torch.nn.functional.pad(part, (0, max(widths) - width), value=fill)
        for part, width in zip(values.split(widths, dim=-1), widths)
    ], dim=-2)


def reconstruction_targets(values, padding, dates, selected, widths, *, family_selected=None):
    """Individual-value targets; token successors preserve a single future row."""
    valid = torch.isfinite(values) & ~padding.unsqueeze(-1)
    clean = torch.nan_to_num(values)
    # A family successor is one coherent observation, not a mixture of
    # individual fields taken from different future dates.
    rows = torch.arange(values.shape[1], device=values.device).view(1, -1, 1).expand(values.shape[0], -1, -1)
    following_parts, valid_parts = [], []
    for local, local_valid in zip(clean.split(widths, -1), valid.split(widths, -1)):
        next_rows, exists = next_observation_targets(rows, local_valid.any(-1, keepdim=True), dates)
        indices = next_rows.expand_as(local)
        following_parts.append(local.gather(1, indices))
        valid_parts.append(local_valid.gather(1, indices) & exists)
    following = torch.cat(following_parts, -1)
    following_valid = torch.cat(valid_parts, -1)
    # Find the next strictly later observation, then retain exactly its vector
    # and missingness. Do not combine channels from different future dates.
    rows = torch.arange(values.shape[1], device=values.device).view(1, -1, 1).expand(values.shape[0], -1, -1)
    next_rows, row_valid = next_observation_targets(rows, valid.any(-1, keepdim=True), dates)
    indices = next_rows.expand_as(values)
    token_next = clean.gather(1, indices)
    token_valid = valid.gather(1, indices) & row_valid
    masked_valid = selected & valid
    token_masked_valid = (family_selected & valid) if family_selected is not None else torch.zeros_like(valid)
    return {
        "next_subtoken": (family_channels(following, widths), family_channels(following_valid, widths)),
        "next_token": (token_next, token_valid),
        "masked_subtoken": (family_channels(clean, widths), family_channels(masked_valid, widths)),
        "masked_token": (clean, token_masked_valid),
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
