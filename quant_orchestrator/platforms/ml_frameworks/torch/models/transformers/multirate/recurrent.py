"""Bounded recurrent token/family state extraction for causal documents."""
import torch


def identical_context_ids(payload, incoming):
    """Share only identical observations and recurrent state within this step.

    The gathered encoding stays in the autograd graph. No encoded values are
    retained across optimizer updates, and different instrument memories remain
    independent. Caller IDs restrict comparisons to the same issuer/window.
    """
    values = payload['values']
    hints = payload.get('context_ids')
    if hints is None or len(values) < 2:
        return torch.arange(len(values), device=values.device)
    hints = hints.detach().cpu().tolist()
    tensors = [payload[key] for key in ('values', 'dates', 'padding')]
    tensors.extend(payload[key] for key in ('presence', 'modalities') if payload.get(key) is not None)
    tensors.extend((incoming or {}).values())
    representatives, groups = [], []
    for row in range(len(values)):
        for group, first in enumerate(representatives):
            if hints[row] != hints[first]:
                continue
            identical = True
            for tensor in tensors:
                left, right = tensor[row], tensor[first]
                equal = left == right
                if tensor.is_floating_point():
                    equal = equal | (torch.isnan(left) & torch.isnan(right))
                if not bool(equal.all()):
                    identical = False
                    break
            if identical:
                groups.append(group)
                break
        else:
            groups.append(len(representatives))
            representatives.append(row)
    return torch.tensor(groups, dtype=torch.long, device=values.device)

def ending_memory(states, subtokens, presence, padding, incoming=None):
    """Retain the last observed state per family; unchanged families keep memory."""
    valid = presence.bool() & ~padding.unsqueeze(-1)
    valid = valid.clone(); valid[:,0] = False
    positions = torch.arange(states.shape[1],device=states.device)[None,:,None]
    last = torch.where(valid, positions, 0).amax(1)
    family_state = subtokens.gather(1,last[:,None,:,None].expand(-1,1,-1,subtokens.shape[-1])).squeeze(1)
    observed = valid.any(1)
    if incoming is not None:
        family_state = torch.where(observed.unsqueeze(-1),family_state,incoming['subtokens'])
        observed = observed | incoming['presence']
    token_last = last.amax(-1)
    token_state = states[torch.arange(len(states),device=states.device),token_last]
    if incoming is not None:
        token_state = torch.where((token_last>0).unsqueeze(-1),token_state,incoming['states'])
    return {'states':token_state.detach(), 'subtokens':family_state.detach(), 'presence':observed.detach()}
