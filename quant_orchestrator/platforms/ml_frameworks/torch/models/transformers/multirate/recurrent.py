"""Bounded recurrent token/family state extraction for causal documents."""
import torch

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
