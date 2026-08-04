"""AlphaStar-style entity policy for non-SC2 Fleetcraft domains.

This module owns the reusable ML model and training loop. Domain adapters only
prepare episode tensors and labels; they do not own the policy architecture.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

DEFAULT_ACTION_NAMES = ("action_0", "action_1", "action_2", "action_3", "action_4", "action_5")


class AlphaStarPolicy(nn.Module):
    """Domain-neutral entity encoder, recurrent core, and discrete heads.

    Any domain can provide embeddings as ``[time, entities, dimensions]`` and
    choose its own action and target vocabularies.
    """

    def __init__(self, feature_size: int, entity_count: int, target_count: int, *, action_names: tuple[str, ...] = DEFAULT_ACTION_NAMES, hidden_size: int = 96):
        super().__init__()
        self.action_names = tuple(action_names)
        self.entity_encoder = nn.Sequential(nn.LayerNorm(feature_size), nn.Linear(feature_size, hidden_size), nn.GELU())
        self.core = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.action_type_head = nn.Linear(hidden_size, len(self.action_names))
        self.target_head = nn.Linear(hidden_size, target_count)

    def forward(self, entities: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # [time, families, features] -> one recurrent manager per family.
        encoded = self.entity_encoder(entities)
        sequence, _ = self.core(encoded.transpose(0, 1))
        sequence = sequence.transpose(0, 1)
        return self.action_type_head(sequence), self.target_head(sequence)


# Compatibility alias for older Fleetcraft checkpoints/adapters.
FleetAlphaStarPolicy = AlphaStarPolicy


def train_policy_episodes(episodes: list[dict], epochs: int, learning_rate: float, output: Path, *, action_names: tuple[str, ...] = DEFAULT_ACTION_NAMES) -> dict:
    """Train and save one shared policy across issuer-year episode tensors."""
    if not episodes:
        raise ValueError("At least one episode is required")
    first = episodes[0]
    family_count = int(first["entities"].shape[1])
    feature_size = int(first["entities"].shape[2])
    instruments = list(first["instruments"])
    torch.manual_seed(7)
    model = AlphaStarPolicy(feature_size, family_count, len(instruments), action_names=action_names)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    history = []
    for _ in range(epochs):
        optimizer.zero_grad()
        losses = []
        for episode in episodes:
            action_logits, instrument_logits = model(episode["entities"])
            action_loss = criterion(action_logits.reshape(-1, len(action_names)), episode["labels"].reshape(-1))
            instrument_loss = criterion(instrument_logits.reshape(-1, len(instruments)), episode["instrument_targets"].reshape(-1))
            losses.append(action_loss + instrument_loss)
        loss = torch.stack(losses).mean()
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    model.eval()
    action_accuracy = []
    instrument_accuracy = []
    with torch.no_grad():
        for episode in episodes:
            action_logits, instrument_logits = model(episode["entities"])
            action_accuracy.append((action_logits.argmax(-1) == episode["labels"]).float().mean())
            instrument_accuracy.append((instrument_logits.argmax(-1) == episode["instrument_targets"]).float().mean())
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "games": [episode.get("game_id") for episode in episodes],
        "families": list(first["families"]),
        "instruments": instruments,
        "actions": action_names,
        "one_instrument_per_manager": True,
        "feature_contract": f"100B_MRL_family_embedding_{feature_size}",
        "loss_history": history,
    }, output)
    return {
        "games": len(episodes),
        "steps": sum(int(episode["entities"].shape[0]) for episode in episodes),
        "families": family_count,
        "instruments": instruments,
        "epochs": epochs,
        "final_loss": history[-1],
        "action_accuracy": float(torch.stack(action_accuracy).mean()),
        "instrument_accuracy": float(torch.stack(instrument_accuracy).mean()),
        "artifact": str(output),
        "actions": list(action_names),
    }
