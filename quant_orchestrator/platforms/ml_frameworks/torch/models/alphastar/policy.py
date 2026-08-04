"""AlphaStar-style entity policy for non-SC2 Fleetcraft domains.

This module owns the reusable ML model and training loop. Domain adapters only
prepare episode tensors and labels; they do not own the policy architecture.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from itertools import chain
from pathlib import Path

import torch
from torch import nn
from torch.distributions import Categorical

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


def train_policy_episode_paths(
    episode_paths: Sequence[Path] | Iterable[Path],
    epochs: int,
    learning_rate: float,
    output: Path,
    *,
    action_names: tuple[str, ...] = DEFAULT_ACTION_NAMES,
    gradient_accumulation: int = 8,
    device: str | torch.device | None = None,
) -> dict:
    """Train from cached issuer-year games without loading the corpus in RAM.

    A 1T universe produces one game per issuer-year and can easily exceed RAM
    when every ``[time, family, embedding]`` tensor is materialized at once.
    This loader keeps only one game on the selected device, while retaining a
    small list of paths so every epoch sees the same deterministic corpus.
    """
    paths = [Path(path) for path in episode_paths]
    if not paths:
        raise ValueError("At least one episode path is required")
    if gradient_accumulation < 1:
        raise ValueError("gradient_accumulation must be positive")

    first = torch.load(paths[0], map_location="cpu", weights_only=False)
    feature_size = int(first["entities"].shape[-1])
    instruments = tuple(str(name) for name in first.get("instruments", ("prices",)))
    if not instruments:
        instruments = ("prices",)
    run_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(7)
    model = AlphaStarPolicy(feature_size, int(first["entities"].shape[1]), len(instruments), action_names=action_names).to(run_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    history: list[float] = []
    games = sorted({str(torch.load(path, map_location="cpu", weights_only=False).get("game_id") or path.stem) for path in paths})

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses: list[float] = []
        pending = 0
        for path_index, path in enumerate(paths):
            episode = torch.load(path, map_location="cpu", weights_only=False)
            entities = episode["entities"].to(run_device, non_blocking=True)
            labels = episode["labels"].to(run_device, dtype=torch.long, non_blocking=True)
            targets = episode["instrument_targets"].to(run_device, dtype=torch.long, non_blocking=True)
            if entities.shape[-1] != feature_size:
                raise ValueError(f"{path} has embedding width {entities.shape[-1]}, expected {feature_size}")
            action_logits, instrument_logits = model(entities)
            loss = criterion(action_logits.reshape(-1, len(action_names)), labels.reshape(-1))
            loss = loss + criterion(instrument_logits.reshape(-1, len(instruments)), targets.reshape(-1))
            (loss / gradient_accumulation).backward()
            losses.append(float(loss.detach().cpu()))
            pending += 1
            if pending == gradient_accumulation or path_index == len(paths) - 1:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                pending = 0
            del episode, entities, labels, targets, action_logits, instrument_logits, loss

        epoch_loss = sum(losses) / len(losses)
        history.append(epoch_loss)

    model.eval()
    action_correct = 0
    action_total = 0
    instrument_correct = 0
    instrument_total = 0
    with torch.no_grad():
        for path in paths:
            episode = torch.load(path, map_location="cpu", weights_only=False)
            entities = episode["entities"].to(run_device, non_blocking=True)
            labels = episode["labels"].to(run_device, dtype=torch.long, non_blocking=True)
            targets = episode["instrument_targets"].to(run_device, dtype=torch.long, non_blocking=True)
            action_logits, instrument_logits = model(entities)
            action_correct += int((action_logits.argmax(-1) == labels).sum().cpu())
            action_total += labels.numel()
            instrument_correct += int((instrument_logits.argmax(-1) == targets).sum().cpu())
            instrument_total += targets.numel()

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "games": games,
        "families": list(first["families"]),
        "instruments": list(instruments),
        "actions": action_names,
        "one_instrument_per_manager": True,
        "feature_contract": f"100B_MRL_family_embedding_{feature_size}",
        "loss_history": history,
        "training_mode": "streaming_episode_paths",
        "training_device": str(run_device),
        "gradient_accumulation": gradient_accumulation,
    }, output)
    return {
        "games": len(paths),
        "steps": sum(int(torch.load(path, map_location="cpu", weights_only=False)["entities"].shape[0]) for path in paths),
        "families": int(first["entities"].shape[1]),
        "instruments": list(instruments),
        "epochs": epochs,
        "final_loss": history[-1],
        "action_accuracy": action_correct / max(action_total, 1),
        "instrument_accuracy": instrument_correct / max(instrument_total, 1),
        "artifact": str(output),
        "actions": list(action_names),
        "device": str(run_device),
        "training_mode": "streaming_episode_paths",
    }


def train_policy_episode_paths_rl(
    episode_paths: Sequence[Path] | Iterable[Path],
    epochs: int,
    learning_rate: float,
    output: Path,
    *,
    action_names: tuple[str, ...] = DEFAULT_ACTION_NAMES,
    gradient_accumulation: int = 8,
    gamma: float = 0.99,
    transaction_cost: float = 0.0005,
    entropy_weight: float = 0.01,
    device: str | torch.device | None = None,
) -> dict:
    """Train the policy with REINFORCE on realized Quant-Fleet P&L.

    Each cached issuer-year file is an episode.  At each date every feature
    family samples an action, the portfolio environment updates its one
    position, and the next price return becomes that manager's reward.  No
    hand-written action labels are consumed.
    """
    paths = [Path(path) for path in episode_paths]
    if not paths:
        raise ValueError("At least one episode path is required")
    if gamma <= 0 or gamma > 1:
        raise ValueError("gamma must be in (0, 1]")
    first = torch.load(paths[0], map_location="cpu", weights_only=False)
    feature_size = int(first["entities"].shape[-1])
    family_count = int(first["entities"].shape[1])
    instruments = tuple(str(name) for name in first.get("instruments", ("prices",))) or ("prices",)
    run_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(7)
    model = AlphaStarPolicy(feature_size, family_count, len(instruments), action_names=action_names).to(run_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    action_id = {name: index for index, name in enumerate(action_names)}
    buy = action_id.get("buy", -1); sell = action_id.get("sell", -1)
    short = action_id.get("short", -1); cover = action_id.get("cover", -1)
    hold = action_id.get("hold", action_id.get("do_nothing", -1))
    history: list[float] = []
    total_rewards: list[float] = []

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        episode_losses: list[float] = []
        epoch_rewards: list[float] = []
        pending = 0
        for path_index, path in enumerate(paths):
            episode = torch.load(path, map_location="cpu", weights_only=False)
            if "returns" not in episode:
                raise ValueError(f"{path} has no returns; rebuild episodes for RL training")
            entities = episode["entities"].to(run_device, non_blocking=True)
            returns = episode["returns"].to(run_device, dtype=torch.float32, non_blocking=True).flatten()
            if entities.shape[-1] != feature_size or entities.shape[1] != family_count:
                raise ValueError(f"{path} does not match the first episode shape contract")
            action_logits, _ = model(entities)
            # The last date has no next-period return and is not a decision.
            steps = min(int(returns.shape[0]) - 1, int(action_logits.shape[0]) - 1)
            positions = torch.zeros(family_count, dtype=torch.float32, device=run_device)
            log_probs: list[torch.Tensor] = []
            rewards: list[torch.Tensor] = []
            entropies: list[torch.Tensor] = []
            for time_index in range(max(0, steps)):
                distribution = Categorical(logits=action_logits[time_index])
                actions = distribution.sample()
                previous = positions.clone()
                # Legal one-position portfolio transitions. Opening a long
                # or short is only allowed from flat; closing uses sell or
                # cover. Invalid commands are treated as no-ops.
                positions = torch.where((previous == 0) & (actions == buy), torch.ones_like(positions), positions)
                positions = torch.where((previous == 0) & (actions == short), -torch.ones_like(positions), positions)
                positions = torch.where((previous == 1) & (actions == sell), torch.zeros_like(positions), positions)
                positions = torch.where((previous == -1) & (actions == cover), torch.zeros_like(positions), positions)
                # Invalid close/open commands have no position effect.  A
                # small cost is charged only when exposure changes.
                trade_cost = (positions - previous).abs() * transaction_cost
                reward = positions * returns[time_index] - trade_cost
                log_probs.append(distribution.log_prob(actions))
                rewards.append(reward)
                entropies.append(distribution.entropy())
            if rewards:
                reward_tensor = torch.stack(rewards)
                log_prob_tensor = torch.stack(log_probs)
                entropy_tensor = torch.stack(entropies)
                discounted = torch.zeros_like(reward_tensor)
                running = torch.zeros(family_count, device=run_device)
                for time_index in range(reward_tensor.shape[0] - 1, -1, -1):
                    running = reward_tensor[time_index] + gamma * running
                    discounted[time_index] = running
                advantages = (discounted - discounted.mean(dim=0, keepdim=True))
                advantages = advantages / advantages.std().clamp_min(1e-6)
                loss = -(log_prob_tensor * advantages.detach()).mean() - entropy_weight * entropy_tensor.mean()
                (loss / gradient_accumulation).backward()
                episode_losses.append(float(loss.detach().cpu()))
                epoch_rewards.append(float(reward_tensor.sum().detach().cpu()))
            pending += 1
            if pending == gradient_accumulation or path_index == len(paths) - 1:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                pending = 0
            del episode, entities, returns, action_logits
        history.append(sum(episode_losses) / max(1, len(episode_losses)))
        total_rewards.append(sum(epoch_rewards))

    model.eval()
    output.parent.mkdir(parents=True, exist_ok=True)
    games = sorted({str(torch.load(path, map_location="cpu", weights_only=False).get("game_id") or path.stem) for path in paths})
    torch.save({
        "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "games": games,
        "families": list(first["families"]),
        "instruments": list(instruments),
        "actions": action_names,
        "one_instrument_per_manager": True,
        "feature_contract": f"100B_MRL_family_embedding_{feature_size}",
        "loss_history": history,
        "reward_history": total_rewards,
        "training_mode": "reinforce_pnl",
        "gamma": gamma,
        "transaction_cost": transaction_cost,
        "entropy_weight": entropy_weight,
        "training_device": str(run_device),
        "gradient_accumulation": gradient_accumulation,
    }, output)
    return {
        "games": len(paths),
        "steps": sum(int(torch.load(path, map_location="cpu", weights_only=False)["entities"].shape[0]) for path in paths),
        "families": family_count,
        "instruments": list(instruments),
        "epochs": epochs,
        "final_loss": history[-1],
        "total_reward": total_rewards[-1],
        "artifact": str(output),
        "actions": list(action_names),
        "device": str(run_device),
        "training_mode": "reinforce_pnl",
    }


def train_policy_episode_stream_rl(
    episode_factory: Callable[[], Iterable[dict]],
    epochs: int,
    learning_rate: float,
    output: Path,
    *,
    action_names: tuple[str, ...] = DEFAULT_ACTION_NAMES,
    gradient_accumulation: int = 8,
    gamma: float = 0.99,
    transaction_cost: float = 0.0005,
    entropy_weight: float = 0.01,
    device: str | torch.device | None = None,
) -> dict:
    """Train from a replay factory without writing episode files."""
    if gamma <= 0 or gamma > 1:
        raise ValueError("gamma must be in (0, 1]")
    first = next(iter(episode_factory()))
    feature_size = int(first["entities"].shape[-1]); family_count = int(first["entities"].shape[1])
    instruments = tuple(str(name) for name in first.get("instruments", ("prices",))) or ("prices",)
    run_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(7)
    model = AlphaStarPolicy(feature_size, family_count, len(instruments), action_names=action_names).to(run_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    action_id = {name: index for index, name in enumerate(action_names)}
    buy = action_id.get("buy", -1); sell = action_id.get("sell", -1)
    short = action_id.get("short", -1); cover = action_id.get("cover", -1)
    history: list[float] = []; total_rewards: list[float] = []; game_ids: list[str] = []; total_steps = 0
    for epoch in range(epochs):
        model.train(); optimizer.zero_grad(set_to_none=True); pending = 0; losses = []; epoch_rewards = []
        for episode in episode_factory():
            entities = episode["entities"].to(run_device, non_blocking=True)
            returns = episode["returns"].to(run_device, dtype=torch.float32, non_blocking=True).flatten()
            if entities.shape[-1] != feature_size or entities.shape[1] != family_count:
                raise ValueError("stream episode does not match the first episode shape contract")
            if epoch == 0:
                game_ids.append(str(episode.get("game_id", "stream-episode"))); total_steps += int(entities.shape[0])
            action_logits, _ = model(entities)
            steps = min(int(returns.shape[0]) - 1, int(action_logits.shape[0]) - 1)
            positions = torch.zeros(family_count, dtype=torch.float32, device=run_device)
            log_probs = []; rewards = []; entropies = []
            for time_index in range(max(0, steps)):
                distribution = Categorical(logits=action_logits[time_index]); actions = distribution.sample(); previous = positions.clone()
                positions = torch.where((previous == 0) & (actions == buy), torch.ones_like(positions), positions)
                positions = torch.where((previous == 0) & (actions == short), -torch.ones_like(positions), positions)
                positions = torch.where((previous == 1) & (actions == sell), torch.zeros_like(positions), positions)
                positions = torch.where((previous == -1) & (actions == cover), torch.zeros_like(positions), positions)
                rewards.append(positions * returns[time_index] - (positions - previous).abs() * transaction_cost)
                log_probs.append(distribution.log_prob(actions)); entropies.append(distribution.entropy())
            if rewards:
                reward_tensor = torch.stack(rewards); log_prob_tensor = torch.stack(log_probs); entropy_tensor = torch.stack(entropies)
                discounted = torch.zeros_like(reward_tensor); running = torch.zeros(family_count, device=run_device)
                for time_index in range(reward_tensor.shape[0] - 1, -1, -1):
                    running = reward_tensor[time_index] + gamma * running; discounted[time_index] = running
                advantages = discounted - discounted.mean(dim=0, keepdim=True); advantages /= advantages.std().clamp_min(1e-6)
                loss = -(log_prob_tensor * advantages.detach()).mean() - entropy_weight * entropy_tensor.mean()
                (loss / gradient_accumulation).backward(); losses.append(float(loss.detach().cpu())); epoch_rewards.append(float(reward_tensor.sum().detach().cpu()))
            pending += 1
            if pending == gradient_accumulation:
                optimizer.step(); optimizer.zero_grad(set_to_none=True); pending = 0
            del episode, entities, returns, action_logits
        if pending: optimizer.step(); optimizer.zero_grad(set_to_none=True)
        history.append(sum(losses) / max(1, len(losses))); total_rewards.append(sum(epoch_rewards))
    model.eval(); output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()}, "games": sorted(game_ids), "families": list(first["families"]), "instruments": list(instruments), "actions": action_names, "one_instrument_per_manager": True, "feature_contract": f"100B_MRL_family_embedding_{feature_size}", "loss_history": history, "reward_history": total_rewards, "training_mode": "reinforce_pnl_streaming", "gamma": gamma, "transaction_cost": transaction_cost, "entropy_weight": entropy_weight, "training_device": str(run_device), "gradient_accumulation": gradient_accumulation}, output)
    return {"games": len(game_ids), "steps": total_steps, "families": family_count, "instruments": list(instruments), "epochs": epochs, "final_loss": history[-1], "total_reward": total_rewards[-1], "artifact": str(output), "actions": list(action_names), "device": str(run_device), "training_mode": "reinforce_pnl_streaming"}


def train_policy_episode_stream_rl_batched(
    episode_factory: Callable[[], Iterable[dict]], epochs: int, learning_rate: float, output: Path, *,
    episode_batch_size: int = 16, action_names: tuple[str, ...] = DEFAULT_ACTION_NAMES,
    gradient_accumulation: int = 8, gamma: float = 0.99, transaction_cost: float = 0.0005,
    entropy_weight: float = 0.01, device: str | torch.device | None = None,
    compile_model: bool = False, checkpoint_every_epoch: bool = False, resume_from: Path | None = None,
) -> dict:
    """RL stream trainer with vectorized multi-episode rollouts."""
    first = next(iter(episode_factory()))
    feature_size = int(first["entities"].shape[-1]); family_count = int(first["entities"].shape[1])
    instruments = tuple(str(x) for x in first.get("instruments", ("prices",))) or ("prices",)
    run_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(7)
    model = AlphaStarPolicy(feature_size, family_count, len(instruments), action_names=action_names).to(run_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    start_epoch = 0
    if resume_from is not None and Path(resume_from).exists():
        checkpoint = torch.load(resume_from, map_location=run_device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
    if compile_model and hasattr(torch, "compile"):
        model = torch.compile(model, mode="reduce-overhead")
    ids: list[str] = []; total_steps = 0; losses_history = []; rewards_history = []
    action_id = {name: index for index, name in enumerate(action_names)}
    buy = action_id.get("buy", -1); sell = action_id.get("sell", -1); short = action_id.get("short", -1); cover = action_id.get("cover", -1)
    probe_stream = iter(episode_factory())
    first_stream_episode = next(probe_stream)
    for epoch in range(start_epoch, epochs):
        model.train(); optimizer.zero_grad(set_to_none=True); pending = 0; epoch_losses = []; epoch_rewards = []
        batch = []
        def update(items):
            nonlocal pending, total_steps
            batch_size = len(items); max_steps = max(int(item["entities"].shape[0]) for item in items)
            entities = torch.zeros((max_steps, batch_size, family_count, feature_size), dtype=torch.float32)
            returns = torch.zeros((max_steps, batch_size), dtype=torch.float32); valid = torch.zeros((max_steps, batch_size), dtype=torch.bool)
            for index, item in enumerate(items):
                steps = int(item["entities"].shape[0]); entities[:steps, index] = item["entities"]; returns[:steps, index] = item["returns"].flatten(); valid[:steps, index] = True
                if epoch == 0: ids.append(str(item.get("game_id", "stream-episode"))); total_steps += steps
            logits, _ = model(entities.to(run_device).reshape(max_steps, batch_size * family_count, feature_size))
            distribution = Categorical(logits=logits); actions = distribution.sample(); positions = torch.zeros(batch_size * family_count, device=run_device)
            return_values = returns.to(run_device).unsqueeze(-1).expand(max_steps, batch_size, family_count).reshape(max_steps, batch_size * family_count)
            valid_values = valid.to(run_device).unsqueeze(-1).expand(max_steps, batch_size, family_count).reshape(max_steps, batch_size * family_count)
            log_probs = distribution.log_prob(actions); entropies = distribution.entropy(); rewards = []; previous_positions = []
            for time_index in range(max_steps):
                previous = positions.clone()
                positions = torch.where((previous == 0) & (actions[time_index] == buy), torch.ones_like(positions), positions)
                positions = torch.where((previous == 0) & (actions[time_index] == short), -torch.ones_like(positions), positions)
                positions = torch.where((previous == 1) & (actions[time_index] == sell), torch.zeros_like(positions), positions)
                positions = torch.where((previous == -1) & (actions[time_index] == cover), torch.zeros_like(positions), positions)
                rewards.append((positions * return_values[time_index] - (positions - previous).abs() * transaction_cost) * valid_values[time_index])
                previous_positions.append(positions)
            reward_tensor = torch.stack(rewards); discounted = torch.zeros_like(reward_tensor); running = torch.zeros(batch_size * family_count, device=run_device)
            for time_index in range(max_steps - 1, -1, -1):
                running = reward_tensor[time_index] + gamma * running; discounted[time_index] = running
            mask = valid_values; advantages = discounted - (discounted * mask).sum(dim=0, keepdim=True) / mask.sum(dim=0, keepdim=True).clamp_min(1)
            scale = torch.sqrt(((advantages * mask) ** 2).sum() / mask.sum().clamp_min(1)); advantages = advantages / scale.clamp_min(1e-6)
            loss = -((log_probs * advantages.detach() + entropy_weight * entropies) * mask).sum() / mask.sum().clamp_min(1)
            (loss / gradient_accumulation).backward(); pending += 1; epoch_losses.append(float(loss.detach().cpu())); epoch_rewards.append(float(reward_tensor.sum().detach().cpu()))
            if pending >= gradient_accumulation: optimizer.step(); optimizer.zero_grad(set_to_none=True); pending = 0
        stream = chain((first_stream_episode,), probe_stream) if epoch == start_epoch else episode_factory()
        for item in stream:
            batch.append(item)
            if len(batch) >= max(1, episode_batch_size): update(batch); batch = []
        if batch: update(batch)
        if pending: optimizer.step(); optimizer.zero_grad(set_to_none=True); pending = 0
        losses_history.append(sum(epoch_losses) / max(1, len(epoch_losses))); rewards_history.append(sum(epoch_rewards))
        if checkpoint_every_epoch:
            epoch_path = output.with_name(f"{output.stem}.epoch{epoch + 1}{output.suffix}")
            torch.save({"model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "optimizer_state": optimizer.state_dict(), "epoch": epoch, "families": list(first["families"]), "instruments": list(instruments), "feature_contract": f"100B_MRL_family_embedding_{feature_size}", "training_mode": "reinforce_pnl_streaming_batched"}, epoch_path)
    output.parent.mkdir(parents=True, exist_ok=True); model.eval()
    torch.save({"model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "games": sorted(ids), "families": list(first["families"]), "instruments": list(instruments), "actions": action_names, "one_instrument_per_manager": True, "feature_contract": f"100B_MRL_family_embedding_{feature_size}", "loss_history": losses_history, "reward_history": rewards_history, "training_mode": "reinforce_pnl_streaming_batched", "gamma": gamma, "transaction_cost": transaction_cost, "entropy_weight": entropy_weight, "training_device": str(run_device), "gradient_accumulation": gradient_accumulation, "episode_batch_size": episode_batch_size}, output)
    return {"games": len(ids), "steps": total_steps, "families": family_count, "instruments": list(instruments), "epochs": epochs, "final_loss": losses_history[-1], "total_reward": rewards_history[-1], "artifact": str(output), "actions": list(action_names), "device": str(run_device), "training_mode": "reinforce_pnl_streaming_batched", "episode_batch_size": episode_batch_size, "compiled": compile_model, "resumed_from": str(resume_from) if resume_from else None}
