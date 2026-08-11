"""AlphaStar-style entity policy for non-SC2 Fleetcraft domains.

This module owns the reusable ML model and training loop. Domain adapters only
prepare episode tensors and labels; they do not own the policy architecture.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from itertools import chain
from pathlib import Path

import torch
from torch import nn
from torch.distributions import Categorical

DEFAULT_ACTION_NAMES = ("action_0", "action_1", "action_2", "action_3", "action_4", "action_5")
DEFAULT_ALLOCATION_BUCKETS = (0.10, 0.25, 0.50, 0.75, 1.00)
PORTFOLIO_INITIAL_BALANCE = 100_000.0


class AlphaStarPolicy(nn.Module):
    """Domain-neutral entity encoder, recurrent core, and discrete heads.

    Any domain can provide embeddings as ``[time, entities, dimensions]`` and
    choose its own action and target vocabularies.
    """

    def __init__(
        self,
        feature_size: int,
        entity_count: int,
        target_count: int,
        *,
        action_names: tuple[str, ...] = DEFAULT_ACTION_NAMES,
        hidden_size: int = 96,
        family_specific_heads: bool = True,
        allocation_buckets: tuple[float, ...] = DEFAULT_ALLOCATION_BUCKETS,
        portfolio_state_size: int = 5,
    ):
        super().__init__()
        self.action_names = tuple(action_names)
        self.entity_count = entity_count
        self.family_specific_heads = family_specific_heads
        self.allocation_buckets = tuple(float(value) for value in allocation_buckets)
        self.portfolio_state_size = int(portfolio_state_size)
        self.entity_encoder = nn.Sequential(nn.LayerNorm(feature_size), nn.Linear(feature_size, hidden_size), nn.GELU())
        self.portfolio_state_encoder = nn.Sequential(nn.LayerNorm(self.portfolio_state_size), nn.Linear(self.portfolio_state_size, hidden_size), nn.GELU())
        self.core = nn.GRU(hidden_size, hidden_size, batch_first=True)
        if family_specific_heads:
            self.family_embedding = nn.Embedding(entity_count, hidden_size)
            self.action_type_heads = nn.ModuleList(nn.Linear(hidden_size, len(self.action_names)) for _ in range(entity_count))
            self.target_heads = nn.ModuleList(nn.Linear(hidden_size, target_count) for _ in range(entity_count))
            # One scalar is converted through sigmoid into a continuous
            # fraction of available capital.
            self.allocation_heads = nn.ModuleList(nn.Linear(hidden_size, 1) for _ in range(entity_count))
        else:
            # Load compatibility for checkpoints produced before family heads
            # were introduced.
            self.action_type_head = nn.Linear(hidden_size, len(self.action_names))
            self.target_head = nn.Linear(hidden_size, target_count)
            self.allocation_head = nn.Linear(hidden_size, 1)

    def forward(self, entities: torch.Tensor, *, portfolio_state: torch.Tensor | None = None, return_all: bool = False):
        # [time, families, features] (or [time, batch*families, features])
        # -> one recurrent manager per family. Flattened batches repeat the
        # family order, so the same family-specific head is used for each
        # issuer-year game.
        if entities.ndim != 3 or (self.family_specific_heads and entities.shape[1] % self.entity_count != 0):
            raise ValueError("entities must have shape [time, batch*families, features]")
        encoded = self.entity_encoder(entities)
        if portfolio_state is None:
            portfolio_state = torch.zeros((*entities.shape[:2], self.portfolio_state_size), device=entities.device, dtype=entities.dtype)
        if portfolio_state.shape[:2] != entities.shape[:2] or portfolio_state.shape[-1] != self.portfolio_state_size:
            raise ValueError("portfolio_state must have shape [time, entities, portfolio_state_size]")
        encoded = encoded + self.portfolio_state_encoder(portfolio_state)
        if self.family_specific_heads:
            family_ids = torch.arange(self.entity_count, device=entities.device).repeat(entities.shape[1] // self.entity_count)
            encoded = encoded + self.family_embedding(family_ids).unsqueeze(0)
        sequence, _ = self.core(encoded.transpose(0, 1))
        sequence = sequence.transpose(0, 1)
        if not self.family_specific_heads:
            action_logits = self.action_type_head(sequence)
            target_logits = self.target_head(sequence)
            allocation_logits = self.allocation_head(sequence)
            return (action_logits, target_logits, allocation_logits) if return_all else (action_logits, target_logits)
        action_logits = torch.empty((sequence.shape[0], sequence.shape[1], len(self.action_names)), device=sequence.device, dtype=sequence.dtype)
        target_logits = torch.empty((sequence.shape[0], sequence.shape[1], self.target_heads[0].out_features), device=sequence.device, dtype=sequence.dtype)
        allocation_logits = torch.empty((sequence.shape[0], sequence.shape[1], 1), device=sequence.device, dtype=sequence.dtype)
        for family_index, (action_head, target_head) in enumerate(zip(self.action_type_heads, self.target_heads)):
            family_sequence = sequence[:, family_index::self.entity_count]
            action_logits[:, family_index::self.entity_count] = action_head(family_sequence)
            target_logits[:, family_index::self.entity_count] = target_head(family_sequence)
            allocation_logits[:, family_index::self.entity_count] = self.allocation_heads[family_index](family_sequence)
        return (action_logits, target_logits, allocation_logits) if return_all else (action_logits, target_logits)

    @staticmethod
    def allocation_value(raw: torch.Tensor) -> torch.Tensor:
        """Map the continuous sizing head to a safe 1%-100% allocation."""
        return raw.sigmoid().clamp(0.01, 1.0)


@dataclass(frozen=True)
class TradingMechanics:
    """One-position-per-manager portfolio rules used by every RL trainer."""

    action_names: tuple[str, ...]
    transaction_cost: float = 0.0005
    slippage: float = 0.0001
    risk_penalty: float = 0.01
    volatility_floor: float = 0.005

    def __post_init__(self) -> None:
        object.__setattr__(self, "_ids", {name: index for index, name in enumerate(self.action_names)})

    def _id(self, name: str) -> int:
        return int(getattr(self, "_ids").get(name, -1))

    def legal_action_mask(self, positions: torch.Tensor, cash: torch.Tensor | None = None, *, spawn_cost: float = 1000.0) -> torch.Tensor:
        """Return [managers, actions] mask for the current positions."""
        mask = torch.zeros((*positions.shape, len(self.action_names)), dtype=torch.bool, device=positions.device)
        hold = self._id("hold")
        noop = self._id("do_nothing")
        if hold >= 0:
            mask[..., hold] = True
        if noop >= 0:
            mask[..., noop] = True
        flat = positions == 0
        long = positions > 0
        short_position = positions < 0
        buy = self._id("buy")
        sell = self._id("sell")
        short = self._id("short")
        cover = self._id("cover")
        if buy >= 0:
            mask[..., buy] |= flat
        if short >= 0:
            mask[..., short] |= flat
        if sell >= 0:
            mask[..., sell] |= long
        if cover >= 0:
            mask[..., cover] |= short_position
        spawn = self._id("spawn")
        if spawn >= 0:
            mask[..., spawn] = True if cash is None else cash >= float(spawn_cost)
        # Generic domains may not expose hold/do_nothing. Keep them usable.
        if not mask.any(dim=-1).all():
            mask[..., 0] = True
        return mask

    def transition(
        self,
        positions: torch.Tensor,
        actions: torch.Tensor,
        returns: torch.Tensor,
        volatility: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        previous = positions
        buy = self._id("buy")
        sell = self._id("sell")
        short = self._id("short")
        cover = self._id("cover")
        positions = torch.where((previous == 0) & (actions == buy), torch.ones_like(previous), previous)
        positions = torch.where((previous == 0) & (actions == short), -torch.ones_like(previous), positions)
        positions = torch.where((previous > 0) & (actions == sell), torch.zeros_like(previous), positions)
        positions = torch.where((previous < 0) & (actions == cover), torch.zeros_like(previous), positions)
        traded = (positions - previous).abs()
        trading_cost = traded * (self.transaction_cost + self.slippage)
        reward = positions * returns - trading_cost - self.risk_penalty * positions.square() * volatility
        return positions, reward

    @staticmethod
    def portfolio_return_reward(manager_reward: torch.Tensor, family_count: int, *, initial_balance: float = PORTFOLIO_INITIAL_BALANCE) -> torch.Tensor:
        """Convert manager rewards into the dashboard's portfolio-return objective.

        The dashboard defines return as total P&L divided by the fixed initial
        portfolio. Every feature-family ape receives the same aggregate
        portfolio reward for credit assignment, rather than optimizing an
        isolated per-ape return.
        """
        if family_count < 1 or manager_reward.numel() % family_count:
            raise ValueError("manager reward does not contain complete family groups")
        grouped = manager_reward.reshape(-1, family_count).mean(dim=-1, keepdim=True) / float(initial_balance)
        return grouped.expand_as(manager_reward.reshape(-1, family_count)).reshape_as(manager_reward)

    @staticmethod
    def benchmark_relative_reward(manager_reward: torch.Tensor, benchmark_return: torch.Tensor, family_count: int) -> torch.Tensor:
        """Reward portfolio outperformance versus absolute stock YTD movement.

        ``benchmark_return`` is the change in absolute stock YTD return for
        the current date. Summing these per-date differences produces the
        desired direction-aware objective: a portfolio must beat a +700%
        rally, while a +30% portfolio return during a -30% stock year meets
        the hurdle and anything above it is rewarded.
        """
        grouped = manager_reward.reshape(-1, family_count).mean(dim=-1)
        benchmark = benchmark_return.reshape(-1).to(grouped)
        if benchmark.numel() != grouped.numel():
            raise ValueError("benchmark return does not match portfolio groups")
        relative = (grouped - benchmark).unsqueeze(-1)
        return relative.expand(-1, family_count).reshape_as(manager_reward)


def historical_volatility(returns: torch.Tensor, floor: float = 0.005) -> torch.Tensor:
    """Estimate volatility using only returns observed before each decision."""
    squared = returns.square()
    prior_sum = torch.cat((torch.zeros_like(squared[:1]), torch.cumsum(squared[:-1], dim=0)))
    count = torch.arange(returns.shape[0], device=returns.device, dtype=returns.dtype).clamp_min(1)
    return (prior_sum / count.reshape(-1, *([1] * (returns.ndim - 1)))).sqrt().clamp_min(floor)


def masked_distribution(logits: torch.Tensor, mask: torch.Tensor) -> Categorical:
    """Build a policy distribution that cannot sample invalid commands."""
    return Categorical(logits=logits.masked_fill(~mask, torch.finfo(logits.dtype).min))


def hierarchical_target_distribution(
    target_logits: torch.Tensor,
    active_targets: torch.Tensor,
    target_available: torch.Tensor | None = None,
) -> Categorical:
    """Select an instrument before selecting an action for each manager.

    ``active_targets`` contains ``-1`` for flat managers. Flat managers may
    select any currently available instrument; managers with a position are
    locked to their existing instrument until that position is closed.
    """
    if target_logits.ndim != 2 or active_targets.ndim != 1 or target_logits.shape[0] != active_targets.shape[0]:
        raise ValueError("target_logits must be [managers, instruments] and active_targets [managers]")
    instrument_count = target_logits.shape[-1]
    mask = torch.ones_like(target_logits, dtype=torch.bool)
    if target_available is not None:
        if target_available.shape != target_logits.shape:
            raise ValueError("target_available must match target_logits")
        mask &= target_available
    locked = active_targets >= 0
    if locked.any():
        locked_mask = torch.zeros_like(mask)
        locked_mask[locked, active_targets[locked].clamp(0, instrument_count - 1)] = True
        mask[locked] = locked_mask[locked]
    if not mask.any(dim=-1).all():
        # An expired/missing instrument must not make the distribution invalid.
        mask[~mask.any(dim=-1), 0] = True
    return masked_distribution(target_logits, mask)


def transition_active_targets(
    active_targets: torch.Tensor,
    positions: torch.Tensor,
    actions: torch.Tensor,
    targets: torch.Tensor,
    mechanics: "TradingMechanics",
) -> torch.Tensor:
    """Update one-instrument-per-manager locks after an action executes."""
    next_targets = active_targets.clone()
    flat = positions == 0
    buy_or_short = (actions == mechanics._id("buy")) | (actions == mechanics._id("short"))
    next_targets = torch.where(flat & buy_or_short, targets, next_targets)
    closes = ((positions > 0) & (actions == mechanics._id("sell"))) | ((positions < 0) & (actions == mechanics._id("cover")))
    next_targets = torch.where(closes, torch.full_like(next_targets, -1), next_targets)
    return next_targets


class AlphaStarValueNetwork(nn.Module):
    """Local PPO value function or centralized MAPPO portfolio critic."""

    def __init__(self, feature_size: int, entity_count: int, *, hidden_size: int = 96, centralized: bool = False):
        super().__init__()
        self.entity_count = entity_count
        self.centralized = centralized
        self.encoder = nn.Sequential(nn.LayerNorm(feature_size), nn.Linear(feature_size, hidden_size), nn.GELU())
        self.core = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, entities: torch.Tensor) -> torch.Tensor:
        if entities.ndim != 3:
            raise ValueError("entities must have shape [time, entities, features]")
        encoded = self.encoder(entities)
        if self.centralized:
            if entities.shape[1] % self.entity_count != 0:
                raise ValueError("centralized critic requires batch*families entities")
            batch_size = entities.shape[1] // self.entity_count
            pooled = encoded.reshape(entities.shape[0], batch_size, self.entity_count, -1).mean(dim=2)
            sequence, _ = self.core(pooled.transpose(0, 1))
            return self.value_head(sequence.transpose(0, 1)).squeeze(-1)
        sequence, _ = self.core(encoded.transpose(0, 1))
        return self.value_head(sequence.transpose(0, 1)).squeeze(-1)


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
        "architecture": "shared_trunk_family_heads_gru",
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
        "architecture": "shared_trunk_family_heads_gru",
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
    mechanics = TradingMechanics(action_names, transaction_cost=transaction_cost)
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
            returns = episode["returns"].to(run_device, dtype=torch.float32, non_blocking=True)
            if returns.ndim == 1:
                returns = returns.unsqueeze(-1)
            if entities.shape[-1] != feature_size or entities.shape[1] != family_count:
                raise ValueError(f"{path} does not match the first episode shape contract")
            action_logits, target_logits = model(entities)
            # The last date has no next-period return and is not a decision.
            steps = min(int(returns.shape[0]) - 1, int(action_logits.shape[0]) - 1)
            positions = torch.zeros(family_count, dtype=torch.float32, device=run_device)
            active_targets = torch.full((family_count,), -1, dtype=torch.long, device=run_device)
            volatility = historical_volatility(returns, mechanics.volatility_floor)
            log_probs: list[torch.Tensor] = []
            rewards: list[torch.Tensor] = []
            entropies: list[torch.Tensor] = []
            for time_index in range(max(0, steps)):
                target_distribution = hierarchical_target_distribution(target_logits[time_index], active_targets)
                targets = target_distribution.sample()
                action_distribution = masked_distribution(action_logits[time_index], mechanics.legal_action_mask(positions))
                actions = action_distribution.sample()
                previous_positions = positions.clone()
                selected_returns = returns[time_index][targets]
                selected_volatility = volatility[time_index][targets]
                positions, reward = mechanics.transition(positions, actions, selected_returns, selected_volatility)
                active_targets = transition_active_targets(active_targets, previous_positions, actions, targets, mechanics)
                reward = mechanics.portfolio_return_reward(reward, family_count)
                log_probs.append(action_distribution.log_prob(actions) + target_distribution.log_prob(targets))
                rewards.append(reward)
                entropies.append(action_distribution.entropy() + target_distribution.entropy())
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
            del episode, entities, returns, action_logits, target_logits
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
        "architecture": "shared_trunk_family_heads_gru",
        "gamma": gamma,
        "transaction_cost": transaction_cost,
        "slippage": mechanics.slippage,
        "risk_penalty": mechanics.risk_penalty,
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
    mechanics = TradingMechanics(action_names, transaction_cost=transaction_cost)
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
            action_logits, target_logits = model(entities)
            steps = min(int(returns.shape[0]) - 1, int(action_logits.shape[0]) - 1)
            positions = torch.zeros(family_count, dtype=torch.float32, device=run_device)
            active_targets = torch.full((family_count,), -1, dtype=torch.long, device=run_device)
            volatility = historical_volatility(returns, mechanics.volatility_floor)
            log_probs = []; rewards = []; entropies = []
            for time_index in range(max(0, steps)):
                target_distribution = hierarchical_target_distribution(target_logits[time_index], active_targets)
                targets = target_distribution.sample()
                distribution = masked_distribution(action_logits[time_index], mechanics.legal_action_mask(positions)); actions = distribution.sample()
                previous_positions = positions.clone()
                positions, reward = mechanics.transition(positions, actions, returns[time_index], volatility[time_index])
                active_targets = transition_active_targets(active_targets, previous_positions, actions, targets, mechanics)
                reward = mechanics.portfolio_return_reward(reward, family_count)
                rewards.append(reward)
                log_probs.append(distribution.log_prob(actions) + target_distribution.log_prob(targets)); entropies.append(distribution.entropy() + target_distribution.entropy())
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
    torch.save({"model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()}, "games": sorted(game_ids), "families": list(first["families"]), "instruments": list(instruments), "actions": action_names, "one_instrument_per_manager": True, "feature_contract": f"100B_MRL_family_embedding_{feature_size}", "loss_history": history, "reward_history": total_rewards, "training_mode": "reinforce_pnl_streaming", "architecture": "shared_trunk_family_heads_gru", "gamma": gamma, "transaction_cost": transaction_cost, "slippage": mechanics.slippage, "risk_penalty": mechanics.risk_penalty, "entropy_weight": entropy_weight, "training_device": str(run_device), "gradient_accumulation": gradient_accumulation}, output)
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
    mechanics = TradingMechanics(action_names, transaction_cost=transaction_cost)
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
            logits, target_logits = model(entities.to(run_device).reshape(max_steps, batch_size * family_count, feature_size))
            positions = torch.zeros(batch_size * family_count, device=run_device)
            active_targets = torch.full((batch_size * family_count,), -1, dtype=torch.long, device=run_device)
            return_values = returns.to(run_device).unsqueeze(-1).expand(max_steps, batch_size, family_count).reshape(max_steps, batch_size * family_count)
            valid_values = valid.to(run_device).unsqueeze(-1).expand(max_steps, batch_size, family_count).reshape(max_steps, batch_size * family_count)
            volatility = historical_volatility(returns.to(run_device), mechanics.volatility_floor).unsqueeze(-1).expand(max_steps, batch_size, family_count).reshape(max_steps, batch_size * family_count)
            log_probs = []; entropies = []; rewards = []
            for time_index in range(max_steps):
                target_distribution = hierarchical_target_distribution(target_logits[time_index], active_targets)
                targets = target_distribution.sample()
                distribution = masked_distribution(logits[time_index], mechanics.legal_action_mask(positions))
                actions = distribution.sample()
                previous_positions = positions.clone()
                positions, reward = mechanics.transition(positions, actions, return_values[time_index], volatility[time_index])
                active_targets = transition_active_targets(active_targets, previous_positions, actions, targets, mechanics)
                reward = mechanics.portfolio_return_reward(reward, family_count)
                log_probs.append(distribution.log_prob(actions) + target_distribution.log_prob(targets)); entropies.append(distribution.entropy() + target_distribution.entropy())
                rewards.append(reward * valid_values[time_index])
            log_probs = torch.stack(log_probs); entropies = torch.stack(entropies)
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
            torch.save({"model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "optimizer_state": optimizer.state_dict(), "epoch": epoch, "families": list(first["families"]), "instruments": list(instruments), "feature_contract": f"100B_MRL_family_embedding_{feature_size}", "training_mode": "reinforce_pnl_streaming_batched", "architecture": "shared_trunk_family_heads_gru", "slippage": mechanics.slippage, "risk_penalty": mechanics.risk_penalty}, epoch_path)
    output.parent.mkdir(parents=True, exist_ok=True); model.eval()
    torch.save({"model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "games": sorted(ids), "families": list(first["families"]), "instruments": list(instruments), "actions": action_names, "one_instrument_per_manager": True, "feature_contract": f"100B_MRL_family_embedding_{feature_size}", "loss_history": losses_history, "reward_history": rewards_history, "training_mode": "reinforce_pnl_streaming_batched", "architecture": "shared_trunk_family_heads_gru", "gamma": gamma, "transaction_cost": transaction_cost, "slippage": mechanics.slippage, "risk_penalty": mechanics.risk_penalty, "entropy_weight": entropy_weight, "training_device": str(run_device), "gradient_accumulation": gradient_accumulation, "episode_batch_size": episode_batch_size}, output)
    return {"games": len(ids), "steps": total_steps, "families": family_count, "instruments": list(instruments), "epochs": epochs, "final_loss": losses_history[-1], "total_reward": rewards_history[-1], "artifact": str(output), "actions": list(action_names), "device": str(run_device), "training_mode": "reinforce_pnl_streaming_batched", "episode_batch_size": episode_batch_size, "compiled": compile_model, "resumed_from": str(resume_from) if resume_from else None}


def train_policy_episode_stream_rl_ppo(
    episode_factory: Callable[[], Iterable[dict]],
    epochs: int,
    learning_rate: float,
    output: Path,
    *,
    episode_batch_size: int = 16,
    action_names: tuple[str, ...] = DEFAULT_ACTION_NAMES,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    transaction_cost: float = 0.0005,
    entropy_weight: float = 0.01,
    clip_epsilon: float = 0.2,
    ppo_epochs: int = 4,
    centralized_critic: bool = False,
    device: str | torch.device | None = None,
) -> dict:
    """Train recurrent PPO, optionally with a centralized MAPPO critic.

    The actor remains one recurrent manager per family. PPO reuses each
    streamed rollout for several clipped minibatch updates. With
    ``centralized_critic=True`` the value network sees all families in an
    issuer-year game while execution remains decentralized, which is MAPPO.
    """
    if not 0 < gamma <= 1 or not 0 < gae_lambda <= 1:
        raise ValueError("gamma and gae_lambda must be in (0, 1]")
    if ppo_epochs < 1 or clip_epsilon <= 0:
        raise ValueError("ppo_epochs must be positive and clip_epsilon must be positive")
    first = next(iter(episode_factory()))
    feature_size = int(first["entities"].shape[-1])
    family_count = int(first["entities"].shape[1])
    instruments = tuple(str(x) for x in first.get("instruments", ("prices",))) or ("prices",)
    benchmark_index = instruments.index("prices") if "prices" in instruments else 0
    run_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(7)
    actor = AlphaStarPolicy(feature_size, family_count, len(instruments), action_names=action_names).to(run_device)
    critic = AlphaStarValueNetwork(feature_size, family_count, centralized=centralized_critic).to(run_device)
    optimizer = torch.optim.AdamW([*actor.parameters(), *critic.parameters()], lr=learning_rate)
    mechanics = TradingMechanics(action_names, transaction_cost=transaction_cost)
    games: list[str] = []
    total_steps = 0
    losses_history: list[float] = []
    rewards_history: list[float] = []

    for epoch in range(epochs):
        actor.train(); critic.train()
        epoch_losses: list[float] = []
        epoch_rewards: list[float] = []
        batch: list[dict] = []

        def update(items: list[dict]) -> None:
            nonlocal total_steps
            batch_size = len(items)
            max_steps = max(int(item["entities"].shape[0]) for item in items)
            entities = torch.zeros((max_steps, batch_size, family_count, feature_size), dtype=torch.float32)
            instrument_count = len(instruments)
            returns = torch.zeros((max_steps, batch_size, instrument_count), dtype=torch.float32)
            valid = torch.zeros((max_steps, batch_size), dtype=torch.bool)
            for index, item in enumerate(items):
                length = int(item["entities"].shape[0])
                entities[:length, index] = item["entities"]
                item_returns = item["returns"].float()
                if item_returns.ndim == 1:
                    item_returns = item_returns.unsqueeze(-1)
                width = min(instrument_count, int(item_returns.shape[-1]))
                returns[:length, index, :width] = item_returns[:, :width]
                valid[:length, index] = True
                if epoch == 0:
                    games.append(str(item.get("game_id", "stream-episode")))
                    total_steps += length
            flat_entities = entities.to(run_device).reshape(max_steps, batch_size * family_count, feature_size)
            portfolio_state = torch.zeros((max_steps, batch_size * family_count, 5), device=run_device)
            portfolio_state[:, :, 0] = 1.0  # available-cash fraction at the decision boundary
            value_prediction = critic(flat_entities)
            returns_device = returns.to(run_device)
            valid_values = valid.to(run_device).unsqueeze(-1).expand(max_steps, batch_size, family_count).reshape(max_steps, batch_size * family_count)
            volatility = historical_volatility(returns_device, mechanics.volatility_floor)
            # Use the change in absolute stock YTD return as the hurdle. A
            # falling stock therefore creates a positive benchmark hurdle,
            # rewarding shorts/puts only when the portfolio beats that move.
            benchmark_returns = returns_device[:, :, benchmark_index]
            benchmark_ytd = torch.cumprod(1.0 + benchmark_returns, dim=0) - 1.0
            previous_ytd = torch.cat((torch.zeros_like(benchmark_ytd[:1]), benchmark_ytd[:-1]), dim=0)
            benchmark_hurdle_delta = benchmark_ytd.abs() - previous_ytd.abs()
            ytd_state = benchmark_ytd.unsqueeze(-1).expand(max_steps, batch_size, family_count).reshape(max_steps, batch_size * family_count)
            portfolio_state[:, :, 4] = ytd_state
            action_logits, target_logits, allocation_logits = actor(flat_entities, portfolio_state=portfolio_state, return_all=True)
            positions = torch.zeros(batch_size * family_count, device=run_device)
            active_targets = torch.full((batch_size * family_count,), -1, dtype=torch.long, device=run_device)
            actions_list: list[torch.Tensor] = []
            targets_list: list[torch.Tensor] = []
            old_log_probs_list: list[torch.Tensor] = []
            masks_list: list[torch.Tensor] = []
            target_masks_list: list[torch.Tensor] = []
            rewards_list: list[torch.Tensor] = []
            entropies_list: list[torch.Tensor] = []
            target_entropies_list: list[torch.Tensor] = []
            for time_index in range(max_steps):
                action_mask = mechanics.legal_action_mask(positions)
                # Hierarchical policy: choose an instrument first. A manager
                # with an open position is locked to that instrument until it
                # sells/covers; only flat managers may choose a new target.
                target_distribution = hierarchical_target_distribution(target_logits[time_index], active_targets)
                targets = target_distribution.sample()
                action_distribution = masked_distribution(action_logits[time_index], action_mask)
                actions = action_distribution.sample()
                previous_positions = positions.clone()
                selected_returns = returns_device[time_index].unsqueeze(1).expand(batch_size, family_count, instrument_count).reshape(batch_size * family_count, instrument_count).gather(1, targets.unsqueeze(1)).squeeze(1)
                selected_volatility = volatility[time_index].unsqueeze(1).expand(batch_size, family_count, instrument_count).reshape(batch_size * family_count, instrument_count).gather(1, targets.unsqueeze(1)).squeeze(1)
                positions, reward = mechanics.transition(positions, actions, selected_returns, selected_volatility)
                active_targets = transition_active_targets(active_targets, previous_positions, actions, targets, mechanics)
                benchmark = benchmark_hurdle_delta[time_index]
                reward = mechanics.benchmark_relative_reward(reward, benchmark, family_count)
                allocation = AlphaStarPolicy.allocation_value(allocation_logits[time_index, :, 0])
                trade = (actions == mechanics._id("buy")) | (actions == mechanics._id("short"))
                reward = reward * torch.where(trade, allocation, torch.ones_like(allocation))
                actions_list.append(actions)
                targets_list.append(targets)
                old_log_probs_list.append((action_distribution.log_prob(actions) + target_distribution.log_prob(targets)).detach())
                masks_list.append(action_mask)
                target_masks_list.append((target_distribution.probs > 0).detach())
                entropies_list.append((action_distribution.entropy() + target_distribution.entropy()).detach())
                rewards_list.append(reward * valid_values[time_index])
            actions = torch.stack(actions_list)
            targets = torch.stack(targets_list)
            old_log_probs = torch.stack(old_log_probs_list)
            action_masks = torch.stack(masks_list)
            rewards = torch.stack(rewards_list)
            family_valid = valid_values
            if centralized_critic:
                group_rewards = rewards.reshape(max_steps, batch_size, family_count).mean(dim=2)
                critic_valid = valid.to(run_device)
                advantages = torch.zeros_like(value_prediction)
                running = torch.zeros(batch_size, device=run_device)
                for time_index in range(max_steps - 1, -1, -1):
                    next_value = value_prediction[time_index + 1] if time_index + 1 < max_steps else torch.zeros_like(running)
                    next_valid = critic_valid[time_index + 1] if time_index + 1 < max_steps else torch.zeros_like(running)
                    delta = group_rewards[time_index] + gamma * next_value * next_valid - value_prediction[time_index]
                    running = delta + gamma * gae_lambda * running * next_valid
                    advantages[time_index] = running * critic_valid[time_index]
                value_targets = advantages.detach() + value_prediction.detach()
                actor_advantages = advantages.unsqueeze(-1).expand(max_steps, batch_size, family_count).reshape(max_steps, batch_size * family_count)
                value_mask = critic_valid
            else:
                advantages = torch.zeros_like(value_prediction)
                running = torch.zeros(batch_size * family_count, device=run_device)
                for time_index in range(max_steps - 1, -1, -1):
                    next_value = value_prediction[time_index + 1] if time_index + 1 < max_steps else torch.zeros_like(running)
                    next_valid = family_valid[time_index + 1] if time_index + 1 < max_steps else torch.zeros_like(running)
                    delta = rewards[time_index] + gamma * next_value * next_valid - value_prediction[time_index]
                    running = delta + gamma * gae_lambda * running * next_valid
                    advantages[time_index] = running * family_valid[time_index]
                value_targets = advantages.detach() + value_prediction.detach()
                actor_advantages = advantages
                value_mask = family_valid
            valid_advantages = actor_advantages[family_valid]
            actor_advantages = (actor_advantages - valid_advantages.mean()) / valid_advantages.std().clamp_min(1e-6)
            actor_advantages = actor_advantages.detach()
            for _ in range(ppo_epochs):
                new_action_logits, new_target_logits, new_allocation_logits = actor(flat_entities, portfolio_state=portfolio_state, return_all=True)
                new_values = critic(flat_entities)
                new_log_probs = []
                entropies = []
                for time_index in range(max_steps):
                    action_distribution = masked_distribution(new_action_logits[time_index], action_masks[time_index])
                    target_distribution = masked_distribution(new_target_logits[time_index], target_masks_list[time_index])
                    new_log_probs.append(action_distribution.log_prob(actions[time_index]) + target_distribution.log_prob(targets[time_index]))
                    entropies.append(action_distribution.entropy() + target_distribution.entropy())
                new_log_probs = torch.stack(new_log_probs)
                entropies = torch.stack(entropies)
                new_allocations = AlphaStarPolicy.allocation_value(new_allocation_logits[..., 0])
                ratio = torch.exp(new_log_probs - old_log_probs)
                clipped = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
                policy_objective = torch.minimum(ratio * actor_advantages, clipped * actor_advantages)
                policy_loss = -(policy_objective * family_valid).sum() / family_valid.sum().clamp_min(1)
                value_loss = ((new_values - value_targets) ** 2 * value_mask).sum() / value_mask.sum().clamp_min(1)
                entropy = (entropies * family_valid).sum() / family_valid.sum().clamp_min(1)
                trade_mask = ((actions == mechanics._id("buy")) | (actions == mechanics._id("short"))).float() * family_valid.float()
                allocation_loss = -(new_allocations * actor_advantages.detach() * trade_mask).sum() / trade_mask.sum().clamp_min(1)
                loss = policy_loss + 0.5 * value_loss - entropy_weight * entropy + 0.05 * allocation_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_([*actor.parameters(), *critic.parameters()], 1.0)
                optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
            epoch_rewards.append(float(rewards.sum().detach().cpu()))

        for item in episode_factory():
            batch.append(item)
            if len(batch) >= max(1, episode_batch_size):
                update(batch); batch = []
        if batch:
            update(batch)
        losses_history.append(sum(epoch_losses) / max(1, len(epoch_losses)))
        rewards_history.append(sum(epoch_rewards))

    actor.eval(); critic.eval(); output.parent.mkdir(parents=True, exist_ok=True)
    algorithm = "mappo" if centralized_critic else "recurrent_ppo"
    torch.save({
        "model_state": {key: value.detach().cpu() for key, value in actor.state_dict().items()},
        "critic_state": {key: value.detach().cpu() for key, value in critic.state_dict().items()},
        "games": sorted(games), "families": list(first["families"]), "instruments": list(instruments), "actions": action_names,
        "one_instrument_per_manager": True, "feature_contract": f"100B_MRL_family_embedding_{feature_size}",
        "architecture": "shared_trunk_family_heads_gru", "algorithm": algorithm, "training_mode": algorithm,
        "centralized_critic": centralized_critic, "loss_history": losses_history, "reward_history": rewards_history,
        "gamma": gamma, "gae_lambda": gae_lambda, "transaction_cost": transaction_cost,
        "slippage": mechanics.slippage, "risk_penalty": mechanics.risk_penalty, "entropy_weight": entropy_weight,
        "clip_epsilon": clip_epsilon, "ppo_epochs": ppo_epochs, "training_device": str(run_device),
        "episode_batch_size": episode_batch_size,
        "objective": "portfolio_return_minus_absolute_stock_ytd_return",
        "benchmark_metric": "absolute_stock_ytd_return",
        "benchmark_instrument": "prices",
    }, output)
    return {"games": len(games), "steps": total_steps, "families": family_count, "instruments": list(instruments), "epochs": epochs, "final_loss": losses_history[-1], "total_reward": rewards_history[-1], "artifact": str(output), "actions": list(action_names), "device": str(run_device), "training_mode": algorithm, "centralized_critic": centralized_critic, "episode_batch_size": episode_batch_size, "ppo_epochs": ppo_epochs}
