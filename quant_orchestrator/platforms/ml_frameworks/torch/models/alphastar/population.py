"""Variable-population portfolio mechanics for Fleetcraft AlphaStar.

The original AlphaStar treats the observable units as a variable-length set
and chooses structured actions against that set.  This adapter gives
Fleetcraft the equivalent environment primitive: an ape is an entity, its
feature family is its unit type, and spawning is a funded state transition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class ApeState:
    ape_id: int
    family: str
    cash: float = 1000.0
    position: float = 0.0
    quantity: float = 0.0
    instrument: str | None = None
    entry_mark: float = 0.0
    current_mark: float = 0.0
    reserved_cash: float = 0.0
    contract_multiplier: float = 1.0
    alive: bool = True
    realized_pnl: float = 0.0

    @property
    def equity(self) -> float:
        mark = self.current_mark or self.entry_mark
        multiplier = max(1.0, self.contract_multiplier)
        if self.position > 0:
            return self.cash + self.quantity * mark * multiplier
        if self.position < 0:
            # Short proceeds are held as collateral.  Only mark-to-market P&L
            # changes equity while the position is open.
            pnl = (self.entry_mark - mark) * self.quantity * multiplier
            return self.cash + self.reserved_cash + pnl
        return self.cash


@dataclass(frozen=True)
class ApeAction:
    """Structured action emitted by the population policy."""

    action: str
    ape_id: int
    instrument: str | None = None
    family: str | None = None
    allocation: float = 0.75
    source_ape_id: int | None = None


@dataclass
class PopulationTransition:
    spawned: list[ApeState] = field(default_factory=list)
    despawned: list[int] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)


class FleetcraftPopulation:
    """Cash-funded variable-length ape population.

    A spawn transfers ``spawn_cost`` from an existing source ape to a new ape;
    no global treasury is created.  Trade allocation is a fraction of the
    ape's available cash, and bankrupt apes are removed from the population.
    """

    def __init__(self, families: Iterable[str], *, portfolio_cash: float = 100_000.0, starting_cash: float | None = None, spawn_cost: float | None = None):
        family_list = [str(family) for family in families]
        if not family_list:
            raise ValueError("At least one feature-family ape is required")
        # The initial population owns the complete portfolio.  Each initial
        # ape receives an equal slice; spawning transfers one such slice from
        # its source ape and never increases total capital.
        self.portfolio_cash = float(portfolio_cash)
        self.starting_cash = float(starting_cash) if starting_cash is not None else self.portfolio_cash / len(family_list)
        self.spawn_cost = float(spawn_cost) if spawn_cost is not None else self.starting_cash
        self._next_id = 0
        self.apes: dict[int, ApeState] = {}
        for family in family_list:
            self._add(family, self.starting_cash)

    def _add(self, family: str, cash: float) -> ApeState:
        ape = ApeState(self._next_id, family, cash=float(cash))
        self._next_id += 1
        self.apes[ape.ape_id] = ape
        return ape

    def observation_entities(self) -> list[ApeState]:
        return [ape for ape in self.apes.values() if ape.alive]

    def legal_spawn_sources(self) -> list[int]:
        return [ape.ape_id for ape in self.observation_entities() if ape.cash >= self.spawn_cost]

    def net_worth(self) -> float:
        """Current total equity, including every live ape."""
        return sum(max(0.0, ape.equity) for ape in self.observation_entities())

    def spawn(self, source_ape_id: int, family: str | None = None) -> ApeState:
        source = self.apes.get(int(source_ape_id))
        if source is None or not source.alive:
            raise ValueError("source ape is not alive")
        if source.cash < self.spawn_cost:
            raise ValueError("source ape cannot fund spawn")
        source.cash -= self.spawn_cost
        return self._add(str(family or source.family), self.spawn_cost)

    def trade(self, ape_id: int, action: str, mark: float, *, multiplier: float = 1.0, allocation: float = 0.75, instrument: str | None = None) -> None:
        ape = self.apes.get(int(ape_id))
        if ape is None or not ape.alive:
            return
        mark = max(0.0, float(mark)); multiplier = max(1.0, float(multiplier)); allocation = min(1.0, max(0.0, float(allocation)))
        if action in {"buy", "short"} and ape.position == 0 and mark > 0:
            budget = ape.cash * allocation
            unit_cost = mark * multiplier
            # Fleetcraft trades whole shares and whole option contracts. Do
            # not create a fractional position or a zero-sized trade.
            quantity = float(int(budget / unit_cost))
            if quantity >= 1:
                ape.cash -= quantity * unit_cost
                ape.position = 1.0 if action == "buy" else -1.0
                ape.quantity = quantity
                ape.instrument = instrument
                ape.entry_mark = mark
                ape.current_mark = mark
                ape.contract_multiplier = multiplier
                ape.reserved_cash = quantity * unit_cost if action == "short" else 0.0
        elif action in {"sell", "cover"} and ape.position != 0:
            value = ape.quantity * mark * multiplier
            cost = ape.quantity * ape.entry_mark * multiplier
            if ape.position > 0:
                ape.realized_pnl += value - cost
                ape.cash += value
            else:
                ape.realized_pnl += cost - value
                ape.cash += ape.reserved_cash + (cost - value)
            ape.position = ape.quantity = ape.entry_mark = 0.0
            ape.instrument = None
            ape.current_mark = ape.reserved_cash = 0.0
            ape.contract_multiplier = 1.0

    def mark_to_market(self, marks: dict[int, float], *, multipliers: dict[int, float] | None = None) -> PopulationTransition:
        transition = PopulationTransition()
        multipliers = multipliers or {}
        for ape in list(self.observation_entities()):
            mark = max(0.0, float(marks.get(ape.ape_id, ape.entry_mark)))
            multiplier = max(1.0, float(multipliers.get(ape.ape_id, 1.0)))
            ape.current_mark = mark
            ape.contract_multiplier = multiplier
            equity = ape.equity
            if equity <= 0:
                ape.alive = False
                transition.despawned.append(ape.ape_id)
        for ape_id in transition.despawned:
            self.apes.pop(ape_id, None)
        return transition

    def apply(self, actions: Iterable[ApeAction], marks: dict[int, float], *, multipliers: dict[int, float] | None = None) -> PopulationTransition:
        transition = PopulationTransition()
        for action in actions:
            if action.action == "spawn":
                try:
                    transition.spawned.append(self.spawn(action.source_ape_id if action.source_ape_id is not None else action.ape_id, action.family))
                except ValueError as error:
                    transition.rejected.append(str(error))
                continue
            mark = marks.get(action.ape_id)
            if mark is None:
                transition.rejected.append(f"missing mark for ape {action.ape_id}")
                continue
            self.trade(action.ape_id, action.action, mark, multiplier=(multipliers or {}).get(action.ape_id, 1.0), allocation=action.allocation, instrument=action.instrument)
        settled = self.mark_to_market(marks, multipliers=multipliers)
        transition.despawned.extend(settled.despawned)
        return transition
