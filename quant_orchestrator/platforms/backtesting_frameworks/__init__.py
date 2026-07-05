"""Built-in backtest engine providers."""

from quant_orchestrator.platforms.backtesting_frameworks.nautilus.provider import nautilus_provider
from quant_orchestrator.platforms.backtesting_frameworks.backtesting_py.provider import (
    backtesting_py_provider,
)
from quant_orchestrator.platforms.backtesting_frameworks.lean.provider import lean_provider
from quant_orchestrator.platforms.backtesting_frameworks.optimal_trader.provider import (
    optimal_trader_provider,
)
from quant_orchestrator.platforms.backtesting_frameworks.scored_panel_replay import (
    ScoredPanelTopKReplayConfig,
    ScoredPanelTopKReplayResult,
    replay_scored_panel_top_k,
    write_scored_panel_top_k_outputs,
)
from quant_orchestrator.platforms.backtesting_frameworks.zipline.provider import zipline_provider

__all__ = [
    "backtesting_py_provider",
    "lean_provider",
    "nautilus_provider",
    "optimal_trader_provider",
    "replay_scored_panel_top_k",
    "ScoredPanelTopKReplayConfig",
    "ScoredPanelTopKReplayResult",
    "write_scored_panel_top_k_outputs",
    "zipline_provider",
]
