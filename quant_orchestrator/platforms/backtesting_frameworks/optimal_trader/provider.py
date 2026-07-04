from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from quant_orchestrator.platforms.backtesting_frameworks.optimal_trader.equity import (
    OptimalTraderBacktestConfig,
    run_optimal_trader_equity_backtest,
)
from quant_orchestrator.platforms.contracts import ProviderManifest


class OptimalTraderBacktestEngine:
    name = "optimal_trader"

    def run(self, strategy: Any, data: Any, **kwargs: Any) -> Any:
        _ = strategy
        if not isinstance(data, pd.DataFrame):
            raise TypeError("optimal_trader.run requires a pandas strategy dataset DataFrame")
        config = _resolve_config(kwargs)
        return run_optimal_trader_equity_backtest(data, config=config)


def _resolve_config(kwargs: Mapping[str, Any]) -> OptimalTraderBacktestConfig | Mapping[str, Any] | None:
    for key in ("config", "cfg", "execution", "spec"):
        value = kwargs.get(key)
        if value is not None:
            return value
    return None


optimal_trader_provider = ProviderManifest(
    name="optimal_trader",
    category="backtesting_framework",
    display_name="Optimal Trader",
    description="Optimal Trader-compatible equity backtest accounting for strategy datasets.",
    website=None,
    capabilities=("run", "equity", "strategy_dataset"),
    adapters={"default": OptimalTraderBacktestEngine},
)
