from __future__ import annotations

from typing import Any

from quant_orchestrator.options_backtesting import OptopsyBacktestSpec, run_optopsy_backtest
from quant_orchestrator.platforms.contracts import ProviderManifest


class OptopsyBacktestEngine:
    name = "optopsy"

    def run(self, strategy: Any, data: Any, **kwargs: Any) -> Any:
        if isinstance(data, OptopsyBacktestSpec):
            return run_optopsy_backtest(data)
        if "spec" in kwargs:
            return run_optopsy_backtest(kwargs["spec"])
        raise ValueError("OptopsyBacktestEngine.run requires data=<OptopsyBacktestSpec> or spec=...")


optopsy_provider = ProviderManifest(
    name="optopsy",
    category="backtesting_framework",
    display_name="Optopsy",
    description="ThetaData options backtests through Optopsy and quant-warehouse.",
    website="https://github.com/goldspanlabs/optopsy",
    capabilities=("run", "options"),
    adapters={"default": OptopsyBacktestEngine},
)
