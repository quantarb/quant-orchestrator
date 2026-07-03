from __future__ import annotations

from typing import Any

from quant_orchestrator.platforms.contracts import ProviderManifest
from quant_orchestrator.platforms.backtesting_frameworks.lean.runner import (
    LeanRunConfig,
    run_lean_backtest,
)


class LeanBacktestEngine:
    name = "lean"

    def run(self, strategy: Any, data: Any, **kwargs: Any) -> Any:
        runner = kwargs.get("runner")
        if runner is not None:
            return runner(strategy=strategy, data=data, **{k: v for k, v in kwargs.items() if k != "runner"})
        if not isinstance(data, LeanRunConfig):
            raise ValueError(
                "LeanBacktestEngine.run requires runner=<callable> or data=<LeanRunConfig>. "
                "Use the notebook helper to build a config for a local LEAN workspace.",
            )
        return run_lean_backtest(data)


lean_provider = ProviderManifest(
    name="lean",
    category="backtesting_framework",
    display_name="QuantConnect LEAN",
    description="Adapter shell for LEAN backtests.",
    website="https://www.quantconnect.com/lean/",
    capabilities=("run",),
    adapters={"default": LeanBacktestEngine},
)
