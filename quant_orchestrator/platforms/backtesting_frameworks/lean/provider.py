from __future__ import annotations

from typing import Any

from quant_orchestrator.platforms.contracts import ProviderManifest


class LeanBacktestEngine:
    name = "lean"

    def run(self, strategy: Any, data: Any, **kwargs: Any) -> Any:
        runner = kwargs.get("runner")
        if runner is None:
            raise ValueError(
                "LeanBacktestEngine.run requires runner=<callable>. "
                "Install the QuantConnect LEAN CLI and supply a runner that invokes "
                "the local LEAN backtest entry point.",
            )
        return runner(strategy=strategy, data=data, **{k: v for k, v in kwargs.items() if k != "runner"})


lean_provider = ProviderManifest(
    name="lean",
    category="backtesting_framework",
    display_name="QuantConnect LEAN",
    description="Adapter shell for LEAN backtests.",
    website="https://www.quantconnect.com/lean/",
    capabilities=("run",),
    adapters={"default": LeanBacktestEngine},
)
