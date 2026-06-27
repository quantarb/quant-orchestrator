from __future__ import annotations

from typing import Any

from quant_orchestrator.platform.contracts import ProviderManifest


class NautilusBacktestEngine:
    name = "nautilus"

    def run(self, strategy: Any, data: Any, **kwargs: Any) -> Any:
        runner = kwargs.get("runner")
        if runner is None:
            raise ValueError("NautilusBacktestEngine.run requires runner=<callable>")
        return runner(strategy=strategy, data=data, **{k: v for k, v in kwargs.items() if k != "runner"})


nautilus_provider = ProviderManifest(
    name="nautilus",
    category="backtesting_framework",
    display_name="NautilusTrader",
    description="Adapter shell for NautilusTrader backtests.",
    website="https://nautilustrader.io",
    capabilities=("run",),
    adapters={"default": NautilusBacktestEngine},
)
