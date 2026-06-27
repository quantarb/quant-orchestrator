from __future__ import annotations

from typing import Any

from quant_orchestrator.platform.contracts import ProviderManifest


class ZiplineBacktestEngine:
    name = "zipline"

    def run(self, strategy: Any, data: Any, **kwargs: Any) -> Any:
        runner = kwargs.get("runner")
        if runner is None:
            raise ValueError("ZiplineBacktestEngine.run requires runner=<callable>")
        return runner(strategy=strategy, data=data, **{k: v for k, v in kwargs.items() if k != "runner"})


zipline_provider = ProviderManifest(
    name="zipline",
    category="backtesting_framework",
    display_name="Zipline Reloaded",
    description="Adapter shell for Zipline Reloaded backtests.",
    website="https://github.com/stefan-jansen/zipline-reloaded",
    capabilities=("run",),
    adapters={"default": ZiplineBacktestEngine},
)
