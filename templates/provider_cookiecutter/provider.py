from __future__ import annotations

from typing import Any

from quant_orchestrator.platforms.contracts import ProviderManifest


class ExampleBacktestEngine:
    name = "example"

    def run(self, strategy: Any, data: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Implement your engine here")


# Change this class and category for ml_framework, broker, or experiment_tracker providers.
provider = ProviderManifest(
    name="example",
    category="backtest_engine",
    display_name="Example Engine",
    description="Replace this with your provider description.",
    adapters={"default": ExampleBacktestEngine},
)
