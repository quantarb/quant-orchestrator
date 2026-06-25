from __future__ import annotations

from typing import Any

from quant_orchestrator.platform.contracts import ProviderManifest


class PandasBacktestEngine:
    name = "pandas"

    def run(self, strategy: Any, data: Any, **kwargs: Any) -> Any:
        if not callable(strategy):
            raise TypeError("PandasBacktestEngine.run requires a callable strategy")
        return strategy(data, **kwargs)


pandas_provider = ProviderManifest(
    name="pandas",
    category="backtest_engine",
    display_name="Pandas",
    description="Lightweight DataFrame-native backtest adapter.",
    capabilities=("run",),
    adapters={"default": PandasBacktestEngine},
)
