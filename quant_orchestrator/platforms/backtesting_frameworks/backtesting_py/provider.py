from __future__ import annotations

from typing import Any

import pandas as pd

from quant_orchestrator.platforms.contracts import ProviderManifest


class BacktestingPyBacktestEngine:
    name = "backtesting_py"

    def run(self, strategy: Any, data: Any, **kwargs: Any) -> Any:
        try:
            from backtesting import Backtest
        except ImportError as exc:
            raise ImportError(
                "backtesting.py is required for the backtesting_py provider. "
                "Install it with: pip install 'quant-orchestrator[backtesting]'",
            ) from exc

        if not isinstance(data, pd.DataFrame):
            raise TypeError("backtesting_py.run requires a pandas DataFrame")
        if not isinstance(strategy, type):
            raise TypeError("backtesting_py.run requires a Strategy subclass")

        bt = Backtest(data, strategy, **kwargs)
        return bt.run()


backtesting_py_provider = ProviderManifest(
    name="backtesting_py",
    category="backtesting_framework",
    display_name="backtesting.py",
    description="Vectorized or event-based backtest adapter for backtesting.py.",
    website="https://kernc.github.io/backtesting.py/",
    capabilities=("run",),
    adapters={"default": BacktestingPyBacktestEngine},
)
