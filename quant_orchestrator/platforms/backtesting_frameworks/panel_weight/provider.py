from __future__ import annotations

from typing import Any

import pandas as pd

from quant_orchestrator.platforms.backtesting_frameworks.panel_weight.engine import (
    ExecutionConfig,
    backtest_panel,
)
from quant_orchestrator.platforms.contracts import ProviderManifest


class PanelWeightBacktestEngine:
    name = "panel_weight"

    def run(self, strategy: Any, data: Any, **kwargs: Any) -> Any:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("panel_weight.run requires a pandas panel DataFrame")
        cfg = kwargs.get("cfg") or kwargs.get("execution") or ExecutionConfig()
        return backtest_panel(data, strategy=strategy, cfg=cfg)


panel_weight_provider = ProviderManifest(
    name="panel_weight",
    category="backtesting_framework",
    display_name="Panel Weight",
    description="Panel-based custom backtest engine for target-weight strategy matrices.",
    website=None,
    capabilities=("run", "panel", "synthetic_options"),
    adapters={"default": PanelWeightBacktestEngine},
)
