from __future__ import annotations

import pandas as pd


def build_backtesting_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
            "signal": "Signal",
        },
    ).copy()
