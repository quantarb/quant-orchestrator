from __future__ import annotations

import pandas as pd


def summarize_equity(equity: pd.Series) -> dict[str, float | str]:
    returns = equity.pct_change().fillna(0.0)
    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    drawdown = equity / equity.cummax() - 1
    return {
        "start": str(equity.index[0].date()),
        "end": str(equity.index[-1].date()),
        "final_equity": round(float(equity.iloc[-1]), 2),
        "total_return": round(float(total_return), 4),
        "max_drawdown": round(float(drawdown.min()), 4),
        "daily_vol": round(float(returns.std()), 4),
    }
