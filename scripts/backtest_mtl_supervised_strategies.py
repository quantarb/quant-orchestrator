"""Backtest long-only and short-only strategies from MTL supervised scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from quant_warehouse import Warehouse


LONG_SCORE_COLUMNS = (
    "oracle_is_buy",
    "hits_long_return_hub",
    "hits_long_return_authority",
    "hits_long_speed_hub",
    "hits_long_speed_authority",
)
SHORT_SCORE_COLUMNS = (
    "oracle_is_short",
    "hits_short_return_hub",
    "hits_short_return_authority",
    "hits_short_speed_hub",
    "hits_short_speed_authority",
)


def _read_prices(warehouse: Warehouse, symbol: str, start: str, end: str, provider: str) -> pd.DataFrame:
    frame = warehouse.read_prices(symbol, provider=provider, start=start, end=end)
    if frame.empty:
        return pd.DataFrame()
    frame = frame.rename(columns=str.lower).reset_index()
    date_column = "date" if "date" in frame.columns else frame.columns[0]
    frame["date"] = pd.to_datetime(frame[date_column], errors="coerce").dt.tz_localize(None).dt.normalize()
    close_column = next((column for column in ("adj_close", "adjusted_close", "close") if column in frame), None)
    if close_column is None:
        return pd.DataFrame()
    return frame[["date", close_column]].rename(columns={close_column: "close"}).dropna().drop_duplicates("date")


def _metrics(equity: pd.Series, active: pd.Series) -> dict[str, float | int]:
    equity = equity.dropna()
    returns = equity.pct_change().fillna(0.0)
    drawdown = equity / equity.cummax() - 1.0
    annualized = (equity.iloc[-1] / equity.iloc[0]) ** (252 / max(1, len(equity))) - 1.0
    return {
        "days": int(len(equity)),
        "active_days": int(active.sum()),
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1.0),
        "annualized_return": float(annualized),
        "annualized_sharpe": float(np.sqrt(252) * returns.mean() / returns.std()) if returns.std() > 0 else 0.0,
        "max_drawdown": float(drawdown.min()),
        "final_equity": float(equity.iloc[-1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--provider", default="fmp")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--fee-bps", type=float, default=0.0)
    parser.add_argument("--slippage-bps", type=float, default=0.0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(args.predictions, parse_dates=["date"])
    predictions["date"] = predictions["date"].dt.tz_localize(None).dt.normalize()
    predictions = predictions.loc[predictions["date"].ge(pd.Timestamp("2021-01-01"))].copy()
    if predictions.empty:
        raise ValueError("no post-2021 predictions found")

    warehouse = Warehouse()
    price_parts: list[pd.DataFrame] = []
    skipped: list[str] = []
    for symbol in sorted(predictions["symbol"].astype(str).unique()):
        prices = _read_prices(warehouse, symbol, str(predictions["date"].min().date()), str((predictions["date"].max() + pd.Timedelta(days=5)).date()), args.provider)
        if prices.empty:
            skipped.append(symbol)
            continue
        prices["symbol"] = symbol
        prices["next_return"] = prices.groupby("symbol")["close"].shift(-1) / prices["close"] - 1.0
        price_parts.append(prices)
    if not price_parts:
        raise ValueError("no price history available for prediction symbols")
    prices = pd.concat(price_parts, ignore_index=True)
    data = predictions.merge(prices[["symbol", "date", "next_return"]], on=["symbol", "date"], how="inner")
    friction = 2.0 * (args.fee_bps + args.slippage_bps) / 10_000.0
    rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    yearly_summaries: list[dict[str, object]] = []
    for strategy, columns, sign in (("long_only", LONG_SCORE_COLUMNS, 1.0), ("short_only", SHORT_SCORE_COLUMNS, -1.0)):
        data["score"] = data[list(columns)].mean(axis=1)
        data["active"] = data["score"].ge(args.threshold)
        daily = data.loc[data["active"] & data["next_return"].notna()].groupby("date").agg(
            strategy_return=("next_return", lambda values: sign * values.mean() - friction),
            positions=("symbol", "count"),
        )
        full_index = pd.date_range(data["date"].min(), data["date"].max(), freq="B")
        daily = daily.reindex(full_index, fill_value=0.0)
        daily.index.name = "date"
        daily["equity"] = (1.0 + daily["strategy_return"]).cumprod()
        daily["strategy"] = strategy
        daily.reset_index().to_csv(args.output_dir / f"{strategy}_daily.csv", index=False)
        summaries.append({"strategy": strategy, "threshold": args.threshold, "friction_bps_round_trip": args.fee_bps + args.slippage_bps, **_metrics(daily["equity"], daily["positions"].gt(0))})
        for year, year_frame in daily.groupby(daily.index.year):
            year_returns = year_frame["strategy_return"]
            year_equity = (1.0 + year_returns).cumprod()
            yearly_summaries.append({
                "strategy": strategy,
                "year": int(year),
                "active_days": int(year_frame["positions"].gt(0).sum()),
                "total_return": float(year_equity.iloc[-1] - 1.0),
                "annualized_sharpe": float(np.sqrt(252) * year_returns.mean() / year_returns.std()) if year_returns.std() > 0 else 0.0,
                "max_drawdown": float((year_equity / year_equity.cummax() - 1.0).min()),
                "win_rate": float((year_returns.loc[year_frame["positions"].gt(0)] > 0).mean()) if year_frame["positions"].gt(0).any() else 0.0,
            })

    pd.DataFrame(summaries).to_csv(args.output_dir / "summary.csv", index=False)
    pd.DataFrame(yearly_summaries).to_csv(args.output_dir / "yearly_summary.csv", index=False)
    (args.output_dir / "metadata.json").write_text(json.dumps({"predictions": str(args.predictions), "provider": args.provider, "skipped_symbols": skipped, "score_columns": {"long_only": LONG_SCORE_COLUMNS, "short_only": SHORT_SCORE_COLUMNS}}, indent=2))
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
