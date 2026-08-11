"""Backtest MTL scores against realized option-panel returns.

This evaluator consumes only entry-time model scores and uses option_return
strictly as an out-of-sample evaluation outcome. Calls are evaluated from the
long score and puts from the short score, matching the option panel's side.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


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


def _summary(equity: pd.Series, active: pd.Series, trades: int, dates: int, threshold: float, friction_bps: float) -> dict[str, object]:
    returns = equity.pct_change().fillna(0.0)
    drawdown = equity / equity.cummax() - 1.0
    return {
        "strategy": "option_panel_long_call_short_put",
        "threshold": threshold,
        "friction_bps_round_trip": friction_bps,
        "days": dates,
        "active_days": int(active.sum()),
        "option_trades": int(trades),
        "total_return": float(equity.iloc[-1] - 1.0),
        "annualized_return": float(equity.iloc[-1] ** (252 / max(1, dates)) - 1.0),
        "annualized_sharpe": float(np.sqrt(252) * returns.mean() / returns.std()) if returns.std() > 0 else 0.0,
        "max_drawdown": float(drawdown.min()),
        "final_equity": float(equity.iloc[-1]),
    }


def _add_executable_option_return(options: pd.DataFrame) -> pd.DataFrame:
    """Convert panel mids into buy-at-ask, sell-at-bid returns."""
    out = options.copy()
    if "execution_return" in out:
        existing = pd.to_numeric(out["execution_return"], errors="coerce")
    else:
        existing = pd.Series(np.nan, index=out.index)
    def column(name: str, fallback: float = np.nan) -> pd.Series:
        value = out[name] if name in out else pd.Series(fallback, index=out.index)
        return pd.to_numeric(value, errors="coerce")
    entry_mid = column("entry_mid")
    exit_mid = column("exit_mid")
    spread = column("spread_pct", 0.0).fillna(0.0).clip(lower=0.0, upper=1.0)
    entry_ask = pd.to_numeric(out.get("entry_ask"), errors="coerce") if "entry_ask" in out else entry_mid * (1.0 + spread / 2.0)
    exit_bid = pd.to_numeric(out.get("exit_bid"), errors="coerce") if "exit_bid" in out else exit_mid * (1.0 - spread / 2.0)
    computed = exit_bid / entry_ask - 1.0
    out["execution_return"] = existing.where(existing.notna(), computed)
    out["execution_return"] = out["execution_return"].where(entry_ask.gt(0.0) & exit_bid.notna())
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--option-panel", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--fee-bps", type=float, default=0.0)
    parser.add_argument("--slippage-bps", type=float, default=0.0)
    parser.add_argument("--option-start-date", default="2025-01-01")
    parser.add_argument("--option-end-date")
    parser.add_argument("--option-dte", type=int, help="Restrict the panel to one DTE bucket.")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(args.predictions, parse_dates=["date"])
    predictions["date"] = pd.to_datetime(predictions["date"], errors="coerce").dt.normalize()
    predictions["symbol"] = predictions["symbol"].astype(str).str.upper()
    direct_option_scores = {"option_long_return", "option_short_return"}.issubset(predictions.columns)
    missing_scores = (set(LONG_SCORE_COLUMNS) | set(SHORT_SCORE_COLUMNS)) - set(predictions.columns)
    if missing_scores and not direct_option_scores:
        raise ValueError(f"prediction file missing score columns: {sorted(missing_scores)}")
    if direct_option_scores:
        predictions["long_score"] = pd.to_numeric(predictions["option_long_return"], errors="coerce")
        predictions["short_score"] = pd.to_numeric(predictions["option_short_return"], errors="coerce")
        # Direct heads predict sign(log(1 + |return|)); zero is the natural
        # decision boundary for expected positive versus negative return.
        if args.threshold == 0.5:
            args.threshold = 0.0
    else:
        predictions["long_score"] = predictions[list(LONG_SCORE_COLUMNS)].mean(axis=1)
        predictions["short_score"] = predictions[list(SHORT_SCORE_COLUMNS)].mean(axis=1)

    options = pd.read_parquet(args.option_panel)
    options = _add_executable_option_return(options)
    required = {"symbol", "entry_date", "side", "execution_return"}
    missing = required - set(options.columns)
    if missing:
        raise ValueError(f"option panel missing required columns: {sorted(missing)}")
    options["symbol"] = options["symbol"].astype(str).str.upper()
    # Document-level option panels use symbols such as
    # ``OPT_AAPL_2025_C_DTE_105``, while an equity-level model predicts the
    # underlying symbol (``AAPL``). Prefer an exact document-symbol match,
    # but fall back to the underlying symbol when the prediction file has no
    # document-level rows. This keeps both panel contracts executable.
    prediction_symbols = set(predictions["symbol"].dropna().astype(str).str.upper())
    options["prediction_symbol"] = options["symbol"]
    if "underlying_symbol" in options:
        underlying = options["underlying_symbol"].astype(str).str.upper()
        use_underlying = ~options["symbol"].isin(prediction_symbols) & underlying.isin(prediction_symbols)
        options.loc[use_underlying, "prediction_symbol"] = underlying.loc[use_underlying]
    options["entry_date"] = pd.to_datetime(options["entry_date"], errors="coerce").dt.normalize()
    if args.option_dte is not None:
        if "dte" not in options:
            raise ValueError("--option-dte requires a dte column in the option panel")
        options = options.loc[pd.to_numeric(options["dte"], errors="coerce").eq(args.option_dte)].copy()
        if options.empty:
            raise ValueError(f"option panel has no rows for DTE {args.option_dte}")
    options = options.loc[options["entry_date"].ge(pd.Timestamp(args.option_start_date))]
    if args.option_end_date:
        options = options.loc[options["entry_date"].le(pd.Timestamp(args.option_end_date))]
    options["side"] = options["side"].astype(str).str.lower()
    options["execution_return"] = pd.to_numeric(options["execution_return"], errors="coerce")
    options = options.dropna(subset=["symbol", "entry_date", "execution_return"])
    data = options.merge(
        predictions[["symbol", "date", "long_score", "short_score"]],
        left_on=["prediction_symbol", "entry_date"], right_on=["symbol", "date"], how="inner",
        suffixes=("", "_prediction"),
    )
    data["score"] = np.where(data["side"].eq("short"), data["short_score"], data["long_score"])
    data["active"] = data["score"].ge(args.threshold)
    active = data.loc[data["active"]].copy()
    if active.empty:
        raise ValueError("no active option rows at the requested threshold")
    friction = 2.0 * (args.fee_bps + args.slippage_bps) / 10_000.0
    daily = active.groupby("entry_date")["execution_return"].mean().to_frame("option_return")
    full_index = pd.date_range(data["entry_date"].min(), data["entry_date"].max(), freq="B")
    daily = daily.reindex(full_index, fill_value=0.0)
    daily.index.name = "date"
    daily["active"] = daily["option_return"].ne(0.0)
    daily["strategy_return"] = daily["option_return"] - daily["active"].astype(float) * friction
    daily["equity"] = (1.0 + daily["strategy_return"]).cumprod()
    daily.reset_index().to_csv(args.output_dir / "option_daily.csv", index=False)
    yearly_rows = []
    for year, frame in daily.groupby(daily.index.year):
        year_returns = frame["strategy_return"]
        year_equity = (1.0 + year_returns).cumprod()
        yearly_rows.append({
            "year": int(year),
            "days": int(len(frame)),
            "active_days": int(frame["active"].sum()),
            "option_trades": int(active.loc[active["entry_date"].dt.year.eq(year)].shape[0]),
            "total_return": float(year_equity.iloc[-1] - 1.0),
            "annualized_return": float(year_equity.iloc[-1] ** (252 / max(1, len(frame))) - 1.0),
            "annualized_sharpe": float(np.sqrt(252) * year_returns.mean() / year_returns.std()) if year_returns.std() > 0 else 0.0,
        })
    pd.DataFrame(yearly_rows).to_csv(args.output_dir / "yearly_summary.csv", index=False)
    summary = _summary(daily["equity"], daily["active"], len(active), len(daily), args.threshold, args.fee_bps + args.slippage_bps)
    pd.DataFrame([summary]).to_csv(args.output_dir / "summary.csv", index=False)
    (args.output_dir / "metadata.json").write_text(json.dumps({
        "predictions": str(args.predictions), "option_panel": str(args.option_panel),
        "option_dte": args.option_dte,
        "matched_rows": int(len(data)), "active_rows": int(len(active)),
        "symbols": sorted(data["prediction_symbol"].unique().tolist()),
        "option_symbols": sorted(data["symbol"].unique().tolist()),
    }, indent=2))
    print(pd.DataFrame([summary]).to_string(index=False))


if __name__ == "__main__":
    main()
