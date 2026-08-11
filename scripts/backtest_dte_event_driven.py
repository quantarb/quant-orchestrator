"""Event-driven DTE basket backtest: hub entry, authority exit, shared top-k book."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _dates(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()


def _volume_weighted_quote(frame: pd.DataFrame, column: str) -> float:
    quote = pd.to_numeric(frame[column], errors="coerce")
    volume = pd.to_numeric(frame["volume"], errors="coerce")
    valid = quote.gt(0.0) & quote.notna() & volume.gt(0.0) & volume.notna()
    if valid.any():
        return float(np.average(quote.loc[valid], weights=volume.loc[valid]))
    quote_only = quote.gt(0.0) & quote.notna()
    return float(quote.loc[quote_only].mean()) if quote_only.any() else float("nan")


def _build_daily_quotes(groups: pd.DataFrame, symbols: set[str]) -> pd.DataFrame:
    from quant_warehouse.platforms.data_providers.thetadata.options import read_thetadata_eod_option_chain

    rows: list[pd.DataFrame] = []
    for underlying, source in groups.loc[groups.underlying_symbol.isin(symbols)].groupby("underlying_symbol"):
        start = source.entry_date.min()
        end = min(source.entry_date.max() + pd.Timedelta(days=366), pd.Timestamp.today().normalize())
        chain = read_thetadata_eod_option_chain(
            underlying, start_date=start, end_date=end,
            columns=["snapshot_date", "contract_symbol", "bid", "ask", "volume"],
        )
        if chain is None or chain.empty:
            continue
        chain["snapshot_date"] = _dates(chain["snapshot_date"])
        chain["contract_symbol"] = chain["contract_symbol"].astype(str)
        chain["bid"] = pd.to_numeric(chain["bid"], errors="coerce")
        chain["ask"] = pd.to_numeric(chain["ask"], errors="coerce")
        chain["volume"] = pd.to_numeric(chain["volume"], errors="coerce")
        for row in source.itertuples(index=False):
            contracts = set(str(row.dte_contracts).split(","))
            member = chain.loc[chain.contract_symbol.isin(contracts)].copy()
            if member.empty:
                continue
            daily = member.groupby("snapshot_date").apply(
                lambda frame: pd.Series({
                    "bid": _volume_weighted_quote(frame, "bid"),
                    "ask": _volume_weighted_quote(frame, "ask"),
                    "priced_contracts": frame["contract_symbol"].nunique(),
                }), include_groups=False,
            ).reset_index()
            daily["symbol"] = row.document_symbol
            daily["option_type"] = row.option_type
            daily["entry_date"] = row.entry_date
            rows.append(daily.rename(columns={"snapshot_date": "date"}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--entry-threshold", type=float, default=0.5)
    parser.add_argument("--exit-threshold", type=float, default=0.5)
    parser.add_argument("--cost-bps-per-side", type=float, default=5.0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(args.predictions, parse_dates=["date"])
    predictions["date"] = _dates(predictions["date"])
    predictions["symbol"] = predictions["symbol"].astype(str).str.upper()
    groups = pd.read_parquet(args.groups)
    groups["entry_date"] = _dates(groups["entry_date"])
    groups["underlying_symbol"] = groups["underlying_symbol"].astype(str).str.upper()
    groups["option_type"] = groups["option_type"].astype(str).str.lower()
    group_symbols = set(predictions.symbol) & set(groups.document_symbol.astype(str).str.upper())
    groups["document_symbol"] = groups["document_symbol"].astype(str).str.upper()
    group_symbols = set(predictions.symbol) & set(groups.document_symbol)
    groups = groups.loc[groups.document_symbol.isin(group_symbols)].copy()
    quotes = _build_daily_quotes(groups, set(groups.underlying_symbol))
    if quotes.empty:
        raise RuntimeError("no daily raw bid/ask quotes found for predicted DTE groups")
    quotes["symbol"] = quotes["symbol"].astype(str).str.upper()
    quote_map = quotes.set_index(["date", "symbol"])
    score = predictions.drop_duplicates(["date", "symbol"]).set_index(["date", "symbol"])
    dates = pd.date_range(max(quotes.date.min(), predictions.date.min()), min(quotes.date.max(), predictions.date.max()), freq="B")
    positions: dict[str, dict[str, float | pd.Timestamp]] = {}
    cash = 1.0
    event_rows: list[dict[str, object]] = []
    equity_rows: list[dict[str, object]] = []
    friction = args.cost_bps_per_side / 10_000.0

    def value_at(date: pd.Timestamp) -> float:
        total = cash
        for symbol, position in positions.items():
            if (date, symbol) in quote_map.index:
                bid = float(quote_map.loc[(date, symbol), "bid"])
                if np.isfinite(bid) and bid > 0:
                    total += float(position["units"]) * bid
        return total

    for date in dates:
        day_scores = score.loc[score.index.get_level_values("date") == date] if date in score.index.get_level_values("date") else pd.DataFrame()
        # Exit first, using the side-specific authority signal.
        for symbol, position in list(positions.items()):
            if (date, symbol) not in quote_map.index or (date, symbol) not in score.index:
                continue
            row = score.loc[(date, symbol)]
            exit_col = "hits_long_return_authority" if position["option_type"] == "call" else "hits_short_return_authority"
            if float(row.get(exit_col, 0.0)) >= args.exit_threshold:
                bid = float(quote_map.loc[(date, symbol), "bid"])
                if np.isfinite(bid) and bid > 0:
                    cash += float(position["units"]) * bid * (1.0 - friction)
                    event_rows.append({"date": date, "symbol": symbol, "action": "exit", "score": float(row.get(exit_col, np.nan))})
                    del positions[symbol]
        # Enter the highest hub scores into available top-k slots.
        candidates = []
        for (candidate_date, symbol), row in score.iterrows():
            if candidate_date != date or symbol in positions or (date, symbol) not in quote_map.index:
                continue
            option_type = str(groups.loc[groups.document_symbol.eq(symbol), "option_type"].iloc[0])
            entry_col = "hits_long_return_hub" if option_type == "call" else "hits_short_return_hub"
            signal = float(row.get(entry_col, 0.0))
            ask = float(quote_map.loc[(date, symbol), "ask"])
            if signal >= args.entry_threshold and np.isfinite(ask) and ask > 0:
                candidates.append((signal, symbol, option_type, ask))
        candidates.sort(reverse=True)
        slots = max(0, int(args.top_k) - len(positions))
        for signal, symbol, option_type, ask in candidates[:slots]:
            equity = value_at(date)
            allocation = min(cash, equity / max(1, int(args.top_k)))
            if allocation <= 0:
                continue
            units = allocation * (1.0 - friction) / ask
            cash -= allocation
            positions[symbol] = {"units": units, "option_type": option_type, "entry_date": date}
            event_rows.append({"date": date, "symbol": symbol, "action": "enter", "score": signal})
        equity_rows.append({"date": date, "equity": value_at(date), "positions": len(positions)})

    equity = pd.DataFrame(equity_rows)
    equity["return"] = equity.equity.pct_change().fillna(0.0)
    equity.to_csv(args.output_dir / "daily.csv", index=False)
    pd.DataFrame(event_rows).to_csv(args.output_dir / "events.csv", index=False)
    total = float(equity.equity.iloc[-1] - 1.0)
    summary = {
        "strategy": "event_driven_dte_hub_entry_authority_exit",
        "top_k": args.top_k,
        "entry_threshold": args.entry_threshold,
        "exit_threshold": args.exit_threshold,
        "days": int(len(equity)),
        "entries": int(sum(row["action"] == "enter" for row in event_rows)),
        "exits": int(sum(row["action"] == "exit" for row in event_rows)),
        "total_return": total,
        "final_equity": float(equity.equity.iloc[-1]),
        "max_drawdown": float((equity.equity / equity.equity.cummax() - 1.0).min()),
    }
    pd.DataFrame([summary]).to_csv(args.output_dir / "summary.csv", index=False)
    print(pd.DataFrame([summary]).to_string(index=False))


if __name__ == "__main__":
    main()
