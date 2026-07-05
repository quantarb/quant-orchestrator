from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
QUANT_WAREHOUSE_ROOT = REPO_ROOT.parent / "quant-warehouse"
for path in (REPO_ROOT, QUANT_WAREHOUSE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from quant_orchestrator.platforms.backtesting_frameworks.panel_weight.thetadata_data_adapter import (  # noqa: E402
    build_panel_weight_thetadata_option_contract_features,
    read_panel_weight_thetadata_option_chain,
)
from quant_orchestrator.research_tools.options_experiment import _normalize_oracle_trades  # noqa: E402
from quant_warehouse import Warehouse  # noqa: E402
from quant_warehouse.platforms.data_providers.fmp.target_engineering import (  # noqa: E402
    LabelBuildSpec,
    build_trade_results,
)
from quant_warehouse.platforms.data_providers.thetadata.settlement import (  # noqa: E402
    iter_option_exit_lookup_dates,
    option_intrinsic_value,
)


OPTION_COLUMNS = (
    "snapshot_date",
    "contract_symbol",
    "expiration",
    "strike",
    "option_type",
    "bid",
    "ask",
    "mid",
    "delta",
    "gamma",
    "theta",
    "vega",
    "rho",
    "iv",
    "implied_volatility",
    "underlying_price",
    "volume",
    "open_interest",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", default="NVDA,GOOG")
    parser.add_argument("--output-dir", default="artifacts/fast_option_selector_nvda_goog")
    parser.add_argument("--start-date", default="2018-01-02")
    parser.add_argument("--end-date", default="2026-06-30")
    parser.add_argument("--train-end", default="2020-12-31")
    parser.add_argument("--eval-start", default="2021-01-01")
    parser.add_argument("--k-min", type=int, default=1)
    parser.add_argument("--k-max", type=int, default=12)
    parser.add_argument("--target-dte", type=int, default=90)
    parser.add_argument("--max-candidates-per-trade", type=int, default=750)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--skip-panel", action="store_true")
    args = parser.parse_args()

    started = perf_counter()
    symbols = _parse_symbols(args.symbols)
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    panel_path = out_dir / "option_candidate_panel.parquet"
    trades_path = out_dir / "oracle_trades.parquet"

    if args.skip_panel and panel_path.exists():
        option_panel = pd.read_parquet(panel_path)
        oracle_trades = pd.read_parquet(trades_path) if trades_path.exists() else pd.DataFrame()
    else:
        warehouse = Warehouse()
        price_frames = _load_price_frames(
            warehouse,
            symbols,
            start=str(args.start_date),
            end=str(args.end_date),
        )
        spec = LabelBuildSpec(
            k_params={"YE": list(range(int(args.k_min), int(args.k_max) + 1))},
            min_profit_pct=0.01,
            buy_execution="high",
            sell_execution="low",
            short_execution="low",
            cover_execution="high",
            start_date=str(args.start_date),
            end_date=str(args.end_date),
        )
        trade_result = build_trade_results(symbols, spec=spec, price_frames=price_frames)
        oracle_trades = _normalize_oracle_trades(pd.DataFrame(trade_result.completed_trades))
        oracle_trades = _with_unique_oracle_trade_ids(oracle_trades)
        option_panel = build_fast_option_candidate_panel(
            oracle_trades,
            target_dte=int(args.target_dte),
            max_candidates_per_trade=int(args.max_candidates_per_trade),
            exit_lookback_days=7,
        )
        oracle_trades.to_parquet(trades_path, index=False)
        option_panel.to_parquet(panel_path, index=False)

    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_option_family_ranker_experiment.py"),
        "--option-panel",
        str(panel_path),
        "--symbols",
        ",".join(symbols),
        "--all-feature-families",
        "--train-end",
        str(args.train_end),
        "--eval-start",
        str(args.eval_start),
        "--n-estimators",
        str(args.n_estimators),
        "--output-dir",
        str(out_dir / "family_rankers"),
    ]
    ranker = subprocess.run(command, cwd=REPO_ROOT, check=True, capture_output=True, text=True)
    (out_dir / "ranker_stdout.json").write_text(ranker.stdout, encoding="utf-8")
    summary = {
        "symbols": symbols,
        "oracle_trades": int(len(oracle_trades)),
        "option_rows": int(len(option_panel)),
        "option_panel": str(panel_path),
        "train_end": str(args.train_end),
        "eval_start": str(args.eval_start),
        "elapsed_seconds": float(perf_counter() - started),
    }
    if not option_panel.empty:
        dates = pd.to_datetime(option_panel["entry_date"], errors="coerce")
        summary.update(
            {
                "trade_windows": int(option_panel["trade_id"].nunique()),
                "min_entry_date": None if dates.dropna().empty else dates.min().date().isoformat(),
                "max_entry_date": None if dates.dropna().empty else dates.max().date().isoformat(),
                "train_rows": int(dates.le(pd.Timestamp(args.train_end)).sum()),
                "eval_rows": int(dates.ge(pd.Timestamp(args.eval_start)).sum()),
                "train_trades": int(option_panel.loc[dates.le(pd.Timestamp(args.train_end)), "trade_id"].nunique()),
                "eval_trades": int(option_panel.loc[dates.ge(pd.Timestamp(args.eval_start)), "trade_id"].nunique()),
            }
        )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))
    print(ranker.stdout)


def build_fast_option_candidate_panel(
    trades: pd.DataFrame,
    *,
    target_dte: int,
    max_candidates_per_trade: int,
    exit_lookback_days: int,
) -> pd.DataFrame:
    rows = []
    quote_cache: dict[tuple[str, pd.Timestamp], pd.DataFrame] = {}
    chain_cache: dict[tuple[str, pd.Timestamp], pd.DataFrame] = {}
    underlying_cache: dict[tuple[str, pd.Timestamp], float | None] = {}
    for trade in trades.sort_values(["entry_date", "symbol", "side"]).to_dict("records"):
        frame = _candidate_rows_for_trade(
            trade,
            target_dte=target_dte,
            max_candidates_per_trade=max_candidates_per_trade,
            exit_lookback_days=exit_lookback_days,
            quote_cache=quote_cache,
            chain_cache=chain_cache,
            underlying_cache=underlying_cache,
        )
        if not frame.empty:
            rows.append(frame)
    if not rows:
        return pd.DataFrame()
    panel = pd.concat(rows, ignore_index=True)
    panel["option_return"] = pd.to_numeric(panel["option_return"], errors="coerce")
    panel = panel.dropna(subset=["option_return"])
    panel["rank_y"] = panel.groupby("trade_id")["option_return"].rank(method="average", pct=True, ascending=True)
    return panel.reset_index(drop=True)


def _candidate_rows_for_trade(
    trade: dict[str, Any],
    *,
    target_dte: int,
    max_candidates_per_trade: int,
    exit_lookback_days: int,
    quote_cache: dict[tuple[str, pd.Timestamp], pd.DataFrame],
    chain_cache: dict[tuple[str, pd.Timestamp], pd.DataFrame],
    underlying_cache: dict[tuple[str, pd.Timestamp], float | None],
) -> pd.DataFrame:
    symbol = str(trade["symbol"]).upper()
    side = str(trade["side"]).lower()
    entry_date = pd.Timestamp(trade["entry_date"]).normalize()
    equity_exit_date = pd.Timestamp(trade["exit_date"]).normalize()
    entry_chain = _load_day_chain(symbol, entry_date, chain_cache=chain_cache)
    if entry_chain.empty:
        return pd.DataFrame()
    spot = _chain_underlying_price(entry_chain)
    if spot is None:
        return pd.DataFrame()
    candidates = entry_chain.copy()
    candidates["option_type"] = candidates["option_type"].astype(str).str.lower().str.strip()
    candidates = candidates.loc[candidates["option_type"].str.startswith(("c", "p"))].copy()
    if candidates.empty:
        return candidates
    candidates["option_type"] = np.where(candidates["option_type"].str.startswith("c"), "call", "put")
    candidates = build_panel_weight_thetadata_option_contract_features(
        candidates,
        underlying_price=float(spot),
        target_dte=int(target_dte),
        compute_model_greeks=False,
    ).df
    candidates = _filter_and_rank_entry_candidates(candidates, max_candidates_per_trade=max_candidates_per_trade)
    if candidates.empty:
        return candidates
    candidates["option_action"] = np.select(
        [
            (side == "long") & candidates["option_type"].eq("call"),
            (side == "long") & candidates["option_type"].eq("put"),
            (side == "short") & candidates["option_type"].eq("put"),
            (side == "short") & candidates["option_type"].eq("call"),
        ],
        ["buy_call", "sell_put", "buy_put", "sell_call"],
        default="",
    )
    candidates = candidates.loc[candidates["option_action"].ne("")].copy()
    if candidates.empty:
        return candidates
    priced = _price_candidate_exits(
        candidates,
        symbol=symbol,
        equity_exit_date=equity_exit_date,
        entry_date=entry_date,
        quote_cache=quote_cache,
        underlying_cache=underlying_cache,
        exit_lookback_days=exit_lookback_days,
    )
    if priced.empty:
        return priced
    priced.insert(0, "trade_id", trade.get("trade_id", f"{symbol}|{entry_date.date()}|{side}"))
    priced.insert(1, "symbol", symbol)
    priced.insert(2, "side", side)
    priced["equity_signal_side"] = side
    priced["entry_date"] = entry_date
    priced["equity_exit_date"] = equity_exit_date
    priced["realized_holding_days"] = max(0, int((equity_exit_date - entry_date).days))
    priced["realized_underlying_trade_return"] = pd.to_numeric(trade.get("ret_dec"), errors="coerce")
    for col in ("freq", "k", "entry_px", "exit_px", "ret_pct", "hold_days"):
        if col in trade:
            priced[col] = trade[col]
    return priced.reset_index(drop=True)


def _filter_and_rank_entry_candidates(chain: pd.DataFrame, *, max_candidates_per_trade: int) -> pd.DataFrame:
    work = chain.copy()
    for col in ("contract_symbol", "expiration", "strike", "bid", "ask", "mid", "dte"):
        if col not in work.columns:
            return pd.DataFrame()
    for col in ("strike", "bid", "ask", "mid", "dte", "spread_pct", "abs_moneyness", "liquidity_score"):
        if col not in work.columns:
            work[col] = np.nan
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.loc[
        work["contract_symbol"].astype(str).ne("")
        & work["expiration"].notna()
        & work["strike"].gt(0)
        & work["bid"].ge(0)
        & work["ask"].gt(0)
        & work["ask"].ge(work["bid"])
        & work["mid"].gt(0)
        & work["dte"].gt(0)
    ].copy()
    if work.empty:
        return work
    sort_cols = ["dte_gap", "abs_moneyness", "spread_pct", "liquidity_score"]
    ascending = [True, True, True, False]
    available = [col for col in sort_cols if col in work.columns]
    if available:
        ascending = [ascending[sort_cols.index(col)] for col in available]
        work = work.sort_values(available, ascending=ascending, kind="stable")
    if max_candidates_per_trade > 0:
        work = work.head(int(max_candidates_per_trade)).copy()
    return work.reset_index(drop=True)


def _price_candidate_exits(
    candidates: pd.DataFrame,
    *,
    symbol: str,
    equity_exit_date: pd.Timestamp,
    entry_date: pd.Timestamp,
    quote_cache: dict[tuple[str, pd.Timestamp], pd.DataFrame],
    underlying_cache: dict[tuple[str, pd.Timestamp], float | None],
    exit_lookback_days: int,
) -> pd.DataFrame:
    parts = []
    work = candidates.copy()
    work["expiration"] = pd.to_datetime(work["expiration"], errors="coerce").dt.normalize()
    work["target_exit_date"] = work["expiration"].where(work["expiration"].lt(equity_exit_date), equity_exit_date)
    for target_exit, group in work.groupby("target_exit_date", dropna=True):
        exit_quote = _lookup_exit_quotes(
            symbol,
            pd.Timestamp(target_exit).normalize(),
            contracts=group["contract_symbol"].astype(str).tolist(),
            entry_date=entry_date,
            quote_cache=quote_cache,
            exit_lookback_days=exit_lookback_days,
        )
        merged = group.merge(exit_quote, on="contract_symbol", how="left")
        expired = merged["expiration"].eq(pd.Timestamp(target_exit).normalize())
        missing_quote = merged["exit_mid"].isna()
        if bool((expired & missing_quote).any()):
            spot = _underlying_from_chain_cache(symbol, pd.Timestamp(target_exit), quote_cache, underlying_cache)
            intrinsic = merged.loc[expired & missing_quote].apply(
                lambda row: option_intrinsic_value(
                    option_type=row["option_type"],
                    strike=row["strike"],
                    underlying_price=spot,
                ),
                axis=1,
            )
            idx = merged.index[expired & missing_quote]
            merged.loc[idx, "exit_bid"] = intrinsic.to_numpy(dtype=float)
            merged.loc[idx, "exit_ask"] = intrinsic.to_numpy(dtype=float)
            merged.loc[idx, "exit_mid"] = intrinsic.to_numpy(dtype=float)
            merged.loc[idx, "option_exit_date"] = pd.Timestamp(target_exit).normalize()
            merged.loc[idx, "exit_price_source"] = "expiration_intrinsic"
        parts.append(merged)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    out = out.loc[out["exit_mid"].notna()].copy()
    if out.empty:
        return out
    out["entry_bid"] = pd.to_numeric(out["bid"], errors="coerce")
    out["entry_ask"] = pd.to_numeric(out["ask"], errors="coerce")
    out["entry_mid"] = pd.to_numeric(out["mid"], errors="coerce")
    out["entry_price"] = np.where(out["option_action"].str.startswith("buy_"), out["entry_ask"], out["entry_bid"])
    out["return_denominator"] = np.select(
        [out["option_action"].eq("sell_put"), out["option_action"].eq("sell_call")],
        [out["strike"], out["underlying_spot_entry"]],
        default=out["entry_price"],
    )
    out["exit_price"] = np.where(out["option_action"].str.startswith("buy_"), out["exit_bid"], out["exit_ask"])
    out["option_pnl"] = np.where(
        out["option_action"].str.startswith("buy_"),
        out["exit_price"] - out["entry_price"],
        out["entry_price"] - out["exit_price"],
    )
    out["option_return"] = out["option_pnl"] / out["return_denominator"].replace(0, np.nan)
    out["fixed_near_atm_score"] = -(
        _scaled(out.get("dte_gap"))
        + _scaled(out.get("abs_moneyness"))
        + _scaled(out.get("spread_pct"))
    )
    return out.replace([np.inf, -np.inf], np.nan).dropna(subset=["entry_price", "exit_price", "option_return"])


def _lookup_exit_quotes(
    symbol: str,
    target_exit: pd.Timestamp,
    *,
    contracts: list[str],
    entry_date: pd.Timestamp,
    quote_cache: dict[tuple[str, pd.Timestamp], pd.DataFrame],
    exit_lookback_days: int,
) -> pd.DataFrame:
    wanted = set(str(contract) for contract in contracts)
    for quote_date in iter_option_exit_lookup_dates(target_exit, exit_lookback_days):
        quote_date = pd.Timestamp(quote_date).normalize()
        if target_exit > entry_date and quote_date <= entry_date:
            continue
        chain = _load_day_quotes(symbol, quote_date, quote_cache=quote_cache)
        if chain.empty:
            continue
        found = chain.loc[chain["contract_symbol"].astype(str).isin(wanted)].copy()
        if found.empty:
            continue
        found = found.rename(columns={"bid": "exit_bid", "ask": "exit_ask", "mid": "exit_mid"})
        found["option_exit_date"] = quote_date
        found["exit_price_source"] = "contract_quote" if quote_date == target_exit else "last_contract_quote"
        return found[["contract_symbol", "option_exit_date", "exit_bid", "exit_ask", "exit_mid", "exit_price_source"]]
    return pd.DataFrame(columns=["contract_symbol", "option_exit_date", "exit_bid", "exit_ask", "exit_mid", "exit_price_source"])


def _load_day_chain(
    symbol: str,
    date: pd.Timestamp,
    *,
    chain_cache: dict[tuple[str, pd.Timestamp], pd.DataFrame],
) -> pd.DataFrame:
    key = (str(symbol).upper(), pd.Timestamp(date).normalize())
    cached = chain_cache.get(key)
    if cached is not None:
        return cached.copy()
    chain = read_panel_weight_thetadata_option_chain(
        key[0],
        start_date=key[1],
        end_date=key[1],
        columns=OPTION_COLUMNS,
        require_rich_columns=True,
    )
    if not chain.empty:
        chain = _dedupe_contracts(chain)
    chain_cache[key] = chain.copy()
    return chain


def _load_day_quotes(
    symbol: str,
    date: pd.Timestamp,
    *,
    quote_cache: dict[tuple[str, pd.Timestamp], pd.DataFrame],
) -> pd.DataFrame:
    key = (str(symbol).upper(), pd.Timestamp(date).normalize())
    cached = quote_cache.get(key)
    if cached is not None:
        return cached.copy()
    chain = read_panel_weight_thetadata_option_chain(
        key[0],
        start_date=key[1],
        end_date=key[1],
        columns=("snapshot_date", "contract_symbol", "bid", "ask", "mid", "underlying_price"),
        require_rich_columns=False,
    )
    if chain.empty:
        quote_cache[key] = chain.copy()
        return chain
    chain = _dedupe_contracts(chain)
    for col in ("bid", "ask", "mid", "underlying_price"):
        if col not in chain.columns:
            chain[col] = np.nan
        chain[col] = pd.to_numeric(chain[col], errors="coerce")
    missing_mid = chain["mid"].isna() & chain["bid"].notna() & chain["ask"].notna()
    chain.loc[missing_mid, "mid"] = (chain.loc[missing_mid, "bid"] + chain.loc[missing_mid, "ask"]) / 2.0
    chain["bid"] = chain["bid"].where(chain["bid"].notna(), chain["mid"])
    chain["ask"] = chain["ask"].where(chain["ask"].notna(), chain["mid"])
    chain = chain.loc[chain["contract_symbol"].astype(str).ne("") & chain["mid"].notna()].copy()
    quote_cache[key] = chain.copy()
    return chain


def _underlying_from_chain_cache(
    symbol: str,
    date: pd.Timestamp,
    quote_cache: dict[tuple[str, pd.Timestamp], pd.DataFrame],
    underlying_cache: dict[tuple[str, pd.Timestamp], float | None],
) -> float | None:
    key = (str(symbol).upper(), pd.Timestamp(date).normalize())
    if key in underlying_cache:
        return underlying_cache[key]
    chain = _load_day_quotes(key[0], key[1], quote_cache=quote_cache)
    spot = _chain_underlying_price(chain)
    underlying_cache[key] = spot
    return spot


def _chain_underlying_price(chain: pd.DataFrame) -> float | None:
    if chain is None or chain.empty or "underlying_price" not in chain.columns:
        return None
    values = pd.to_numeric(chain["underlying_price"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    values = values.loc[values > 0]
    return None if values.empty else float(values.median())


def _dedupe_contracts(chain: pd.DataFrame) -> pd.DataFrame:
    work = chain.copy()
    if "snapshot_date" in work.columns:
        work["_snapshot_sort"] = pd.to_datetime(work["snapshot_date"], errors="coerce")
        work = work.sort_values(["contract_symbol", "_snapshot_sort"])
    else:
        work = work.sort_values(["contract_symbol"])
    work = work.drop_duplicates("contract_symbol", keep="last")
    return work.drop(columns=["_snapshot_sort"], errors="ignore").reset_index(drop=True)


def _load_price_frames(warehouse: Warehouse, symbols: tuple[str, ...], *, start: str, end: str | None) -> dict[str, pd.DataFrame]:
    frames = {}
    for symbol in symbols:
        frame = warehouse.read_prices(symbol, provider="fmp", start=start, end=end).copy()
        if frame.empty:
            continue
        frame.index = pd.to_datetime(frame.index, errors="coerce").normalize()
        frames[symbol] = frame.sort_index()
    return frames


def _parse_symbols(value: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(symbol.strip().upper() for symbol in str(value).split(",") if symbol.strip()))


def _with_unique_oracle_trade_ids(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return trades.copy()
    out = trades.copy()
    for col in ("symbol", "side", "entry_date", "exit_date", "freq", "k"):
        if col not in out.columns:
            out[col] = ""
    entry = pd.to_datetime(out["entry_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    exit_ = pd.to_datetime(out["exit_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["trade_id"] = (
        out["symbol"].astype(str).str.upper()
        + "|"
        + out["side"].astype(str).str.lower()
        + "|"
        + entry.astype(str)
        + "|"
        + exit_.astype(str)
        + "|"
        + out["freq"].astype(str)
        + "|k"
        + out["k"].astype(str)
    )
    return out


def _scaled(values: pd.Series | None) -> pd.Series:
    if values is None:
        return pd.Series(dtype=float)
    out = pd.to_numeric(values, errors="coerce")
    scale = out.abs().median()
    if pd.notna(scale) and float(scale) > 0:
        out = out / float(scale)
    return out.fillna(out.max())


if __name__ == "__main__":
    main()
