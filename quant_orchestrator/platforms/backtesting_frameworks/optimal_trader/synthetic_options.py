from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from quant_orchestrator.platforms.backtesting_frameworks.optimal_trader.artifact_replay import clean_price_frame
from quant_orchestrator.research_tools.option_trade_execution import OptionTradeExecutionBatch, OptionTradeExecutor


@dataclass(frozen=True)
class FmpSyntheticOptionReplayConfig:
    target_dte: int = 90
    option_type_for_long: str = "call"
    option_type_for_short: str = "put"
    selector_name: str = "rule_atm_90d"
    workers: int = 1


def run_fmp_synthetic_option_trade_replay(
    trade_windows: pd.DataFrame,
    *,
    config: FmpSyntheticOptionReplayConfig | None = None,
) -> OptionTradeExecutionBatch:
    cfg = config or FmpSyntheticOptionReplayConfig()
    normalized = _normalize_trade_windows(trade_windows)
    if normalized.empty:
        return OptionTradeExecutionBatch(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {})
    started = perf_counter()
    batch = OptionTradeExecutor(
        lambda: FmpSyntheticOptionRetriever(config=cfg),
        selector_name=cfg.selector_name,
        workers=max(1, int(cfg.workers)),
    ).execute(normalized)
    metrics = dict(batch.metrics)
    metrics["elapsed_seconds"] = float(perf_counter() - started)
    metrics["trade_windows"] = float(len(normalized))
    metrics["selected_option_trades"] = float(len(batch.selected_option_trades))
    metrics["selected_option_paths"] = float(len(batch.selected_option_paths))
    metrics["workers"] = float(max(1, int(cfg.workers)))
    return OptionTradeExecutionBatch(
        selected_option_trades=batch.selected_option_trades,
        selected_option_paths=batch.selected_option_paths,
        trade_status=batch.trade_status,
        metrics=metrics,
    )


class FmpSyntheticOptionRetriever:
    def __init__(self, *, config: FmpSyntheticOptionReplayConfig | None = None) -> None:
        from quant_warehouse import Warehouse
        from quant_warehouse.platforms.data_providers.fmp.synthetic_options import FmpSyntheticOptionSpec

        self.config = config or FmpSyntheticOptionReplayConfig()
        self.warehouse = Warehouse()
        self.spec = FmpSyntheticOptionSpec(tenor_days=(int(self.config.target_dte),))
        self._price_cache: dict[tuple[str, pd.Timestamp, pd.Timestamp], pd.DataFrame] = {}
        self.chain_read_count = 0
        self.price_read_count = 0
        self.selection_count = 0
        self.selected_priced_count = 0

    def select_rule_atm_entry(self, trade: pd.Series) -> pd.DataFrame:
        from quant_warehouse.platforms.data_providers.fmp.synthetic_options import read_fmp_synthetic_option_chain

        symbol = str(trade.get("symbol", "")).upper()
        entry_date = pd.Timestamp(trade.get("entry_date")).normalize()
        exit_date = pd.Timestamp(trade.get("exit_date")).normalize()
        side = str(trade.get("side", "long")).lower()
        option_type = self.config.option_type_for_short if side.startswith("short") else self.config.option_type_for_long
        prices = self._prices(symbol, entry_date, exit_date)
        chain = read_fmp_synthetic_option_chain(
            symbol,
            start_date=entry_date,
            end_date=entry_date,
            spec=self.spec,
            prices=prices,
            warehouse=self.warehouse,
        )
        self.chain_read_count += 1
        if chain.empty:
            return pd.DataFrame()
        candidates = chain.loc[chain["option_type"].astype(str).str.lower().eq(str(option_type).lower())].copy()
        if candidates.empty:
            return pd.DataFrame()
        candidates["dte_distance"] = (pd.to_numeric(candidates.get("dte"), errors="coerce") - float(self.config.target_dte)).abs()
        candidates["abs_moneyness"] = pd.to_numeric(candidates.get("abs_moneyness"), errors="coerce").fillna(np.inf)
        candidates = candidates.sort_values(["dte_distance", "abs_moneyness", "spread_pct", "contract_symbol"], kind="stable")
        row = candidates.iloc[0]
        entry_ask = _finite_float(row.get("ask"))
        entry_bid = _finite_float(row.get("bid"))
        entry_mid = _finite_float(row.get("mid"))
        if entry_ask is None or entry_ask <= 0.0:
            return pd.DataFrame()
        self.selection_count += 1
        return pd.DataFrame(
            [
                {
                    "trade_id": trade.get("trade_id"),
                    "symbol": symbol,
                    "side": "short" if side.startswith("short") else "long",
                    "entry_date": entry_date,
                    "equity_exit_date": exit_date,
                    "option_action": "buy_put" if str(option_type).lower().startswith("p") else "buy_call",
                    "option_type": str(option_type).lower(),
                    "contract_symbol": row.get("contract_symbol"),
                    "expiration": pd.Timestamp(row.get("expiration")).normalize(),
                    "strike": float(row.get("strike")),
                    "underlying_spot_entry": float(row.get("underlying_price")),
                    "entry_bid": float(entry_bid) if entry_bid is not None else np.nan,
                    "entry_ask": float(entry_ask),
                    "entry_mid": float(entry_mid) if entry_mid is not None else np.nan,
                    "entry_price": float(entry_ask),
                    "return_denominator": float(entry_ask),
                    "dte": int(row.get("dte")),
                    "moneyness": float(row.get("moneyness")),
                    "abs_moneyness": float(row.get("abs_moneyness")),
                    "option_source": row.get("option_source"),
                    "underlying_price_basis": row.get("underlying_price_basis"),
                    "equity_entry_price": _finite_float(trade.get("entry_price")),
                    "equity_exit_price": _finite_float(trade.get("exit_price")),
                    "ret_dec": _finite_float(trade.get("ret_dec")),
                    "entry_score": _finite_float(trade.get("entry_score")),
                    "top_k": trade.get("top_k"),
                }
            ]
        )

    def price_selected_options_with_paths(
        self,
        selected: pd.DataFrame,
        *,
        selector_name: str | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if selected.empty:
            return selected.copy(), pd.DataFrame()
        priced_rows: list[dict[str, Any]] = []
        path_frames: list[pd.DataFrame] = []
        for payload in selected.to_dict("records"):
            priced, path = self._price_payload(payload, selector_name=selector_name)
            if priced is not None:
                priced_rows.append(priced)
                self.selected_priced_count += 1
            if not path.empty:
                path_frames.append(path)
        return (
            pd.DataFrame(priced_rows) if priced_rows else pd.DataFrame(columns=selected.columns),
            pd.concat(path_frames, ignore_index=True) if path_frames else pd.DataFrame(),
        )

    def metrics(self) -> dict[str, float]:
        return {
            "fmp_synthetic_chain_read_count": float(self.chain_read_count),
            "fmp_synthetic_price_read_count": float(self.price_read_count),
            "fmp_synthetic_selection_count": float(self.selection_count),
            "fmp_synthetic_selected_priced_count": float(self.selected_priced_count),
        }

    def _price_payload(self, payload: dict[str, Any], *, selector_name: str | None) -> tuple[dict[str, Any] | None, pd.DataFrame]:
        symbol = str(payload.get("symbol", "")).upper()
        entry_date = pd.Timestamp(payload.get("entry_date")).normalize()
        equity_exit_date = pd.Timestamp(payload.get("equity_exit_date")).normalize()
        expiration = pd.Timestamp(payload.get("expiration")).normalize()
        target_exit = min(equity_exit_date, expiration)
        prices = self._prices(symbol, entry_date, equity_exit_date)
        path = self._price_path(payload, prices=prices, target_exit=target_exit, selector_name=selector_name)
        if path.empty:
            return None, path
        final = path.iloc[-1]
        entry_price = _finite_float(payload.get("entry_price"))
        exit_price = _finite_float(final.get("bid"))
        denominator = _finite_float(payload.get("return_denominator"))
        if entry_price is None or exit_price is None or denominator is None or denominator <= 0.0:
            return None, path
        pnl = float(exit_price) - float(entry_price)
        priced = dict(payload)
        priced.update(
            {
                "option_exit_date": pd.Timestamp(final["snapshot_date"]).normalize(),
                "exit_mid": float(final.get("mid")),
                "exit_bid": float(final.get("bid")),
                "exit_ask": float(final.get("ask")),
                "exit_price": float(exit_price),
                "exit_price_source": final.get("price_source"),
                "option_pnl": float(pnl),
                "option_return": float(pnl) / float(denominator),
                "path_observations": int(len(path)),
                "expired_before_equity_exit": bool(expiration < equity_exit_date),
            }
        )
        return priced, path

    def _price_path(
        self,
        payload: dict[str, Any],
        *,
        prices: pd.DataFrame,
        target_exit: pd.Timestamp,
        selector_name: str | None,
    ) -> pd.DataFrame:
        from quant_warehouse.platforms.data_providers.fmp.synthetic_options import (
            option_intrinsic_value,
            price_fmp_synthetic_contract,
        )

        if prices.empty:
            return pd.DataFrame()
        symbol = str(payload.get("symbol", "")).upper()
        entry_date = pd.Timestamp(payload.get("entry_date")).normalize()
        equity_exit_date = pd.Timestamp(payload.get("equity_exit_date")).normalize()
        expiration = pd.Timestamp(payload.get("expiration")).normalize()
        available_dates = pd.DatetimeIndex(prices.index).normalize()
        path_dates = available_dates[(available_dates >= entry_date) & (available_dates <= target_exit)]
        if len(path_dates) == 0:
            return pd.DataFrame()
        rows: list[dict[str, Any]] = []
        for snapshot_date in path_dates:
            quote = price_fmp_synthetic_contract(
                symbol=symbol,
                snapshot_date=snapshot_date,
                option_type=str(payload.get("option_type")),
                strike=float(payload.get("strike")),
                expiration=expiration,
                spec=self.spec,
                prices=prices,
                warehouse=self.warehouse,
            )
            if quote is None:
                continue
            source = "contract_quote"
            bid = float(quote["bid"])
            ask = float(quote["ask"])
            mid = float(quote["mid"])
            if snapshot_date == path_dates[-1] and equity_exit_date >= expiration:
                spot = _finite_float(prices.loc[snapshot_date, "close"])
                if spot is not None:
                    intrinsic = option_intrinsic_value(
                        option_type=str(payload.get("option_type")),
                        strike=float(payload.get("strike")),
                        underlying_price=float(spot),
                    )
                    bid = ask = mid = float(intrinsic)
                    source = "expiration_intrinsic"
            rows.append(
                {
                    "snapshot_date": pd.Timestamp(snapshot_date).normalize(),
                    "contract_symbol": payload.get("contract_symbol"),
                    "bid": bid,
                    "ask": ask,
                    "mid": mid,
                    "underlying_price": float(quote["underlying_price"]),
                    "iv": quote.get("iv"),
                    "price_source": source,
                }
            )
        if not rows:
            return pd.DataFrame()
        path = pd.DataFrame(rows)
        entry_price = _finite_float(payload.get("entry_price"))
        denominator = _finite_float(payload.get("return_denominator"))
        path["mark_price"] = pd.to_numeric(path["bid"], errors="coerce")
        path["path_pnl"] = path["mark_price"] - float(entry_price) if entry_price is not None else np.nan
        path["path_return"] = path["path_pnl"] / float(denominator) if denominator is not None and denominator > 0.0 else np.nan
        path["selector"] = selector_name
        path["trade_id"] = payload.get("trade_id")
        path["symbol"] = symbol
        path["side"] = payload.get("side")
        path["option_action"] = payload.get("option_action")
        path["option_type"] = payload.get("option_type")
        path["entry_date"] = entry_date
        path["equity_exit_date"] = equity_exit_date
        path["option_exit_date"] = pd.Timestamp(path["snapshot_date"].iloc[-1]).normalize()
        path["expiration"] = expiration
        path["strike"] = float(payload.get("strike"))
        path["entry_price"] = float(entry_price) if entry_price is not None else np.nan
        path["return_denominator"] = float(denominator) if denominator is not None else np.nan
        path["expired_before_equity_exit"] = bool(expiration < equity_exit_date)
        return path.sort_values(["snapshot_date", "contract_symbol"]).reset_index(drop=True)

    def _prices(self, symbol: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
        fetch_start = pd.Timestamp(start_date).normalize() - pd.Timedelta(days=max(int(self.spec.realized_vol_window) * 3, 10))
        fetch_end = pd.Timestamp(end_date).normalize()
        key = (str(symbol).upper(), fetch_start, fetch_end)
        cached = self._price_cache.get(key)
        if cached is not None:
            return cached.copy()
        raw = self.warehouse.read_prices(str(symbol).upper(), provider="fmp", start=fetch_start, end=fetch_end)
        frame = clean_price_frame(raw)
        self.price_read_count += 1
        self._price_cache[key] = frame.copy()
        return frame


def _normalize_trade_windows(trade_windows: pd.DataFrame) -> pd.DataFrame:
    if trade_windows is None or trade_windows.empty:
        return pd.DataFrame()
    required = {"symbol", "entry_date", "exit_date"}
    missing = required.difference(trade_windows.columns)
    if missing:
        raise ValueError(f"trade windows missing required columns: {sorted(missing)}")
    out = trade_windows.copy()
    if "trade_id" not in out.columns:
        out["trade_id"] = [f"trade_{idx + 1}" for idx in range(len(out))]
    if "side" not in out.columns:
        out["side"] = "long"
    out["symbol"] = out["symbol"].astype(str).str.upper()
    out["entry_date"] = pd.to_datetime(out["entry_date"], errors="coerce").dt.normalize()
    out["exit_date"] = pd.to_datetime(out["exit_date"], errors="coerce").dt.normalize()
    out = out.dropna(subset=["symbol", "entry_date", "exit_date"])
    out = out.loc[out["exit_date"].ge(out["entry_date"])].copy()
    return out.reset_index(drop=True)


def _finite_float(value: object) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None
