from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from quant_warehouse.target_engineering.thetadata_loader import (
    ThetaDataDownloadSpec,
    load_thetadata_option_snapshots,
)

from quant_orchestrator.data import load_ohlcv


@dataclass(frozen=True)
class OptopsyBacktestSpec:
    symbols: tuple[str, ...]
    start_date: str
    end_date: str
    strategy: str = "long_calls"
    max_entry_dte: int = 60
    exit_dte: int = 30
    exit_dte_tolerance: int = 2
    min_bid_ask: float = 0.05
    delta_target: float = 0.50
    delta_min: float = 0.30
    delta_max: float = 0.70
    capital: float = 100_000.0
    quantity: int = 1
    max_positions: int = 5
    multiplier: int = 100
    selector: str = "nearest"
    use_cache: bool = True
    download_missing: bool = False
    allow_delta_proxy: bool = False
    price_provider: str = "fmp"
    strike_range: int = 10
    extra_strategy_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OptopsyBacktestResult:
    spec: OptopsyBacktestSpec
    options: pd.DataFrame
    raw_trades: pd.DataFrame
    trade_log: pd.DataFrame
    equity_curve: pd.Series
    summary: dict[str, Any]


def load_thetadata_options_for_optopsy(
    symbols: Sequence[str],
    *,
    start_date: str,
    end_date: str,
    max_dte: int = 60,
    strike_range: int = 10,
    use_cache: bool = True,
    download_missing: bool = False,
    allow_delta_proxy: bool = False,
    price_provider: str = "fmp",
) -> pd.DataFrame:
    """Load ThetaData EOD options through quant-warehouse and map to Optopsy schema."""

    dates = [ts.normalize() for ts in pd.date_range(start_date, end_date, freq="B")]
    if not dates:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    spec = ThetaDataDownloadSpec(max_dte=max_dte, strike_range=strike_range)
    for symbol in _normalize_symbols(symbols):
        snapshots = load_thetadata_option_snapshots(
            symbol,
            dates,
            max_dte=max_dte,
            strike_range=strike_range,
            use_cache=use_cache,
            download_spec=spec,
            download_missing=download_missing,
        )
        for snapshot_date, chain in snapshots.items():
            if chain is None or chain.empty:
                continue
            frame = chain.copy()
            frame["snapshot_date"] = pd.Timestamp(snapshot_date).normalize()
            frames.append(frame)

    if not frames:
        return pd.DataFrame()
    return theta_chain_to_optopsy(
        pd.concat(frames, ignore_index=True),
        allow_delta_proxy=allow_delta_proxy,
        price_provider=price_provider,
    )


def theta_chain_to_optopsy(
    chain: pd.DataFrame,
    *,
    allow_delta_proxy: bool = False,
    price_provider: str = "fmp",
) -> pd.DataFrame:
    """Convert normalized quant-warehouse ThetaData chains to Optopsy columns."""

    if chain is None or chain.empty:
        return pd.DataFrame()
    out = chain.copy()
    out.columns = [str(col).strip().lower() for col in out.columns]
    if "snapshot_date" in out.columns and "eod_date" in out.columns:
        out = out.drop(columns=["eod_date"])

    rename = {
        "snapshot_date": "quote_date",
        "eod_date": "quote_date",
        "right": "option_type",
        "symbol": "underlying_symbol",
        "iv": "implied_volatility",
        "implied_vol": "implied_volatility",
        "open_interest": "open_interest",
    }
    out = out.rename(columns={key: value for key, value in rename.items() if key in out.columns})
    out = out.loc[:, ~out.columns.duplicated()].copy()
    required = {"underlying_symbol", "option_type", "expiration", "quote_date", "strike", "bid", "ask"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"ThetaData option chain is missing required columns: {sorted(missing)}")
    if "delta" not in out.columns:
        if not allow_delta_proxy:
            raise ValueError(
                "ThetaData option chain is missing delta; Optopsy requires a delta column. "
                "Set allow_delta_proxy=True to select contracts with a moneyness-based proxy "
                "while still using real ThetaData bid/ask prices for P&L.",
            )
        out["delta"] = _proxy_delta_from_moneyness(out, price_provider=price_provider)

    keep = [
        "underlying_symbol",
        "option_type",
        "expiration",
        "quote_date",
        "strike",
        "bid",
        "ask",
        "delta",
        "implied_volatility",
        "gamma",
        "theta",
        "vega",
        "volume",
        "open_interest",
    ]
    keep = [col for col in keep if col in out.columns]
    out = out.loc[:, keep].copy()
    out["underlying_symbol"] = out["underlying_symbol"].astype(str).str.strip().str.upper()
    out["option_type"] = out["option_type"].astype(str).str.strip().str.lower()
    out["option_type"] = out["option_type"].replace({"c": "call", "p": "put"})
    out["expiration"] = pd.to_datetime(out["expiration"], errors="coerce").dt.normalize()
    out["quote_date"] = pd.to_datetime(out["quote_date"], errors="coerce").dt.normalize()
    for col in ("strike", "bid", "ask", "delta", "implied_volatility", "gamma", "theta", "vega", "volume", "open_interest"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.loc[
        out["underlying_symbol"].ne("")
        & out["option_type"].isin(["call", "put"])
        & out["expiration"].notna()
        & out["quote_date"].notna()
        & out["strike"].notna()
        & out["bid"].notna()
        & out["ask"].notna()
        & out["delta"].notna()
        & out["ask"].gt(0)
    ].copy()
    return out.sort_values(["underlying_symbol", "quote_date", "expiration", "option_type", "strike"]).reset_index(drop=True)


def run_optopsy_backtest(spec: OptopsyBacktestSpec) -> OptopsyBacktestResult:
    """Run an Optopsy options backtest using ThetaData loaded by quant-warehouse."""

    import optopsy as op

    options = load_thetadata_options_for_optopsy(
        spec.symbols,
        start_date=spec.start_date,
        end_date=spec.end_date,
        max_dte=spec.max_entry_dte,
        strike_range=spec.strike_range,
        use_cache=spec.use_cache,
        download_missing=spec.download_missing,
        allow_delta_proxy=spec.allow_delta_proxy,
        price_provider=spec.price_provider,
    )
    if options.empty:
        raise RuntimeError(
            "No ThetaData option rows were loaded. Backfill quant-warehouse ThetaData "
            "cache or set download_missing=True."
        )

    strategy = _resolve_optopsy_strategy(op, spec.strategy)
    strategy_kwargs = {
        "max_entry_dte": int(spec.max_entry_dte),
        "exit_dte": int(spec.exit_dte),
        "exit_dte_tolerance": int(spec.exit_dte_tolerance),
        "min_bid_ask": float(spec.min_bid_ask),
        "leg1_delta": op.TargetRange(
            target=float(spec.delta_target),
            min=float(spec.delta_min),
            max=float(spec.delta_max),
        ),
        **dict(spec.extra_strategy_kwargs),
    }
    raw_trades = strategy(options, raw=True, **strategy_kwargs)
    simulation = op.simulate(
        options,
        strategy,
        capital=float(spec.capital),
        quantity=int(spec.quantity),
        max_positions=int(spec.max_positions),
        multiplier=int(spec.multiplier),
        selector=spec.selector,
        **strategy_kwargs,
    )
    return OptopsyBacktestResult(
        spec=spec,
        options=options,
        raw_trades=raw_trades,
        trade_log=simulation.trade_log,
        equity_curve=simulation.equity_curve,
        summary=dict(simulation.summary),
    )


def summarize_optopsy_result(result: OptopsyBacktestResult) -> pd.DataFrame:
    summary = {
        "strategy": result.spec.strategy,
        "symbols": ",".join(result.spec.symbols),
        "start_date": result.spec.start_date,
        "end_date": result.spec.end_date,
        "option_rows": int(len(result.options)),
        "raw_trades": int(len(result.raw_trades)),
        "closed_trades": int(len(result.trade_log)),
        **result.summary,
    }
    return pd.DataFrame([summary])


def _resolve_optopsy_strategy(optopsy_module: Any, name: str) -> Callable[..., pd.DataFrame]:
    strategy_name = str(name).strip()
    if not strategy_name:
        raise ValueError("Optopsy strategy name is required")
    try:
        strategy = getattr(optopsy_module, strategy_name)
    except AttributeError as exc:
        raise ValueError(f"Unknown Optopsy strategy: {strategy_name}") from exc
    if not callable(strategy):
        raise ValueError(f"Optopsy attribute is not callable: {strategy_name}")
    return strategy


def _normalize_symbols(symbols: Sequence[str]) -> tuple[str, ...]:
    return tuple(
        sorted({str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()}),
    )


def _proxy_delta_from_moneyness(chain: pd.DataFrame, *, price_provider: str) -> pd.Series:
    """Build an absolute-delta proxy when ThetaData chains lack greeks."""

    frame = chain.copy()
    frame["quote_date"] = pd.to_datetime(
        frame.get("quote_date", frame.get("snapshot_date")),
        errors="coerce",
    ).dt.normalize()
    frame["strike"] = pd.to_numeric(frame["strike"], errors="coerce")
    spots = _load_underlying_spots(
        frame["underlying_symbol"].dropna().astype(str).str.upper().unique(),
        frame["quote_date"].min(),
        frame["quote_date"].max(),
        provider=price_provider,
    )
    spot_values = [
        spots.get((str(symbol).upper(), pd.Timestamp(date).normalize()))
        for symbol, date in zip(frame["underlying_symbol"], frame["quote_date"], strict=False)
    ]
    spot = pd.Series(spot_values, index=frame.index, dtype="float64")
    fallback_spot = frame.groupby("quote_date")["strike"].transform("median")
    spot = spot.fillna(fallback_spot)
    scale = (spot * 0.10).clip(lower=1.0)
    call_delta = 1.0 / (1.0 + np.exp(-(spot - frame["strike"]) / scale))
    put_delta = -(1.0 / (1.0 + np.exp((spot - frame["strike"]) / scale)))
    option_type = frame["option_type"].astype(str).str.lower()
    return pd.Series(np.where(option_type.str.startswith("p"), put_delta, call_delta), index=frame.index)


def _load_underlying_spots(
    symbols: Iterable[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    provider: str,
) -> dict[tuple[str, pd.Timestamp], float]:
    spots: dict[tuple[str, pd.Timestamp], float] = {}
    if pd.isna(start) or pd.isna(end):
        return spots
    for symbol in _normalize_symbols(tuple(symbols)):
        try:
            prices = load_ohlcv(
                symbol,
                provider=provider,
                start=pd.Timestamp(start).date().isoformat(),
                end=pd.Timestamp(end).date().isoformat(),
            )
        except Exception:
            continue
        if prices is None or prices.empty or "close" not in prices:
            continue
        close = pd.to_numeric(prices["close"], errors="coerce")
        for date, value in close.dropna().items():
            spots[(symbol, pd.Timestamp(date).normalize())] = float(value)
    return spots
