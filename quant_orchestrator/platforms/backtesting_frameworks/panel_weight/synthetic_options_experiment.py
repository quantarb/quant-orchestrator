from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Callable, Literal

import numpy as np
import pandas as pd

try:
    from scipy.special import ndtr as _scipy_ndtr
except Exception:  # pragma: no cover - scipy is optional at import time
    _scipy_ndtr = None

from quant_orchestrator.platforms.backtesting_frameworks.panel_weight.synthetic_backtest import (
    backtest_positions_with_directional_asset_returns,
    prepare_backtest_position_state,
    resolve_component_cols,
    resolve_short_score_col,
    run_top_k_long_only_score_rule,
    run_top_k_long_short_score_rule,
    run_top_k_momentum_baseline,
    summarize_curve,
)
from quant_orchestrator.platforms.backtesting_frameworks.panel_weight.synthetic_options import (
    build_realized_vol_panel,
    build_synthetic_option_return_panels,
)

FamilyPanelBuilder = Callable[[str], pd.DataFrame]
OptionChainLoader = Callable[[str, pd.Timestamp, pd.Timestamp], pd.DataFrame]
OptionPricingMode = Literal["maturing", "constant_maturity", "real_quotes"]


@dataclass(frozen=True)
class SyntheticOptionsBacktestConfig:
    score_col: str = "prob_buy"
    component_threshold: float = 0.50
    top_k_values: tuple[int, ...] = (5, 10, 20, 40)
    strategy_variants: tuple[str, ...] = ("classifier_prob", "momentum_21d")
    baseline_lookback_days: int = 21
    tenor_days: int = 60
    option_pricing_mode: OptionPricingMode = "maturing"
    option_chain_loader: OptionChainLoader | None = None
    real_quote_entry_price_col: str = "ask"
    real_quote_exit_price_col: str = "bid"
    real_quote_fallback_to_synthetic: bool = False
    enforce_option_capacity: bool = False
    option_contract_multiplier: float = 100.0
    max_volume_participation: float = 0.10
    max_open_interest_participation: float = 0.02
    realized_vol_window: int = 21
    vol_floor: float | None = 0.15
    vol_cap: float | None = 0.80
    iv_multiplier: float = 1.0
    rate: float = 0.0
    premium_floor: float = 0.25
    option_buckets: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            "atm_option": {"long_strike_multiplier": 1.00, "short_strike_multiplier": 1.00},
            "otm_option": {"long_strike_multiplier": 1.05, "short_strike_multiplier": 0.95},
            "ditm_option": {"long_strike_multiplier": 0.90, "short_strike_multiplier": 1.10},
        },
    )
    initial_balance: float = 100000.0
    fee_bps: float = 5.0
    slippage_bps: float = 5.0


@dataclass(frozen=True)
class SyntheticOptionsBacktestResult:
    config: SyntheticOptionsBacktestConfig
    summary: pd.DataFrame
    yearly_summary: pd.DataFrame
    variant_runs: dict[tuple[str, str, int], dict[str, object]]
    close_panel: pd.DataFrame
    realized_vol_panel: pd.DataFrame
    synthetic_price_panels: dict[str, dict[str, object]]
    real_quote_coverage: pd.DataFrame


def run_synthetic_options_backtest(
    bt_panel: pd.DataFrame,
    *,
    config: SyntheticOptionsBacktestConfig | None = None,
    family_names: tuple[str, ...] = (),
    family_panel_builder: FamilyPanelBuilder | None = None,
) -> SyntheticOptionsBacktestResult:
    """Run the synthetic option comparison loop.

    `bt_panel` is expected to be indexed by `(date, symbol)` and contain `close`,
    `prob_buy`, `prob_short`, and the configured score/component columns.
    """

    cfg = config or SyntheticOptionsBacktestConfig()
    panel = _normalize_backtest_panel(bt_panel)
    score_col = str(cfg.score_col)
    short_score_col = resolve_short_score_col(score_col)
    component_cols = resolve_component_cols(score_col)
    short_component_cols = resolve_component_cols(short_score_col)
    years = _panel_years(panel)
    close_panel = _build_close_panel(panel)
    realized_vol_panel = build_realized_vol_panel(
        close_panel,
        window=int(cfg.realized_vol_window),
        vol_floor=cfg.vol_floor,
        vol_cap=cfg.vol_cap,
    )
    instrument_return_panels, synthetic_price_panels = build_synthetic_option_return_panels(
        close_panel,
        realized_vol=realized_vol_panel,
        option_buckets=cfg.option_buckets,
        tenor_days=int(cfg.tenor_days),
        iv_multiplier=float(cfg.iv_multiplier),
        premium_floor=float(cfg.premium_floor),
    )

    variant_runs: dict[tuple[str, str, int], dict[str, object]] = {}
    summary_rows: list[dict[str, object]] = []
    yearly_rows: list[pd.DataFrame] = []
    option_chain_cache: dict[tuple[str, pd.Timestamp, pd.Timestamp], pd.DataFrame] = {}
    real_quote_coverage_rows: list[dict[str, object]] = []

    for strategy_spec in _strategy_specs(cfg, family_names):
        strategy_name = str(strategy_spec["strategy"])
        for top_k in cfg.top_k_values:
            signal_run = _build_signal_run(
                strategy_spec,
                panel=panel,
                score_col=score_col,
                short_score_col=short_score_col,
                component_cols=component_cols,
                short_component_cols=short_component_cols,
                config=cfg,
                top_k=int(top_k),
                family_panel_builder=family_panel_builder,
            )
            if signal_run is None:
                continue
            positions = signal_run["positions"]
            position_state = prepare_backtest_position_state(positions)
            prior_positions = positions.shift(1).fillna(0)
            counts = _position_counts(positions, prior_positions)
            for instrument_name in instrument_return_panels:
                asset_returns = _resolve_instrument_returns(
                    instrument_name,
                    positions=positions,
                    constant_maturity_returns=instrument_return_panels,
                    close_panel=close_panel,
                    realized_vol_panel=realized_vol_panel,
                    config=cfg,
                    option_chain_cache=option_chain_cache,
                    coverage_rows=real_quote_coverage_rows,
                    strategy_name=strategy_name,
                    top_k=int(top_k),
                )
                backtest = backtest_positions_with_directional_asset_returns(
                    positions,
                    asset_returns["long"],
                    short_asset_returns=asset_returns["short"],
                    initial_balance=float(cfg.initial_balance),
                    fee_bps=float(cfg.fee_bps),
                    slippage_bps=float(cfg.slippage_bps),
                    position_state=position_state,
                )
                mode = f"{strategy_name}_{instrument_name}_top_{top_k}"
                curve_summary = summarize_curve(backtest["returns"], years, mode=mode)
                variant_runs[(strategy_name, instrument_name, int(top_k))] = {
                    "positions": positions,
                    "backtest": backtest,
                    "summary": curve_summary,
                }
                summary_rows.append(
                    {
                        "strategy": strategy_name,
                        "instrument": instrument_name,
                        "top_k": int(top_k),
                        "total_return_pct": curve_summary["total_return_pct"],
                        "sharpe": curve_summary["sharpe"],
                        "max_drawdown_pct": curve_summary["max_drawdown_pct"],
                        **counts,
                        "avg_turnover": (
                            float(backtest["turnover"].mean())
                            if len(backtest["turnover"])
                            else np.nan
                        ),
                    },
                )
                yearly_df = curve_summary["yearly_df"].copy()
                yearly_df.insert(0, "strategy", strategy_name)
                yearly_df.insert(1, "instrument", instrument_name)
                yearly_df.insert(2, "top_k", int(top_k))
                yearly_rows.append(yearly_df)

    summary = (
        pd.DataFrame(summary_rows)
        .sort_values(["strategy", "instrument", "top_k"])
        .reset_index(drop=True)
        if summary_rows
        else pd.DataFrame()
    )
    yearly_summary = pd.concat(yearly_rows, ignore_index=True) if yearly_rows else pd.DataFrame()
    real_quote_coverage = (
        pd.DataFrame(real_quote_coverage_rows)
        .sort_values(["strategy", "instrument", "top_k", "side", "symbol"])
        .reset_index(drop=True)
        if real_quote_coverage_rows
        else pd.DataFrame()
    )
    return SyntheticOptionsBacktestResult(
        config=cfg,
        summary=summary,
        yearly_summary=yearly_summary,
        variant_runs=variant_runs,
        close_panel=close_panel,
        realized_vol_panel=realized_vol_panel,
        synthetic_price_panels=synthetic_price_panels,
        real_quote_coverage=real_quote_coverage,
    )


def _build_signal_run(
    strategy_spec: dict[str, object],
    *,
    panel: pd.DataFrame,
    score_col: str,
    short_score_col: str,
    component_cols: list[str],
    short_component_cols: list[str],
    config: SyntheticOptionsBacktestConfig,
    top_k: int,
    family_panel_builder: FamilyPanelBuilder | None,
) -> dict[str, pd.DataFrame] | None:
    kind = str(strategy_spec["kind"])
    if kind == "ml_long_only":
        return run_top_k_long_only_score_rule(
            panel=panel,
            score_col=score_col,
            component_cols=component_cols,
            component_threshold=float(config.component_threshold),
            price_col="close",
            top_k=top_k,
        )
    if kind == "ml_long_short":
        return run_top_k_long_short_score_rule(
            panel=panel,
            long_score_col=score_col,
            short_score_col=short_score_col,
            long_component_cols=component_cols,
            short_component_cols=short_component_cols,
            component_threshold=float(config.component_threshold),
            price_col="close",
            top_k=top_k,
        )
    if kind == "family_model":
        if family_panel_builder is None:
            return None
        family_panel = family_panel_builder(str(strategy_spec["family_name"]))
        if family_panel is None or family_panel.empty:
            return None
        family_panel = _normalize_backtest_panel(family_panel)
        return run_top_k_long_short_score_rule(
            panel=family_panel,
            long_score_col="prob_buy",
            short_score_col="prob_short",
            long_component_cols=["prob_buy"],
            short_component_cols=["prob_short"],
            component_threshold=float(config.component_threshold),
            price_col="close",
            top_k=top_k,
        )
    return run_top_k_momentum_baseline(
        panel=panel,
        price_col="close",
        top_k=top_k,
        lookback_days=int(strategy_spec.get("lookback_days", config.baseline_lookback_days)),
    )


def _strategy_specs(
    config: SyntheticOptionsBacktestConfig,
    family_names: tuple[str, ...],
) -> list[dict[str, object]]:
    specs: list[dict[str, object]] = []
    if "classifier_prob" in config.strategy_variants or "raw_pct6" in config.strategy_variants:
        specs.append({"strategy": "classifier_prob_long_only", "kind": "ml_long_only"})
        specs.append({"strategy": "classifier_prob_long_short", "kind": "ml_long_short"})
    if "momentum_21d" in config.strategy_variants:
        specs.append(
            {
                "strategy": "momentum_21d",
                "kind": "momentum",
                "lookback_days": int(config.baseline_lookback_days),
            },
        )
    for family_name in family_names:
        specs.append(
            {
                "strategy": f"family::{family_name}",
                "kind": "family_model",
                "family_name": str(family_name),
            },
        )
    return specs


def _resolve_instrument_returns(
    instrument_name: str,
    *,
    positions: pd.DataFrame,
    constant_maturity_returns: dict[str, dict[str, pd.DataFrame]],
    close_panel: pd.DataFrame,
    realized_vol_panel: pd.DataFrame,
    config: SyntheticOptionsBacktestConfig,
    option_chain_cache: dict[tuple[str, pd.Timestamp, pd.Timestamp], pd.DataFrame],
    coverage_rows: list[dict[str, object]],
    strategy_name: str,
    top_k: int,
) -> dict[str, pd.DataFrame]:
    if instrument_name == "equity" or config.option_pricing_mode == "constant_maturity":
        return constant_maturity_returns[instrument_name]
    bucket = config.option_buckets[instrument_name]
    if config.option_pricing_mode == "real_quotes":
        return {
            "long": _build_real_quote_option_return_panel(
                positions,
                close_panel,
                realized_vol_panel,
                strike_multiplier=float(bucket["long_strike_multiplier"]),
                option_type="call",
                side=1,
                config=config,
                option_chain_cache=option_chain_cache,
                coverage_rows=coverage_rows,
                coverage_context={
                    "strategy": strategy_name,
                    "instrument": instrument_name,
                    "top_k": int(top_k),
                    "side": "long",
                },
            ),
            "short": _build_real_quote_option_return_panel(
                positions,
                close_panel,
                realized_vol_panel,
                strike_multiplier=float(bucket["short_strike_multiplier"]),
                option_type="put",
                side=-1,
                config=config,
                option_chain_cache=option_chain_cache,
                coverage_rows=coverage_rows,
                coverage_context={
                    "strategy": strategy_name,
                    "instrument": instrument_name,
                    "top_k": int(top_k),
                    "side": "short",
                },
            ),
        }
    return {
        "long": _build_maturing_option_return_panel(
            positions,
            close_panel,
            realized_vol_panel,
            strike_multiplier=float(bucket["long_strike_multiplier"]),
            option_type="call",
            side=1,
            config=config,
        ),
        "short": _build_maturing_option_return_panel(
            positions,
            close_panel,
            realized_vol_panel,
            strike_multiplier=float(bucket["short_strike_multiplier"]),
            option_type="put",
            side=-1,
            config=config,
        ),
    }


def _build_real_quote_option_return_panel(
    positions: pd.DataFrame,
    close_panel: pd.DataFrame,
    realized_vol_panel: pd.DataFrame,
    *,
    strike_multiplier: float,
    option_type: Literal["call", "put"],
    side: int,
    config: SyntheticOptionsBacktestConfig,
    option_chain_cache: dict[tuple[str, pd.Timestamp, pd.Timestamp], pd.DataFrame],
    coverage_rows: list[dict[str, object]],
    coverage_context: dict[str, object],
) -> pd.DataFrame:
    dates = pd.DatetimeIndex(positions.index)
    symbols = list(positions.columns)
    close = close_panel.reindex(index=dates, columns=symbols).astype(float)
    if bool(config.real_quote_fallback_to_synthetic):
        out = _build_maturing_option_return_panel(
            positions,
            close_panel,
            realized_vol_panel,
            strike_multiplier=strike_multiplier,
            option_type=option_type,
            side=side,
            config=config,
        )
    else:
        out = pd.DataFrame(0.0, index=dates, columns=symbols, dtype=float)
    tenor_days = max(int(config.tenor_days), 1)

    for symbol in symbols:
        signal = positions[symbol].reindex(dates).fillna(0).astype(int).to_numpy()
        desired = signal == int(side)
        if not bool(desired.any()):
            continue
        spot_values = close[symbol].to_numpy(dtype=float)
        chain = _load_option_chain_frame(
            symbol,
            dates.min(),
            dates.max(),
            config=config,
            option_chain_cache=option_chain_cache,
        )
        entry_price_col = str(config.real_quote_entry_price_col)
        exit_price_col = str(config.real_quote_exit_price_col)
        chain = _normalize_quote_chain_frame(
            chain,
            required_price_cols=(entry_price_col, exit_price_col),
        )
        if chain.empty:
            _append_quote_coverage(
                coverage_rows,
                coverage_context,
                symbol=symbol,
                selected_segments=0,
                fallback_segments=int(_count_position_runs(desired)),
                quote_return_count=0,
                fallback_return_count=int(desired.sum()),
            )
            continue
        quote_by_contract = {
            str(contract): group.sort_values("snapshot_date").copy()
            for contract, group in chain.groupby("contract_symbol", sort=False)
        }
        quote_by_snapshot = {
            pd.Timestamp(snapshot_date).normalize(): group.copy()
            for snapshot_date, group in chain.groupby("snapshot_date", sort=False)
        }
        returns = out[symbol].to_numpy(dtype=float).copy()
        selected_segments = 0
        fallback_segments = 0
        quote_return_count = 0
        fallback_return_count = 0
        capacity_fill_sum = 0.0
        capacity_fill_count = 0
        cursor = 0
        while cursor < len(dates):
            if not desired[cursor] or not np.isfinite(spot_values[cursor]) or spot_values[cursor] <= 0.0:
                cursor += 1
                continue
            run_end = cursor
            while run_end + 1 < len(dates) and desired[run_end + 1]:
                run_end += 1
            segment_start = cursor
            while segment_start <= run_end:
                segment_end = min(run_end, segment_start + tenor_days)
                idx = np.arange(segment_start, segment_end + 1)
                entry_date = pd.Timestamp(dates[segment_start]).normalize()
                entry_snapshot = quote_by_snapshot.get(entry_date, pd.DataFrame())
                contract = _select_entry_contract(
                    entry_snapshot,
                    entry_date=entry_date,
                    option_type=option_type,
                    target_strike=float(spot_values[segment_start]) * float(strike_multiplier),
                    target_dte=tenor_days,
                    entry_price_col=entry_price_col,
                    exit_price_col=exit_price_col,
                )
                if contract is None:
                    fallback_segments += 1
                    fallback_return_count += max(len(idx) - 1, 0)
                    if segment_end >= run_end:
                        break
                    segment_start = segment_end
                    continue

                selected_segments += 1
                fill_fraction = _capacity_fill_fraction(
                    contract,
                    active_names=int(positions.loc[dates[segment_start]].abs().sum()),
                    positions=positions,
                    date=dates[segment_start],
                    config=config,
                    entry_price_col=entry_price_col,
                )
                prices = _real_quote_segment_prices(
                    contract,
                    quote_by_contract.get(str(contract["contract_symbol"]), pd.DataFrame()),
                    dates=dates[idx],
                    spot_values=spot_values[idx],
                    option_type=option_type,
                    entry_price_col=entry_price_col,
                    exit_price_col=exit_price_col,
                )
                previous = prices[:-1]
                current = prices[1:]
                valid = np.isfinite(previous) & np.isfinite(current) & (previous > 0.0)
                if valid.any():
                    segment_returns = returns[idx[1:]].copy()
                    segment_returns[valid] = ((current[valid] / previous[valid]) - 1.0) * fill_fraction
                    returns[idx[1:]] = segment_returns
                    quote_return_count += int(valid.sum())
                    fallback_return_count += int((~valid).sum())
                    capacity_fill_sum += float(fill_fraction) * int(valid.sum())
                    capacity_fill_count += int(valid.sum())
                else:
                    fallback_segments += 1
                    fallback_return_count += len(valid)
                if segment_end >= run_end:
                    break
                segment_start = segment_end
            cursor = run_end + 1
        out[symbol] = returns
        _append_quote_coverage(
            coverage_rows,
            coverage_context,
            symbol=symbol,
            selected_segments=selected_segments,
            fallback_segments=fallback_segments,
            quote_return_count=quote_return_count,
            fallback_return_count=fallback_return_count,
            avg_capacity_fill_fraction=(
                float(capacity_fill_sum / capacity_fill_count) if capacity_fill_count else np.nan
            ),
        )
    return out


def _load_option_chain_frame(
    symbol: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    *,
    config: SyntheticOptionsBacktestConfig,
    option_chain_cache: dict[tuple[str, pd.Timestamp, pd.Timestamp], pd.DataFrame],
) -> pd.DataFrame:
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    key = (str(symbol).upper(), start, end)
    if key in option_chain_cache:
        return option_chain_cache[key]
    loader = config.option_chain_loader
    if loader is None:
        try:
            from quant_warehouse.platforms.data_providers.thetadata.options import read_option_chain_arctic
        except Exception:
            frame = pd.DataFrame()
        else:
            frame = read_option_chain_arctic(
                str(symbol).upper(),
                start_date=start,
                end_date=end,
                columns=[
                    "snapshot_date",
                    "contract_symbol",
                    "expiration",
                    "strike",
                    "option_type",
                    "bid",
                    "ask",
                    "mid",
                    "volume",
                    "open_interest",
                ],
            )
    else:
        frame = loader(str(symbol).upper(), start, end)
    option_chain_cache[key] = pd.DataFrame() if frame is None else frame.copy()
    return option_chain_cache[key]


def _capacity_fill_fraction(
    contract: dict[str, object],
    *,
    active_names: int,
    positions: pd.DataFrame,
    date: pd.Timestamp,
    config: SyntheticOptionsBacktestConfig,
    entry_price_col: str,
) -> float:
    _ = positions, date
    if not bool(config.enforce_option_capacity):
        return 1.0
    entry_price = _finite_positive(contract.get(entry_price_col))
    if entry_price is None:
        return 0.0
    target_names = max(int(active_names), 1)
    target_dollars = float(config.initial_balance) / float(target_names)
    target_contracts = target_dollars / (entry_price * float(config.option_contract_multiplier))
    caps: list[float] = [target_contracts]
    volume = _finite_positive(contract.get("volume"))
    if volume is not None:
        caps.append(volume * max(float(config.max_volume_participation), 0.0))
    open_interest = _finite_positive(contract.get("open_interest"))
    if open_interest is not None:
        caps.append(open_interest * max(float(config.max_open_interest_participation), 0.0))
    fillable_contracts = max(0.0, min(caps))
    if target_contracts <= 0.0:
        return 0.0
    return float(np.clip(fillable_contracts / target_contracts, 0.0, 1.0))


def _finite_positive(value: object) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out) or out <= 0.0:
        return None
    return out


def _normalize_quote_chain_frame(
    frame: pd.DataFrame,
    *,
    required_price_cols: tuple[str, ...],
) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    out = frame.copy()
    out.columns = [str(column).strip().lower() for column in out.columns]
    if "right" in out.columns and "option_type" not in out.columns:
        out = out.rename(columns={"right": "option_type"})
    required = {"snapshot_date", "contract_symbol", "expiration", "strike", "option_type"}
    if not required.issubset(out.columns):
        return pd.DataFrame()
    out["snapshot_date"] = pd.to_datetime(out["snapshot_date"], errors="coerce").dt.normalize()
    out["expiration"] = pd.to_datetime(out["expiration"], errors="coerce").dt.normalize()
    out["strike"] = pd.to_numeric(out["strike"], errors="coerce")
    out["option_type"] = out["option_type"].astype(str).str.lower().str.strip()
    out["contract_symbol"] = out["contract_symbol"].astype(str)
    if "mid" not in out.columns and {"bid", "ask"}.issubset(out.columns):
        out["mid"] = (
            pd.to_numeric(out["bid"], errors="coerce") + pd.to_numeric(out["ask"], errors="coerce")
        ) / 2.0
    missing_prices = [column for column in required_price_cols if column not in out.columns]
    if missing_prices:
        return pd.DataFrame()
    for column in set(required_price_cols):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    for column in ("volume", "open_interest"):
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.dropna(subset=["snapshot_date", "expiration", "strike", "contract_symbol", *required_price_cols])
    valid_prices = pd.Series(True, index=out.index)
    for column in set(required_price_cols):
        valid_prices &= out[column] > 0.0
    if {"bid", "ask"}.issubset(out.columns):
        valid_prices &= pd.to_numeric(out["ask"], errors="coerce") >= pd.to_numeric(out["bid"], errors="coerce")
    return out.loc[valid_prices].copy()


def _select_entry_contract(
    snapshot: pd.DataFrame,
    *,
    entry_date: pd.Timestamp,
    option_type: Literal["call", "put"],
    target_strike: float,
    target_dte: int,
    entry_price_col: str,
    exit_price_col: str,
) -> dict[str, object] | None:
    if snapshot is None or snapshot.empty:
        return None
    target_type = str(option_type).lower()
    out = snapshot.loc[snapshot["option_type"].astype(str).str.lower().str.startswith(target_type[0])].copy()
    if out.empty:
        return None
    out["dte"] = (pd.to_datetime(out["expiration"], errors="coerce") - pd.Timestamp(entry_date)).dt.days
    quoteable = (
        (out["dte"] >= 0)
        & pd.to_numeric(out[entry_price_col], errors="coerce").gt(0.0)
        & pd.to_numeric(out[exit_price_col], errors="coerce").gt(0.0)
    )
    out = out.loc[quoteable].copy()
    if out.empty:
        return None
    out["dte_gap"] = (out["dte"] - int(target_dte)).abs()
    out["strike_gap"] = (pd.to_numeric(out["strike"], errors="coerce") - float(target_strike)).abs()
    out = out.sort_values(["dte_gap", "strike_gap", "expiration", "strike"], kind="mergesort")
    return out.iloc[0].to_dict()


def _real_quote_segment_prices(
    contract: dict[str, object],
    contract_quotes: pd.DataFrame,
    *,
    dates: pd.DatetimeIndex,
    spot_values: np.ndarray,
    option_type: Literal["call", "put"],
    entry_price_col: str,
    exit_price_col: str,
) -> np.ndarray:
    prices = np.full(len(dates), np.nan, dtype=float)
    if not contract_quotes.empty:
        quote_frame = (
            contract_quotes.assign(snapshot_date=pd.to_datetime(contract_quotes["snapshot_date"]).dt.normalize())
            .drop_duplicates(subset=["snapshot_date"], keep="last")
            .set_index("snapshot_date")
        )
        aligned = quote_frame.reindex(pd.DatetimeIndex(dates).normalize())
        prices = pd.to_numeric(aligned[exit_price_col], errors="coerce").to_numpy(dtype=float)
        if len(prices):
            entry_price = pd.to_numeric(aligned[entry_price_col], errors="coerce").iloc[0]
            prices[0] = float(entry_price) if np.isfinite(entry_price) else np.nan
    expiration = pd.Timestamp(contract["expiration"]).normalize()
    strike = float(contract["strike"])
    expired = pd.DatetimeIndex(dates).normalize() >= expiration
    if expired.any():
        if str(option_type).lower() == "put":
            intrinsic = np.maximum(strike - spot_values, 0.0)
        else:
            intrinsic = np.maximum(spot_values - strike, 0.0)
        missing_expired = expired & ~np.isfinite(prices)
        prices[missing_expired] = intrinsic[missing_expired]
    return prices


def _count_position_runs(mask: np.ndarray) -> int:
    if len(mask) == 0:
        return 0
    starts = mask & np.concatenate([[True], ~mask[:-1]])
    return int(starts.sum())


def _append_quote_coverage(
    rows: list[dict[str, object]],
    context: dict[str, object],
    *,
    symbol: str,
    selected_segments: int,
    fallback_segments: int,
    quote_return_count: int,
    fallback_return_count: int,
    avg_capacity_fill_fraction: float = np.nan,
) -> None:
    if selected_segments == 0 and fallback_segments == 0 and quote_return_count == 0 and fallback_return_count == 0:
        return
    rows.append(
        {
            **context,
            "symbol": str(symbol).upper(),
            "selected_segments": int(selected_segments),
            "fallback_segments": int(fallback_segments),
            "quote_return_count": int(quote_return_count),
            "fallback_return_count": int(fallback_return_count),
            "avg_capacity_fill_fraction": float(avg_capacity_fill_fraction),
        }
    )


def _build_maturing_option_return_panel(
    positions: pd.DataFrame,
    close_panel: pd.DataFrame,
    realized_vol_panel: pd.DataFrame,
    *,
    strike_multiplier: float,
    option_type: Literal["call", "put"],
    side: int,
    config: SyntheticOptionsBacktestConfig,
) -> pd.DataFrame:
    dates = pd.DatetimeIndex(positions.index)
    symbols = list(positions.columns)
    close = close_panel.reindex(index=dates, columns=symbols).astype(float)
    realized_vol = realized_vol_panel.reindex(index=dates, columns=symbols).astype(float)
    out = pd.DataFrame(0.0, index=dates, columns=symbols, dtype=float)
    tenor_days = max(int(config.tenor_days), 1)
    for symbol in symbols:
        signal = positions[symbol].reindex(dates).fillna(0).astype(int).to_numpy()
        desired = signal == int(side)
        spot_values = close[symbol].to_numpy(dtype=float)
        sigma_values = realized_vol[symbol].to_numpy(dtype=float) * float(config.iv_multiplier)
        returns = np.zeros(len(dates), dtype=float)
        cursor = 0
        while cursor < len(dates):
            if not desired[cursor] or not np.isfinite(spot_values[cursor]) or spot_values[cursor] <= 0.0:
                cursor += 1
                continue
            run_end = cursor
            while run_end + 1 < len(dates) and desired[run_end + 1]:
                run_end += 1
            segment_start = cursor
            while segment_start <= run_end:
                segment_end = min(run_end, segment_start + tenor_days)
                idx = np.arange(segment_start, segment_end + 1)
                strike = float(spot_values[segment_start]) * float(strike_multiplier)
                remaining_days = np.maximum(tenor_days - np.arange(len(idx)), 0)
                prices = _black_scholes_array(
                    spot=spot_values[idx],
                    strike=strike,
                    sigma=sigma_values[idx],
                    remaining_days=remaining_days,
                    rate=float(config.rate),
                    option_type=option_type,
                    premium_floor=float(config.premium_floor),
                )
                previous = prices[:-1]
                current = prices[1:]
                valid = np.isfinite(previous) & np.isfinite(current) & (previous > 0.0)
                segment_returns = np.zeros(len(current), dtype=float)
                segment_returns[valid] = (current[valid] / previous[valid]) - 1.0
                returns[idx[1:]] = segment_returns
                if segment_end >= run_end:
                    break
                segment_start = segment_end
            cursor = run_end + 1
        out[symbol] = returns
    return out


def _black_scholes_array(
    *,
    spot: np.ndarray,
    strike: float,
    sigma: np.ndarray,
    remaining_days: np.ndarray,
    rate: float,
    option_type: Literal["call", "put"],
    premium_floor: float,
) -> np.ndarray:
    spot = np.asarray(spot, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    remaining_days = np.asarray(remaining_days, dtype=float)
    strike = max(float(strike), 0.0)
    if strike <= 0.0:
        return np.full_like(spot, max(float(premium_floor), 0.0), dtype=float)
    if str(option_type).lower() == "put":
        intrinsic = np.maximum(strike - spot, 0.0)
    else:
        intrinsic = np.maximum(spot - strike, 0.0)
    out = intrinsic.astype(float)
    live = (spot > 0.0) & (remaining_days > 0.0)
    if not bool(live.any()):
        return np.maximum(out, 0.0)
    tau = np.maximum(remaining_days[live] / 252.0, 1.0 / 252.0)
    sigma_live = np.maximum(sigma[live], 1e-8)
    sqrt_tau = np.sqrt(tau)
    d1 = (
        np.log(spot[live] / strike)
        + (float(rate) + 0.5 * sigma_live * sigma_live) * tau
    ) / (sigma_live * sqrt_tau)
    d2 = d1 - sigma_live * sqrt_tau
    discount = np.exp(-float(rate) * tau)
    if str(option_type).lower() == "put":
        price = strike * discount * _norm_cdf_array(-d2) - spot[live] * _norm_cdf_array(-d1)
    else:
        price = spot[live] * _norm_cdf_array(d1) - strike * discount * _norm_cdf_array(d2)
    price = np.where(np.isfinite(price), price, intrinsic[live])
    out[live] = np.maximum(price, float(premium_floor))
    return np.maximum(out, 0.0)


def _norm_cdf_array(value: np.ndarray) -> np.ndarray:
    if _scipy_ndtr is not None:
        return _scipy_ndtr(value)
    return 0.5 * (1.0 + np.vectorize(math.erf)(value / np.sqrt(2.0)))


def _normalize_backtest_panel(panel: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(panel.index, pd.MultiIndex):
        raise TypeError("bt_panel must be indexed by date and symbol")
    out = panel.copy()
    if out.index.names[:2] != ["date", "symbol"]:
        out.index = out.index.set_names(["date", "symbol"])
    dates = pd.to_datetime(out.index.get_level_values("date"), errors="coerce").normalize()
    symbols = out.index.get_level_values("symbol").astype(str).str.upper()
    out.index = pd.MultiIndex.from_arrays([dates, symbols], names=["date", "symbol"])
    out = out.loc[~out.index.get_level_values("date").isna()]
    return out.replace([np.inf, -np.inf], np.nan).sort_index()


def _build_close_panel(panel: pd.DataFrame) -> pd.DataFrame:
    close_rows = panel[["close"]].reset_index()
    if close_rows.duplicated(subset=["date", "symbol"]).any():
        close_rows = (
            close_rows.sort_values(["date", "symbol"])
            .groupby(["date", "symbol"], as_index=False, sort=False)
            .last()
        )
    return (
        close_rows.pivot(index="date", columns="symbol", values="close")
        .sort_index()
        .replace([np.inf, -np.inf], np.nan)
        .ffill()
        .fillna(0.0)
    )


def _panel_years(panel: pd.DataFrame) -> list[int]:
    dates = pd.DatetimeIndex(panel.index.get_level_values("date"))
    return list(range(int(dates.min().year), int(dates.max().year) + 1))


def _position_counts(positions: pd.DataFrame, prior_positions: pd.DataFrame) -> dict[str, float | int]:
    return {
        "buy_count": int(((positions == 1) & (prior_positions != 1)).sum().sum()),
        "sell_count": int(((prior_positions == 1) & (positions != 1)).sum().sum()),
        "short_count": int(((positions == -1) & (prior_positions != -1)).sum().sum()),
        "cover_count": int(((prior_positions == -1) & (positions != -1)).sum().sum()),
        "avg_active_names": float(positions.ne(0).sum(axis=1).mean()) if len(positions) else np.nan,
        "avg_long_names": float((positions == 1).sum(axis=1).mean()) if len(positions) else np.nan,
        "avg_short_names": float((positions == -1).sum(axis=1).mean()) if len(positions) else np.nan,
    }
