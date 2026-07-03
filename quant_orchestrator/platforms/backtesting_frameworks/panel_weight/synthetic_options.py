from __future__ import annotations

import math

import numpy as np
import pandas as pd

try:
    from scipy.special import ndtr as _scipy_ndtr
except Exception:  # pragma: no cover - scipy is optional at import time
    _scipy_ndtr = None


def build_realized_vol_panel(
    close: pd.DataFrame,
    *,
    window: int = 21,
    vol_floor: float | None = 0.15,
    vol_cap: float | None = 0.80,
    annualization: float = 252.0,
) -> pd.DataFrame:
    close = close.astype(float)
    rolling_window = max(int(window), 1)
    min_periods = 2 if rolling_window > 1 else 1
    log_returns = np.log(close / close.shift(1)).replace([np.inf, -np.inf], np.nan)
    realized_vol = log_returns.rolling(rolling_window, min_periods=min_periods).std()
    realized_vol = realized_vol * math.sqrt(float(annualization))
    if vol_floor is not None:
        realized_vol = realized_vol.clip(lower=float(vol_floor)).fillna(float(vol_floor))
    if vol_cap is not None:
        realized_vol = realized_vol.clip(upper=float(vol_cap))
    return realized_vol


def build_constant_maturity_call_price_panel(
    close: pd.DataFrame,
    realized_vol: pd.DataFrame,
    *,
    strike_multiplier: float,
    tenor_days: int = 30,
    rate: float = 0.0,
    iv_multiplier: float = 1.0,
    premium_floor: float = 0.25,
) -> pd.DataFrame:
    return _black_scholes_panel(
        close,
        realized_vol,
        strike_multiplier=strike_multiplier,
        tenor_days=tenor_days,
        rate=rate,
        iv_multiplier=iv_multiplier,
        premium_floor=premium_floor,
        option_type="call",
    )


def build_constant_maturity_put_price_panel(
    close: pd.DataFrame,
    realized_vol: pd.DataFrame,
    *,
    strike_multiplier: float,
    tenor_days: int = 30,
    rate: float = 0.0,
    iv_multiplier: float = 1.0,
    premium_floor: float = 0.25,
) -> pd.DataFrame:
    return _black_scholes_panel(
        close,
        realized_vol,
        strike_multiplier=strike_multiplier,
        tenor_days=tenor_days,
        rate=rate,
        iv_multiplier=iv_multiplier,
        premium_floor=premium_floor,
        option_type="put",
    )


def build_synthetic_option_return_panels(
    close: pd.DataFrame,
    *,
    option_buckets: dict[str, dict[str, float]],
    realized_vol: pd.DataFrame | None = None,
    realized_vol_window: int = 21,
    vol_floor: float | None = 0.15,
    vol_cap: float | None = 0.80,
    tenor_days: int = 30,
    rate: float = 0.0,
    iv_multiplier: float = 1.0,
    premium_floor: float = 0.25,
) -> tuple[dict[str, dict[str, pd.DataFrame]], dict[str, dict[str, object]]]:
    close = close.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    if realized_vol is None:
        realized_vol = build_realized_vol_panel(
            close,
            window=realized_vol_window,
            vol_floor=vol_floor,
            vol_cap=vol_cap,
        )
    return_panels: dict[str, dict[str, pd.DataFrame]] = {
        "equity": {
            "long": close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).fillna(0.0),
            "short": close.pct_change(fill_method=None).mul(-1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0),
        },
    }
    price_panels: dict[str, dict[str, object]] = {}
    for name, bucket in option_buckets.items():
        long_strike_multiplier = float(bucket["long_strike_multiplier"])
        short_strike_multiplier = float(bucket["short_strike_multiplier"])
        call_prices = build_constant_maturity_call_price_panel(
            close,
            realized_vol,
            strike_multiplier=long_strike_multiplier,
            tenor_days=tenor_days,
            rate=rate,
            iv_multiplier=iv_multiplier,
            premium_floor=premium_floor,
        )
        put_prices = build_constant_maturity_put_price_panel(
            close,
            realized_vol,
            strike_multiplier=short_strike_multiplier,
            tenor_days=tenor_days,
            rate=rate,
            iv_multiplier=iv_multiplier,
            premium_floor=premium_floor,
        )
        price_panels[str(name)] = {
            "call": call_prices,
            "put": put_prices,
            "long_strike_multiplier": long_strike_multiplier,
            "short_strike_multiplier": short_strike_multiplier,
        }
        return_panels[str(name)] = {
            "long": call_prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).fillna(0.0),
            "short": put_prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).fillna(0.0),
        }
    return return_panels, price_panels


def _black_scholes_panel(
    close: pd.DataFrame,
    realized_vol: pd.DataFrame,
    *,
    strike_multiplier: float,
    tenor_days: int,
    rate: float,
    iv_multiplier: float,
    premium_floor: float,
    option_type: str,
) -> pd.DataFrame:
    tau = max(float(tenor_days) / 252.0, 1.0 / 252.0)
    multiplier = float(strike_multiplier)
    sqrt_tau = math.sqrt(tau)
    log_term = math.log(1.0 / multiplier)
    discount = math.exp(-float(rate) * tau)
    spot = close.astype(float)
    sigma = realized_vol.astype(float) * float(iv_multiplier)
    denom = (sigma * sqrt_tau).replace(0.0, np.nan)
    d1 = (log_term + (float(rate) + 0.5 * sigma * sigma) * tau) / denom
    d2 = d1 - sigma * sqrt_tau
    if str(option_type).lower() == "put":
        n1 = pd.DataFrame(_norm_cdf((-d1).to_numpy(dtype=float)), index=d1.index, columns=d1.columns)
        n2 = pd.DataFrame(_norm_cdf((-d2).to_numpy(dtype=float)), index=d2.index, columns=d2.columns)
        price = (spot * multiplier) * discount * n2 - spot * n1
        intrinsic = ((spot * multiplier) - spot).clip(lower=0.0)
    else:
        n1 = pd.DataFrame(_norm_cdf(d1.to_numpy(dtype=float)), index=d1.index, columns=d1.columns)
        n2 = pd.DataFrame(_norm_cdf(d2.to_numpy(dtype=float)), index=d2.index, columns=d2.columns)
        price = spot * n1 - (spot * multiplier) * discount * n2
        intrinsic = (spot - (spot * multiplier)).clip(lower=0.0)
    price = price.where(np.isfinite(price), intrinsic)
    return price.clip(lower=float(premium_floor))


def _norm_cdf(values: np.ndarray) -> np.ndarray:
    if _scipy_ndtr is not None:
        return _scipy_ndtr(values)
    return 0.5 * (1.0 + np.vectorize(math.erf)(values / np.sqrt(2.0)))
