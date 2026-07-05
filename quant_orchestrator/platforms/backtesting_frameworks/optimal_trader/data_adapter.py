from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import pandas as pd


OPTIMAL_TRADER_THETADATA_OPTION_COLUMNS: tuple[str, ...] = (
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


def load_strategy_dataset_artifact(path: str | Path) -> pd.DataFrame:
    """Load a saved optimal_trader strategy dataset for backtest replay."""

    source = Path(path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Missing strategy dataset artifact: {source}")
    suffix = source.suffix.lower()
    if suffix in {".pkl", ".pickle"}:
        frame = pd.read_pickle(source)
    elif suffix in {".parquet", ".pq"}:
        frame = pd.read_parquet(source)
    else:
        frame = pd.read_csv(source)
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"Strategy dataset artifact did not load as a DataFrame: {source}")
    return normalize_strategy_dataset_frame(frame)


def normalize_strategy_dataset_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    out = frame.copy()
    out = out.rename(columns={column: _canonical_column_name(column) for column in out.columns})
    required = {"date", "symbol", "target_weight"}
    missing = sorted(required.difference(out.columns))
    if missing:
        raise ValueError(f"Strategy dataset artifact missing required columns: {missing}")
    if "ret_1" not in out.columns and "asset_return" not in out.columns:
        raise ValueError("Strategy dataset artifact requires 'ret_1' or 'asset_return'")
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["symbol"] = out["symbol"].astype(str).str.strip().str.upper()
    out = out.loc[out["date"].notna() & out["symbol"].ne("")].copy()
    return out.reset_index(drop=True)


def _canonical_column_name(column: object) -> str:
    text = str(column).strip()
    mapping = {
        "Date": "date",
        "Symbol": "symbol",
        "Target Weight": "target_weight",
        "TargetWeight": "target_weight",
        "Asset Return": "asset_return",
        "AssetReturn": "asset_return",
        "Return": "ret_1",
    }
    return mapping.get(text, text)


def read_optimal_trader_thetadata_option_chain(
    symbol: str,
    *,
    start_date: Any = None,
    end_date: Any = None,
    columns: Sequence[str] | None = None,
    require_rich_columns: bool = False,
    fallback_legacy: bool = False,
) -> pd.DataFrame:
    """Read ThetaData EOD options for optimal_trader option-equivalent backtests."""

    from quant_warehouse.platforms.data_providers.thetadata.options import read_thetadata_eod_option_chain

    requested_columns = tuple(columns) if columns is not None else OPTIMAL_TRADER_THETADATA_OPTION_COLUMNS
    return read_thetadata_eod_option_chain(
        str(symbol).upper(),
        start_date=start_date,
        end_date=end_date,
        columns=requested_columns,
        require_rich_columns=bool(require_rich_columns),
        fallback_legacy=bool(fallback_legacy),
    )


def optimal_trader_thetadata_option_chain_coverage(symbols: Sequence[str] | None = None) -> pd.DataFrame:
    from quant_warehouse.platforms.data_providers.thetadata.options import option_chain_coverage

    return option_chain_coverage(symbols)


def build_optimal_trader_thetadata_option_contract_features(*args: Any, **kwargs: Any) -> Any:
    from quant_warehouse.platforms.data_providers.thetadata.feature_engineering import build_option_contract_features

    return build_option_contract_features(*args, **kwargs)


def optimal_trader_thetadata_option_ranker_feature_columns(*args: Any, **kwargs: Any) -> Any:
    from quant_warehouse.platforms.data_providers.thetadata.feature_engineering import option_ranker_feature_columns

    return option_ranker_feature_columns(*args, **kwargs)


def build_optimal_trader_thetadata_option_mean_variance_labels(*args: Any, **kwargs: Any) -> Any:
    from quant_warehouse.platforms.data_providers.thetadata.target_engineering import build_option_mean_variance_labels

    return build_option_mean_variance_labels(*args, **kwargs)


def settle_optimal_trader_thetadata_option_exit(*args: Any, **kwargs: Any) -> Any:
    from quant_warehouse.platforms.data_providers.thetadata.settlement import settle_option_exit

    return settle_option_exit(*args, **kwargs)
