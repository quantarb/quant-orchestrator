from __future__ import annotations

from typing import Any, Sequence

import pandas as pd


PANEL_WEIGHT_THETADATA_OPTION_COLUMNS: tuple[str, ...] = (
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


def read_panel_weight_thetadata_option_chain(
    symbol: str,
    *,
    start_date: Any = None,
    end_date: Any = None,
    columns: Sequence[str] | None = None,
    require_rich_columns: bool = False,
    fallback_legacy: bool = False,
) -> pd.DataFrame:
    """Read ThetaData EOD options in the shape expected by panel-weight tests."""

    from quant_warehouse.platforms.data_providers.thetadata.options import read_thetadata_eod_option_chain

    requested_columns = tuple(columns) if columns is not None else PANEL_WEIGHT_THETADATA_OPTION_COLUMNS
    return read_thetadata_eod_option_chain(
        str(symbol).upper(),
        start_date=start_date,
        end_date=end_date,
        columns=requested_columns,
        require_rich_columns=bool(require_rich_columns),
        fallback_legacy=bool(fallback_legacy),
    )


def panel_weight_thetadata_option_chain_coverage(symbols: Sequence[str] | None = None) -> pd.DataFrame:
    from quant_warehouse.platforms.data_providers.thetadata.options import option_chain_coverage

    return option_chain_coverage(symbols)


def build_panel_weight_thetadata_option_contract_features(*args: Any, **kwargs: Any) -> Any:
    from quant_warehouse.platforms.data_providers.thetadata.feature_engineering import build_option_contract_features

    return build_option_contract_features(*args, **kwargs)


def panel_weight_thetadata_option_ranker_feature_columns(*args: Any, **kwargs: Any) -> Any:
    from quant_warehouse.platforms.data_providers.thetadata.feature_engineering import option_ranker_feature_columns

    return option_ranker_feature_columns(*args, **kwargs)


def build_panel_weight_thetadata_option_mean_variance_labels(*args: Any, **kwargs: Any) -> Any:
    from quant_warehouse.platforms.data_providers.thetadata.target_engineering import build_option_mean_variance_labels

    return build_option_mean_variance_labels(*args, **kwargs)


def settle_panel_weight_thetadata_option_exit(*args: Any, **kwargs: Any) -> Any:
    from quant_warehouse.platforms.data_providers.thetadata.settlement import settle_option_exit

    return settle_option_exit(*args, **kwargs)
