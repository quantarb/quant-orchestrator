from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sqlalchemy import create_engine

from quant_orchestrator.platforms.backtesting_frameworks.shared import (
    OHLCV_COLUMNS,
    normalize_session_label,
)


@dataclass(frozen=True)
class ZiplineInMemoryData:
    asset: Any
    asset_finder: Any
    data_portal: Any
    sim_params: Any
    benchmark_returns: pd.Series
    trade_size: int
    signal_map: dict[pd.Timestamp, bool]
    calendar: Any


def build_zipline_in_memory_data(
    frame: pd.DataFrame,
    *,
    symbol: str,
    capital_base: float,
) -> ZiplineInMemoryData:
    from zipline.assets import AssetDBWriter, AssetFinder
    from zipline.data.data_portal import DataPortal
    from zipline.data.in_memory_daily_bars import InMemoryDailyBarReader
    from zipline.finance.trading import SimulationParameters
    from zipline.utils.calendar_utils import get_calendar

    engine = create_engine("sqlite://")
    writer = AssetDBWriter(engine)
    writer.init_db()

    naive_index = pd.DatetimeIndex(frame.index)
    if naive_index.tz is not None:
        naive_index = naive_index.tz_convert(None)
    calendar = get_calendar("XNYS")
    sessions = calendar.sessions_in_range(naive_index.min(), naive_index.max())

    equities = pd.DataFrame(
        {
            "symbol": [symbol.upper()],
            "asset_name": [symbol.upper()],
            "start_date": [sessions[0]],
            "end_date": [sessions[-1]],
            "first_traded": [sessions[0]],
            "auto_close_date": [sessions[-1]],
            "exchange": ["TEST"],
        },
        index=[1],
    )
    exchanges = pd.DataFrame(
        {"exchange": ["TEST"], "canonical_name": ["TEST"], "country_code": ["US"]},
    )
    symbol_mappings = pd.DataFrame(
        {
            "sid": [1],
            "symbol": [symbol.upper()],
            "company_symbol": [symbol.upper()],
            "share_class_symbol": [""],
            "start_date": [sessions[0]],
            "end_date": [sessions[-1]],
        },
    )
    writer.write_direct(
        equities=equities,
        equity_symbol_mappings=symbol_mappings,
        exchanges=exchanges,
    )
    asset_finder = AssetFinder(engine)
    asset = asset_finder.retrieve_asset(1)

    sessions_naive = pd.DatetimeIndex(sessions)
    if sessions_naive.tz is not None:
        sessions_naive = sessions_naive.tz_convert(None)
    aligned = frame.reindex(sessions).ffill()
    aligned["volume"] = aligned["volume"].fillna(0.0)
    currency_codes = pd.Series({asset: "USD"})
    bar_frames = {
        column: pd.DataFrame({asset: aligned[column].to_numpy()}, index=sessions)
        for column in OHLCV_COLUMNS
    }
    reader = InMemoryDailyBarReader.from_dfs(bar_frames, calendar, currency_codes)
    reader.frames = bar_frames

    benchmark_returns = pd.Series(0.0, index=sessions_naive)
    sim_params = SimulationParameters(
        start_session=sessions_naive[0],
        end_session=sessions_naive[-1],
        trading_calendar=calendar,
        capital_base=capital_base,
    )
    signal_map = {
        normalize_session_label(date): bool(signal)
        for date, signal in frame["signal"].items()
    }
    return ZiplineInMemoryData(
        asset=asset,
        asset_finder=asset_finder,
        data_portal=DataPortal(
            asset_finder,
            calendar,
            sessions_naive[0],
            equity_daily_reader=reader,
            last_available_session=sessions_naive[-1],
        ),
        sim_params=sim_params,
        benchmark_returns=benchmark_returns,
        trade_size=max(1, int((capital_base * 0.25) / float(frame["close"].iloc[0]))),
        signal_map=signal_map,
        calendar=calendar,
    )
