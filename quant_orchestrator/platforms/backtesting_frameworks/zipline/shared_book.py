from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Iterable

import pandas as pd
from sqlalchemy import create_engine

from quant_orchestrator.platforms.backtesting_frameworks.reporting import build_common_summary
from quant_orchestrator.platforms.backtesting_frameworks.shared import OHLCV_COLUMNS, normalize_session_label


@dataclass(frozen=True)
class ZiplineSharedBookResult:
    perf: pd.DataFrame
    summary: pd.DataFrame
    equity_curve: pd.Series
    orders: pd.DataFrame


@dataclass(frozen=True)
class ZiplineSharedBookSummaryJob:
    price_frames: dict[str, pd.DataFrame]
    target_weights: pd.DataFrame
    metadata: dict[str, Any]
    capital_base: float = 1_000_000.0
    commission_per_share: float = 0.005
    slippage_bps: float = 5.0


@dataclass(frozen=True)
class _ZiplineSharedBookData:
    assets: dict[str, Any]
    asset_finder: Any
    data_portal: Any
    sim_params: Any
    benchmark_returns: pd.Series
    target_weights: dict[pd.Timestamp, dict[str, float]]
    calendar: Any


def run_zipline_shared_book(
    price_frames: dict[str, pd.DataFrame],
    target_weights: pd.DataFrame,
    *,
    capital_base: float = 1_000_000.0,
    commission_per_share: float = 0.005,
    slippage_bps: float = 5.0,
) -> ZiplineSharedBookResult:
    """Run a native Zipline multi-asset shared-book target-weight strategy."""

    started = perf_counter()
    adapter = _build_zipline_shared_book_data(
        price_frames,
        target_weights,
        capital_base=capital_base,
    )

    from zipline.algorithm import TradingAlgorithm
    from zipline.api import order_target_percent, record, set_commission, set_slippage
    from zipline.finance import commission, slippage

    def initialize(context, **kwargs):
        context.assets_by_symbol = adapter.assets
        set_commission(commission.PerShare(cost=float(commission_per_share), min_trade_cost=0.0))
        set_slippage(slippage.FixedBasisPointsSlippage(basis_points=float(slippage_bps), volume_limit=1.0))

    def handle_data(context, data):
        session = normalize_session_label(context.get_datetime())
        weights = adapter.target_weights.get(session, {})
        gross = 0.0
        net = 0.0
        for symbol, asset in context.assets_by_symbol.items():
            target = float(weights.get(symbol, 0.0))
            order_target_percent(asset, target)
            gross += abs(target)
            net += target
        record(gross_exposure_target=gross, net_exposure_target=net)

    algo = TradingAlgorithm(
        sim_params=adapter.sim_params,
        data_portal=adapter.data_portal,
        asset_finder=adapter.asset_finder,
        initialize=initialize,
        handle_data=handle_data,
        capital_base=capital_base,
        benchmark_returns=adapter.benchmark_returns,
    )
    perf = algo.run()
    equity = perf["portfolio_value"].rename("portfolio_value")
    summary = build_common_summary(
        framework="zipline_shared_book_native",
        symbol="PORTFOLIO",
        equity=equity,
        elapsed_seconds=perf_counter() - started,
        bars=len(equity),
        trades=_count_transactions(perf),
    )
    orders = _extract_transactions(perf)
    return ZiplineSharedBookResult(perf=perf, summary=summary, equity_curve=equity, orders=orders)


def run_zipline_shared_book_summary_jobs(
    jobs: Iterable[ZiplineSharedBookSummaryJob],
    *,
    max_workers: int | None = None,
) -> pd.DataFrame:
    """Run independent native Zipline shared-book jobs and return summary rows.

    The worker returns only the compact summary row, not the full Zipline perf
    frame. This keeps multiprocessing IPC small enough for notebook workflows.
    """

    job_list = list(jobs)
    if not job_list:
        return pd.DataFrame()
    if max_workers == 1:
        return pd.DataFrame([_run_zipline_shared_book_summary_job(job) for job in job_list])

    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_run_zipline_shared_book_summary_job, job) for job in job_list]
        for future in as_completed(futures):
            rows.append(future.result())
    return pd.DataFrame(rows)


def _run_zipline_shared_book_summary_job(job: ZiplineSharedBookSummaryJob) -> dict[str, Any]:
    result = run_zipline_shared_book(
        job.price_frames,
        job.target_weights,
        capital_base=job.capital_base,
        commission_per_share=job.commission_per_share,
        slippage_bps=job.slippage_bps,
    )
    row = result.summary.iloc[0].to_dict()
    row.update(job.metadata)
    return row


def _build_zipline_shared_book_data(
    price_frames: dict[str, pd.DataFrame],
    target_weights: pd.DataFrame,
    *,
    capital_base: float,
) -> _ZiplineSharedBookData:
    from zipline.assets import AssetDBWriter, AssetFinder
    from zipline.data.data_portal import DataPortal
    from zipline.data.in_memory_daily_bars import InMemoryDailyBarReader
    from zipline.finance.trading import SimulationParameters
    from zipline.utils.calendar_utils import get_calendar

    cleaned_prices = _normalize_price_frames(price_frames)
    if not cleaned_prices:
        raise ValueError("price_frames must contain at least one non-empty OHLCV frame")
    weights = _normalize_target_weights(target_weights)
    symbols = tuple(symbol for symbol in weights.columns if symbol in cleaned_prices)
    if not symbols:
        raise ValueError("target_weights columns must overlap price_frames symbols")
    weights = weights.loc[:, list(symbols)]

    calendar = get_calendar("XNYS")
    start = weights.index.min()
    end = weights.index.max()
    data_start = pd.Timestamp(start) - pd.Timedelta(days=10)
    data_sessions = calendar.sessions_in_range(data_start, end)
    sim_sessions = calendar.sessions_in_range(weights.index.min(), weights.index.max())
    if len(sim_sessions) == 0:
        raise ValueError("target_weights has no Zipline trading sessions")

    engine = create_engine("sqlite://")
    writer = AssetDBWriter(engine)
    writer.init_db()
    sids = {symbol: sid for sid, symbol in enumerate(symbols, start=1)}
    equities = pd.DataFrame(
        {
            "symbol": list(symbols),
            "asset_name": list(symbols),
            "start_date": [sim_sessions[0]] * len(symbols),
            "end_date": [sim_sessions[-1]] * len(symbols),
            "first_traded": [sim_sessions[0]] * len(symbols),
            "auto_close_date": [sim_sessions[-1]] * len(symbols),
            "exchange": ["TEST"] * len(symbols),
        },
        index=[sids[symbol] for symbol in symbols],
    )
    exchanges = pd.DataFrame(
        {"exchange": ["TEST"], "canonical_name": ["TEST"], "country_code": ["US"]},
        index=["TEST"],
    )
    mappings = pd.DataFrame(
        {
            "sid": [sids[symbol] for symbol in symbols],
            "symbol": list(symbols),
            "company_symbol": list(symbols),
            "share_class_symbol": [""] * len(symbols),
            "start_date": [pd.Timestamp(sim_sessions[0]).value] * len(symbols),
            "end_date": [pd.Timestamp(sim_sessions[-1]).value] * len(symbols),
            "country_code": ["US"] * len(symbols),
        }
    )
    writer.write_direct(equities=equities, equity_symbol_mappings=mappings, exchanges=exchanges)
    asset_finder = AssetFinder(engine)
    assets = {symbol: asset_finder.retrieve_asset(sid) for symbol, sid in sids.items()}

    aligned_prices = {
        symbol: cleaned_prices[symbol].reindex(data_sessions).ffill().bfill()
        for symbol in symbols
    }
    for frame in aligned_prices.values():
        frame["volume"] = frame["volume"].fillna(0.0)
    bar_frames = {
        column: pd.DataFrame(
            {assets[symbol]: aligned_prices[symbol][column].to_numpy() for symbol in symbols},
            index=data_sessions,
        )
        for column in OHLCV_COLUMNS
    }
    currency_codes = pd.Series({assets[symbol]: "USD" for symbol in symbols})
    reader = InMemoryDailyBarReader.from_dfs(bar_frames, calendar, currency_codes)
    reader.frames = bar_frames

    sim_sessions_naive = pd.DatetimeIndex(sim_sessions)
    if sim_sessions_naive.tz is not None:
        sim_sessions_naive = sim_sessions_naive.tz_convert(None)
    benchmark_returns = pd.Series(0.0, index=sim_sessions_naive)
    sim_params = SimulationParameters(
        start_session=sim_sessions_naive[0],
        end_session=sim_sessions_naive[-1],
        trading_calendar=calendar,
        capital_base=capital_base,
    )
    target_map = {
        normalize_session_label(date): {symbol: float(row[symbol]) for symbol in symbols}
        for date, row in weights.iterrows()
    }
    return _ZiplineSharedBookData(
        assets=assets,
        asset_finder=asset_finder,
        data_portal=DataPortal(
            asset_finder,
            calendar,
            sim_sessions_naive[0],
            equity_daily_reader=reader,
            last_available_session=sim_sessions_naive[-1],
        ),
        sim_params=sim_params,
        benchmark_returns=benchmark_returns,
        target_weights=target_map,
        calendar=calendar,
    )


def _normalize_price_frames(price_frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for symbol, frame in price_frames.items():
        if frame is None or frame.empty:
            continue
        normalized = frame.rename(columns=str.lower).copy()
        missing = set(OHLCV_COLUMNS) - set(normalized.columns)
        if missing:
            continue
        normalized.index = pd.DatetimeIndex(normalized.index)
        if normalized.index.tz is not None:
            normalized.index = normalized.index.tz_convert(None)
        normalized = normalized.loc[:, list(OHLCV_COLUMNS)].apply(pd.to_numeric, errors="coerce")
        normalized = normalized.dropna(subset=["open", "high", "low", "close"])
        if not normalized.empty:
            out[str(symbol).upper()] = normalized.sort_index()
    return out


def _normalize_target_weights(target_weights: pd.DataFrame) -> pd.DataFrame:
    weights = target_weights.copy()
    weights.index = pd.DatetimeIndex(weights.index)
    if weights.index.tz is not None:
        weights.index = weights.index.tz_convert(None)
    weights.columns = [str(column).upper() for column in weights.columns]
    weights = weights.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return weights.sort_index()


def _extract_transactions(perf: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if "transactions" not in perf.columns:
        return pd.DataFrame()
    for date, transactions in perf["transactions"].items():
        for transaction in transactions or []:
            row = dict(transaction)
            row["date"] = date
            rows.append(row)
    return pd.DataFrame(rows)


def _count_transactions(perf: pd.DataFrame) -> int:
    if "transactions" not in perf.columns:
        return 0
    return int(sum(len(transactions or []) for transactions in perf["transactions"]))
