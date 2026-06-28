from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import vectorbt as vbt

from quant_orchestrator.experiments import WindowSpec, build_walk_forward_windows
from quant_orchestrator.monte_carlo import simulate_return_paths
from quant_orchestrator.platforms.backtesting_frameworks.backtesting_py.data_adapter import (
    build_backtesting_frame,
)
from quant_orchestrator.platforms.backtesting_frameworks.nautilus.data_adapter import (
    build_nautilus_in_memory_data,
)
from quant_orchestrator.platforms.backtesting_frameworks.shared import (
    MAG7_SYMBOLS,
    load_price_frame,
    normalize_session_label,
)
from quant_orchestrator.platforms.backtesting_frameworks.zipline.data_adapter import (
    build_zipline_in_memory_data,
)
from quant_orchestrator.strategy import summarize_backtest, summarize_equity


FAST_WINDOWS = (5, 10, 15, 20, 25, 30, 35, 40, 45, 50)
SLOW_WINDOWS = (60, 70, 80, 90, 100, 120, 140, 160, 180, 200)
SYMBOL_COUNT = len(MAG7_SYMBOLS)
WINDOW_GRID = tuple(product(FAST_WINDOWS, SLOW_WINDOWS))
WINDOW_COLUMNS = pd.MultiIndex.from_tuples(WINDOW_GRID, names=["fast_window", "slow_window"])


@dataclass(frozen=True)
class FrameworkRun:
    raw_result: object
    summary: pd.DataFrame
    equity: pd.Series


def _patch_zipline_compatibility() -> None:
    if getattr(_patch_zipline_compatibility, "_patched", False):
        return

    import zipline.finance.ledger as ledger_mod
    from zipline.data.in_memory_daily_bars import InMemoryDailyBarReader

    InMemoryDailyBarReader.frames = property(lambda self: self._frames)
    ledger_mod.PositionTracker.stats = property(lambda self: self._stats)
    _patch_zipline_compatibility._patched = True  # type: ignore[attr-defined]


def build_sma_frame(prices: pd.DataFrame, *, fast_window: int, slow_window: int) -> pd.DataFrame:
    if fast_window >= slow_window:
        raise ValueError("fast_window must be less than slow_window")

    frame = prices.copy()
    fast = frame["close"].rolling(fast_window).mean()
    slow = frame["close"].rolling(slow_window).mean()
    frame["fast_sma"] = fast
    frame["slow_sma"] = slow
    frame["signal"] = (fast > slow).astype(int).fillna(0)
    frame = frame.dropna(subset=["open", "high", "low", "close", "volume"]).copy()
    return frame


def _combine_equity_sleeves(sleeves: list[pd.Series]) -> pd.Series:
    combined_index = pd.Index([])
    for sleeve in sleeves:
        combined_index = combined_index.union(sleeve.index)
    combined = pd.Series(0.0, index=combined_index)
    for sleeve in sleeves:
        reindexed = sleeve.reindex(combined_index)
        reindexed = reindexed.ffill().fillna(sleeve.iloc[0])
        combined = combined.add(reindexed, fill_value=0.0)
    return combined.sort_index()


def _load_symbol_price_frames(
    *,
    provider: str,
    start: str,
    end: str | None,
    symbols: tuple[str, ...] = MAG7_SYMBOLS,
) -> dict[str, pd.DataFrame]:
    frames = {}
    for symbol in symbols:
        frames[symbol] = load_price_frame(symbol, provider=provider, start=start, end=end)
    return frames


def run_backtesting_py(frame: pd.DataFrame, *, symbol: str, capital_base: float) -> FrameworkRun:
    from backtesting import Backtest, Strategy

    trade_size = max(1, int((capital_base * 0.25) / float(frame["close"].iloc[0])))
    bt_frame = build_backtesting_frame(frame).drop(columns=["Signal"])
    signal_map = {
        normalize_session_label(date): bool(signal)
        for date, signal in frame["signal"].items()
    }

    class SignalStrategy(Strategy):
        def init(self):
            self.trade_size = trade_size

        def next(self):
            bullish = signal_map.get(normalize_session_label(self.data.index[-1]), False)
            if bullish and not self.position:
                self.buy(size=self.trade_size)
            elif not bullish and self.position:
                self.position.close()

    started = perf_counter()
    stats = Backtest(
        bt_frame,
        SignalStrategy,
        cash=capital_base,
        commission=0.0,
        trade_on_close=False,
        exclusive_orders=True,
    ).run()
    elapsed = perf_counter() - started
    equity = stats["_equity_curve"]["Equity"].rename("portfolio_value")
    summary = summarize_backtest(
        framework="backtesting.py",
        symbol=symbol,
        equity=equity,
        elapsed_seconds=elapsed,
        bars=len(bt_frame),
        trades=int(stats["# Trades"]),
    )
    summary["native_return_pct"] = float(stats["Return [%]"])
    summary["native_sharpe"] = float(stats["Sharpe Ratio"]) if pd.notna(stats["Sharpe Ratio"]) else None
    summary["native_max_drawdown_pct"] = float(stats["Max. Drawdown [%]"])
    return FrameworkRun(stats, summary, equity)


def run_zipline(frame: pd.DataFrame, *, symbol: str, capital_base: float) -> FrameworkRun:
    from zipline.algorithm import TradingAlgorithm
    from zipline.api import order_target, record, symbol as zipline_symbol

    _patch_zipline_compatibility()
    adapter = build_zipline_in_memory_data(frame, symbol=symbol, capital_base=capital_base)
    trade_size = adapter.trade_size

    def initialize(context, **kwargs):
        context.asset = zipline_symbol(symbol.upper())
        context.is_long = False

    def handle_data(context, data):
        dt = normalize_session_label(context.get_datetime())
        bullish = adapter.signal_map.get(dt, False)
        if bullish and not context.is_long:
            order_target(context.asset, trade_size)
            context.is_long = True
        elif not bullish and context.is_long:
            order_target(context.asset, 0)
            context.is_long = False
        record(price=data.current(context.asset, "price"), signal=float(bullish))

    started = perf_counter()
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
    elapsed = perf_counter() - started
    equity = perf["portfolio_value"].rename("portfolio_value")
    transactions = perf.get("transactions", pd.Series(index=perf.index, data=[[]] * len(perf)))
    summary = summarize_backtest(
        framework="zipline",
        symbol=symbol,
        equity=equity,
        elapsed_seconds=elapsed,
        bars=len(perf),
        trades=int(transactions.map(len).sum()),
    )
    summary["native_last_value"] = float(perf["portfolio_value"].iloc[-1])
    return FrameworkRun(perf, summary, equity)


def run_nautilus(frame: pd.DataFrame, *, symbol: str, capital_base: float) -> FrameworkRun:
    from decimal import Decimal

    from nautilus_trader.backtest.engine import BacktestEngine
    from nautilus_trader.config import BacktestEngineConfig, LoggingConfig, StrategyConfig
    from nautilus_trader.model.currencies import USD
    from nautilus_trader.model.enums import AccountType, OmsType, OrderSide, TimeInForce
    from nautilus_trader.model.objects import Money, Quantity
    from nautilus_trader.trading.strategy import Strategy

    adapter = build_nautilus_in_memory_data(frame, symbol=symbol, capital_base=capital_base)

    class SignalStrategyConfig(StrategyConfig, frozen=True):
        instrument_id: object
        bar_type: object
        trade_size: Decimal
        signal_map: dict

    class SignalStrategy(Strategy):
        def __init__(self, config: SignalStrategyConfig):
            super().__init__(config)
            self.is_long = False

        def on_start(self) -> None:
            self.subscribe_bars(self.config.bar_type)

        def on_bar(self, bar) -> None:
            bullish = self.config.signal_map.get(normalize_session_label(bar.ts_event), False)
            if bullish == self.is_long:
                return
            side = OrderSide.BUY if bullish else OrderSide.SELL
            order = self.order_factory.market(
                instrument_id=self.config.instrument_id,
                order_side=side,
                quantity=Quantity.from_int(int(self.config.trade_size)),
                time_in_force=TimeInForce.IOC,
            )
            self.submit_order(order)
            self.is_long = bullish

    engine = BacktestEngine(
        config=BacktestEngineConfig(
            logging=LoggingConfig(log_level="OFF", bypass_logging=True),
        ),
    )
    engine.add_venue(
        venue=adapter.venue,
        oms_type=OmsType.NETTING,
        account_type=AccountType.CASH,
        starting_balances=[Money(capital_base, USD)],
        base_currency=USD,
    )
    engine.add_instrument(adapter.instrument)
    engine.add_data(adapter.bars)
    engine.add_strategy(
        SignalStrategy(
            SignalStrategyConfig(
                instrument_id=adapter.instrument.id,
                bar_type=adapter.bar_type,
                trade_size=Decimal(adapter.trade_size),
                signal_map=adapter.signal_map,
            ),
        ),
    )

    started = perf_counter()
    engine.run()
    fills_report = engine.trader.generate_order_fills_report()
    elapsed = perf_counter() - started
    engine.dispose()

    equity = _equity_from_fills(prices=frame, fills=fills_report, capital_base=capital_base)
    summary = summarize_backtest(
        framework="nautilus",
        symbol=symbol,
        equity=equity,
        elapsed_seconds=elapsed,
        bars=len(frame),
        trades=len(fills_report),
    )
    summary["native_fills"] = int(len(fills_report))
    summary["native_last_value"] = float(equity.iloc[-1])
    return FrameworkRun(fills_report, summary, equity)


def _equity_from_fills(*, prices: pd.DataFrame, fills: pd.DataFrame, capital_base: float) -> pd.Series:
    cash = float(capital_base)
    position = 0.0
    values = []
    fills_by_date: dict[pd.Timestamp, list[pd.Series]] = {}

    for _, fill in fills.iterrows():
        fill_date = normalize_session_label(fill["ts_last"])
        fills_by_date.setdefault(fill_date, []).append(fill)

    for date, row in prices.iterrows():
        normalized = normalize_session_label(date)
        for fill in fills_by_date.get(normalized, []):
            quantity = float(fill["filled_qty"])
            price = float(fill["avg_px"])
            if str(fill["side"]) == "BUY":
                cash -= quantity * price
                position += quantity
            else:
                cash += quantity * price
                position -= quantity
        values.append(cash + position * float(row["close"]))

    return pd.Series(values, index=prices.index, name="portfolio_value")


def build_combo_signal_matrices(close: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    fast_ma = vbt.MA.run(close, window=list(FAST_WINDOWS)).ma
    slow_ma = vbt.MA.run(close, window=list(SLOW_WINDOWS)).ma
    fast_values = fast_ma.to_numpy()
    slow_values = slow_ma.to_numpy()
    bullish = fast_values[:, :, None] > slow_values[:, None, :]
    prev_bullish = np.zeros_like(bullish, dtype=bool)
    prev_bullish[1:] = bullish[:-1]
    entries = bullish & ~prev_bullish
    exits = (~bullish) & prev_bullish
    entries_frame = pd.DataFrame(entries.reshape(len(close), -1), index=close.index, columns=WINDOW_COLUMNS)
    exits_frame = pd.DataFrame(exits.reshape(len(close), -1), index=close.index, columns=WINDOW_COLUMNS)
    return entries_frame, exits_frame


def run_vectorbt_portfolio_search(*, provider: str, start: str, end: str | None, capital_base: float) -> tuple[pd.DataFrame, pd.Series, float]:
    started = perf_counter()
    capital_per_symbol = capital_base / SYMBOL_COUNT
    combined_values: pd.DataFrame | None = None
    combined_trades: pd.Series | None = None

    for symbol in MAG7_SYMBOLS:
        prices = load_price_frame(symbol, provider=provider, start=start, end=end)
        entries, exits = build_combo_signal_matrices(prices["close"])
        pf = vbt.Portfolio.from_signals(
            prices["close"],
            entries=entries,
            exits=exits,
            init_cash=capital_per_symbol,
            freq="1D",
            upon_opposite_entry="close",
        )
        values = pf.value()
        trades = pf.trades.count()
        if combined_values is None:
            combined_values = values.copy()
            combined_trades = trades.copy()
        else:
            combined_index = combined_values.index.union(values.index)
            combined_values = combined_values.reindex(combined_index).ffill().bfill()
            values = values.reindex(combined_index).ffill().bfill()
            combined_values = combined_values.add(values, fill_value=0.0)
            combined_trades = combined_trades.add(trades, fill_value=0.0)

    elapsed = perf_counter() - started
    if combined_values is None or combined_trades is None:
        raise ValueError("vectorbt search produced no output")
    return combined_values.sort_index(), combined_trades.sort_index(), elapsed


def build_search_grid(values: pd.DataFrame, trades: pd.Series, *, provider: str, elapsed_seconds: float) -> pd.DataFrame:
    rows = []
    for combo in values.columns:
        equity = values.loc[:, combo].rename("portfolio_value")
        summary = summarize_equity(equity)
        rows.append(
            {
                "provider": provider,
                "fast_window": int(combo[0]),
                "slow_window": int(combo[1]),
                "bars": len(equity),
                "trades": int(round(float(trades.loc[combo]))),
                "final_equity": summary["final_equity"],
                "total_return": summary["total_return"],
                "max_drawdown": summary["max_drawdown"],
                "daily_vol": summary["daily_vol"],
                "elapsed_seconds": round(elapsed_seconds, 4),
            }
        )
    grid = pd.DataFrame(rows).sort_values(
        by=["total_return", "max_drawdown", "final_equity"],
        ascending=[False, True, False],
    ).reset_index(drop=True)
    grid["train_rank"] = grid.index + 1
    return grid


def run_framework_portfolio(
    framework: str,
    *,
    symbol_frames: dict[str, pd.DataFrame],
    capital_base: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, FrameworkRun]]:
    capital_per_symbol = capital_base / max(1, len(symbol_frames))
    runs: dict[str, FrameworkRun] = {}
    summaries = []
    sleeves = []
    elapsed = 0.0
    trades = 0

    for symbol, frame in symbol_frames.items():
        started = perf_counter()
        run = {
            "backtesting.py": run_backtesting_py,
            "zipline": run_zipline,
            "nautilus": run_nautilus,
        }[framework](frame, symbol=symbol, capital_base=capital_per_symbol)
        elapsed += perf_counter() - started
        summaries.append(run.summary)
        sleeves.append(run.equity.rename(symbol))
        trades += int(run.summary["trades"].iloc[0])
        runs[symbol] = run

    summary = pd.concat(summaries, ignore_index=True)
    portfolio_equity = _combine_equity_sleeves(sleeves)
    portfolio_summary = summarize_backtest(
        framework=framework,
        symbol="PORTFOLIO",
        equity=portfolio_equity,
        elapsed_seconds=elapsed,
        bars=len(portfolio_equity),
        trades=trades,
    )
    return summary, portfolio_summary, runs


def build_symbol_frames(
    *,
    provider: str,
    fast_window: int,
    slow_window: int,
    start: str,
    end: str | None,
    symbols: tuple[str, ...] = MAG7_SYMBOLS,
) -> dict[str, pd.DataFrame]:
    frames = {}
    for symbol in symbols:
        prices = load_price_frame(symbol, provider=provider, start=start, end=end)
        frames[symbol] = build_sma_frame(prices, fast_window=fast_window, slow_window=slow_window)
    return frames


def run_multi_vendor_backtesting_py_sma_comparison(
    *,
    providers: tuple[str, ...] = ("fmp", "yfinance"),
    symbols: tuple[str, ...] = MAG7_SYMBOLS,
    start: str = "2020-01-01",
    end: str | None = None,
    fast_window: int = 50,
    slow_window: int = 200,
    capital_base: float = 100_000.0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, FrameworkRun]]]:
    symbol_tables = []
    portfolio_tables = []
    runs_by_provider: dict[str, dict[str, FrameworkRun]] = {}

    for provider in providers:
        capital_per_symbol = capital_base / max(1, len(symbols))
        provider_runs: dict[str, FrameworkRun] = {}
        provider_rows = []
        provider_sleeves = []
        elapsed = 0.0
        trades = 0

        for symbol in symbols:
            prices = load_price_frame(symbol, provider=provider, start=start, end=end)
            frame = build_sma_frame(prices, fast_window=fast_window, slow_window=slow_window)
            started = perf_counter()
            run = run_backtesting_py(frame, symbol=symbol, capital_base=capital_per_symbol)
            elapsed += perf_counter() - started
            provider_rows.append(run.summary.assign(provider=provider))
            provider_sleeves.append(run.equity.rename(symbol))
            trades += int(run.summary["trades"].iloc[0])
            provider_runs[symbol] = run

        symbol_summary = pd.concat(provider_rows, ignore_index=True)
        portfolio_equity = _combine_equity_sleeves(provider_sleeves)
        portfolio_summary = summarize_backtest(
            framework="backtesting.py",
            symbol="MAG7",
            equity=portfolio_equity,
            elapsed_seconds=elapsed,
            bars=len(portfolio_equity),
            trades=trades,
        )
        portfolio_summary["provider"] = provider
        portfolio_summary["fast_window"] = fast_window
        portfolio_summary["slow_window"] = slow_window

        symbol_tables.append(symbol_summary)
        portfolio_tables.append(portfolio_summary)
        runs_by_provider[provider] = provider_runs

    return (
        pd.concat(symbol_tables, ignore_index=True),
        pd.concat(portfolio_tables, ignore_index=True),
        runs_by_provider,
    )


def run_cross_framework_sma_search_monte_carlo(
    *,
    providers: tuple[str, ...] = ("fmp", "yfinance"),
    forward_symbols: tuple[str, ...] = ("QQQ", "SPY"),
    capital_base: float = 100_000.0,
    top_mc_count: int = 10,
    mc_iterations: int = 1_000,
    mc_block_size: int = 5,
    start: str = "2020-01-01",
    train_end: str = "2025-12-31",
    test_start: str = "2026-01-01",
    test_end: str = "2026-12-31",
) -> dict[str, Any]:
    train_values_by_provider = {}
    train_grid_by_provider = {}
    search_elapsed_rows = []
    search_tables = []

    for provider in providers:
        provider_started = perf_counter()
        train_values, train_trades, vectorbt_elapsed = run_vectorbt_portfolio_search(
            provider=provider,
            start=start,
            end=train_end,
            capital_base=capital_base,
        )
        train_grid = build_search_grid(train_values, train_trades, provider=provider, elapsed_seconds=vectorbt_elapsed)
        train_values_by_provider[provider] = train_values
        train_grid_by_provider[provider] = train_grid
        search_elapsed_rows.append(
            {
                "provider": provider,
                "vectorized_search_seconds": round(vectorbt_elapsed, 4),
                "wall_seconds": round(perf_counter() - provider_started, 4),
            }
        )
        search_tables.append(train_grid.head(top_mc_count))

    train_search_report = pd.concat(search_tables, ignore_index=True)
    search_elapsed = pd.DataFrame(search_elapsed_rows)

    mc_rows = []
    mc_elapsed_rows = []
    for provider in providers:
        provider_grid = train_grid_by_provider[provider]
        provider_values = train_values_by_provider[provider]
        top10 = provider_grid.head(top_mc_count).copy()

        mc_started = perf_counter()
        for _, row in top10.iterrows():
            fast_window = int(row["fast_window"])
            slow_window = int(row["slow_window"])
            key = (fast_window, slow_window)
            equity = provider_values.loc[:, key].rename("portfolio_value")
            returns = equity.pct_change().dropna()
            mc = simulate_return_paths(
                returns,
                iterations=mc_iterations,
                block_size=mc_block_size,
                horizon=len(returns),
            )
            mc_rows.append(
                {
                    "provider": provider,
                    "fast_window": fast_window,
                    "slow_window": slow_window,
                    "train_total_return": float(row["total_return"]),
                    "train_max_drawdown": float(row["max_drawdown"]),
                    "terminal_return_mean": float(mc.summary.loc[0, "terminal_return_mean"]),
                    "terminal_return_p05": float(mc.summary.loc[0, "terminal_return_p05"]),
                    "terminal_return_p50": float(mc.summary.loc[0, "terminal_return_p50"]),
                    "terminal_return_p95": float(mc.summary.loc[0, "terminal_return_p95"]),
                    "max_drawdown_mean": float(mc.summary.loc[0, "max_drawdown_mean"]),
                    "max_drawdown_p05": float(mc.summary.loc[0, "max_drawdown_p05"]),
                }
            )
        mc_elapsed_rows.append(
            {
                "provider": provider,
                "mc_seconds": round(perf_counter() - mc_started, 4),
            }
        )

    mc_report = pd.DataFrame(mc_rows)
    mc_report["mc_robustness_score"] = mc_report["terminal_return_p50"] + mc_report["max_drawdown_mean"]
    mc_report = mc_report.sort_values(
        by=["provider", "mc_robustness_score", "terminal_return_p50"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    mc_report["mc_rank"] = mc_report.groupby("provider").cumcount() + 1
    mc_elapsed = pd.DataFrame(mc_elapsed_rows)

    selection_rows = []
    for provider in providers:
        provider_grid = train_grid_by_provider[provider]
        provider_mc = mc_report[mc_report["provider"] == provider].copy()
        train_best = provider_grid.iloc[0]
        mc_best = provider_mc.iloc[0]
        merged = provider_grid.loc[:, ["fast_window", "slow_window", "total_return", "train_rank"]].merge(
            provider_mc.loc[:, ["fast_window", "slow_window", "mc_robustness_score", "mc_rank"]],
            on=["fast_window", "slow_window"],
            how="inner",
        )
        merged["hybrid_rank"] = merged["train_rank"] + merged["mc_rank"]
        merged = merged.sort_values(
            by=["hybrid_rank", "mc_robustness_score", "total_return"],
            ascending=[True, False, False],
        ).reset_index(drop=True)
        hybrid_best = merged.iloc[0]

        selection_rows.extend(
            [
                {
                    "provider": provider,
                    "selection": "best_train",
                    "fast_window": int(train_best["fast_window"]),
                    "slow_window": int(train_best["slow_window"]),
                    "train_total_return": float(train_best["total_return"]),
                    "mc_robustness_score": float(
                        provider_mc.loc[
                            (provider_mc["fast_window"] == int(train_best["fast_window"]))
                            & (provider_mc["slow_window"] == int(train_best["slow_window"])),
                            "mc_robustness_score",
                        ].iloc[0]
                    ),
                },
                {
                    "provider": provider,
                    "selection": "best_mc",
                    "fast_window": int(mc_best["fast_window"]),
                    "slow_window": int(mc_best["slow_window"]),
                    "train_total_return": float(mc_best["train_total_return"]),
                    "mc_robustness_score": float(mc_best["mc_robustness_score"]),
                },
                {
                    "provider": provider,
                    "selection": "best_hybrid",
                    "fast_window": int(hybrid_best["fast_window"]),
                    "slow_window": int(hybrid_best["slow_window"]),
                    "train_total_return": float(hybrid_best["total_return"]),
                    "mc_robustness_score": float(hybrid_best["mc_robustness_score"]),
                },
            ]
        )

    selections = pd.DataFrame(selection_rows)

    forward_rows = []
    forward_runs = {}
    for provider in providers:
        provider_selections = selections[selections["provider"] == provider]
        for selection_name, fast_window, slow_window in provider_selections[["selection", "fast_window", "slow_window"]].itertuples(index=False):
            for framework_name in ("backtesting.py", "zipline", "nautilus"):
                symbol_frames = build_symbol_frames(
                    provider=provider,
                    fast_window=fast_window,
                    slow_window=slow_window,
                    start=start,
                    end=test_end,
                    symbols=forward_symbols,
                )
                symbol_frames = {
                    symbol: frame.loc[test_start:test_end].copy()
                    for symbol, frame in symbol_frames.items()
                }
                _, portfolio_summary, runs = run_framework_portfolio(
                    framework_name,
                    symbol_frames=symbol_frames,
                    capital_base=capital_base,
                )
                row = portfolio_summary.iloc[0].to_dict()
                row.update(
                    {
                        "provider": provider,
                        "selection": selection_name,
                        "fast_window": int(fast_window),
                        "slow_window": int(slow_window),
                        "framework": framework_name,
                        "symbols": ",".join(forward_symbols),
                    }
                )
                forward_rows.append(row)
                forward_runs[(provider, selection_name, framework_name)] = runs

    forward_report = pd.DataFrame(forward_rows)
    summary = pd.DataFrame(
        [
            {
                "stage": "vectorized_search",
                "engine": "vectorbt",
                "rows": len(train_search_report),
                "elapsed_seconds": round(float(search_elapsed["vectorized_search_seconds"].sum()), 4),
            },
            {
                "stage": "monte_carlo",
                "engine": "quant_orchestrator.monte_carlo",
                "rows": len(mc_report),
                "elapsed_seconds": round(float(mc_elapsed["mc_seconds"].sum()), 4),
            },
            {
                "stage": "forward_test",
                "engine": "backtesting.py/zipline/nautilus",
                "rows": len(forward_report),
                "symbols": ",".join(forward_symbols),
                "elapsed_seconds": round(float(forward_report["elapsed_seconds"].sum()), 4),
            },
        ]
    )

    return {
        "train_search_report": train_search_report,
        "search_elapsed": search_elapsed,
        "mc_report": mc_report,
        "mc_elapsed": mc_elapsed,
        "selections": selections,
        "forward_report": forward_report,
        "summary": summary,
        "runs": forward_runs,
    }
