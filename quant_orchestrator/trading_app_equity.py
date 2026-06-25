from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Sequence

import numpy as np
import pandas as pd

from quant_orchestrator.data import load_ohlcv, write_zipline_csv

DEFAULT_ARTIFACT_DIR = Path(
    "/home/jlee153232/PycharmProjects/optimal_trader/data/pipeline_artifacts"
)
OPTIONS_NOTEBOOK_SCORED = Path(
    "/home/jlee153232/PycharmProjects/optimal_trader/artifacts/moe_paper_trading/latest_scored.pkl"
)
TRAINED_ARTIFACT_DIR = Path("artifacts/trading_app_equity")
FEATURE_COLUMNS = (
    "ret_1d",
    "ret_5d",
    "ret_21d",
    "ret_63d",
    "dist_sma_20",
    "dist_sma_50",
    "vol_21d",
    "vol_63d",
    "volume_z_21d",
)


def options_notebook_universe() -> list[str]:
    if not OPTIONS_NOTEBOOK_SCORED.exists():
        raise FileNotFoundError(OPTIONS_NOTEBOOK_SCORED)
    scored = pd.read_pickle(OPTIONS_NOTEBOOK_SCORED)
    if "symbol" in scored.columns:
        symbols = scored["symbol"]
    else:
        symbols = pd.Series(scored.index)
    return sorted({str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()})


def train_price_model_artifact(
    *,
    provider: str,
    train_start: str,
    train_end: str,
    backtest_start: str,
    end: str | None,
    max_symbols: int | None = None,
    horizon_days: int = 21,
    symbols: Sequence[str] | None = None,
) -> Path:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline

    train_start_ts = pd.Timestamp(train_start).normalize()
    train_end_ts = pd.Timestamp(train_end).normalize()
    backtest_start_ts = pd.Timestamp(backtest_start).normalize()
    feature_start = (train_start_ts - pd.Timedelta(days=430)).date().isoformat()
    training_symbols = _normalize_symbol_list(symbols) or options_notebook_universe()
    if max_symbols is not None and int(max_symbols) > 0:
        training_symbols = training_symbols[: int(max_symbols)]

    frames = []
    missing = []
    for symbol in training_symbols:
        try:
            prices = load_ohlcv(symbol, provider=provider, start=feature_start, end=end)
        except Exception:
            missing.append(symbol)
            continue
        features = _price_feature_frame(symbol, prices, horizon_days=horizon_days)
        if not features.empty:
            frames.append(features)

    if not frames:
        raise ValueError("No Quant Warehouse price data was available for the options notebook universe")

    panel = pd.concat(frames, ignore_index=True).replace([np.inf, -np.inf], np.nan)
    train = panel.loc[
        panel["date"].between(train_start_ts, train_end_ts) & panel["target"].notna()
    ].copy()
    score = panel.loc[panel["date"].ge(backtest_start_ts)].copy()
    if train.empty:
        raise ValueError(f"No training rows available from {train_start} to {train_end}")
    if score.empty:
        raise ValueError(f"No scoring rows available from {backtest_start} onward")

    model = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=10,
            max_features="sqrt",
            class_weight="balanced",
            random_state=1337,
            n_jobs=-1,
        ),
    )
    model.fit(train.loc[:, FEATURE_COLUMNS], train["target"].astype(int))
    prob_buy = model.predict_proba(score.loc[:, FEATURE_COLUMNS])[:, 1]
    predictions = score.loc[:, ["date", "symbol", "close"]].copy()
    predictions["prob_buy"] = prob_buy
    predictions["side"] = "long"
    predictions = predictions.sort_values(["date", "prob_buy"], ascending=[True, False])

    TRAINED_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    output = TRAINED_ARTIFACT_DIR / (
        f"trained_price_model_train_{train_start_ts.date()}_{train_end_ts.date()}"
        f"_score_{backtest_start_ts.date()}.csv"
    )
    predictions.to_csv(output, index=False)
    meta = {
        "universe_source": str(OPTIONS_NOTEBOOK_SCORED),
        "universe_symbols": len(training_symbols),
        "loaded_symbols": int(panel["symbol"].nunique()),
        "missing_symbols": len(missing),
        "train_start": train_start_ts.date().isoformat(),
        "train_end": train_end_ts.date().isoformat(),
        "backtest_start": backtest_start_ts.date().isoformat(),
        "end": None if end is None else str(end),
        "train_rows": int(len(train)),
        "score_rows": int(len(predictions)),
        "feature_columns": list(FEATURE_COLUMNS),
        "horizon_days": int(horizon_days),
    }
    output.with_suffix(".json").write_text(pd.Series(meta).to_json(indent=2), encoding="utf-8")
    return output


def _price_feature_frame(
    symbol: str,
    prices: pd.DataFrame,
    *,
    horizon_days: int,
) -> pd.DataFrame:
    if prices.empty:
        return pd.DataFrame()
    df = prices.copy()
    close = pd.to_numeric(df["close"], errors="coerce")
    volume = pd.to_numeric(df["volume"], errors="coerce")
    out = pd.DataFrame(
        {
            "date": close.index.tz_convert(None).normalize(),
            "symbol": symbol.upper(),
            "close": close.to_numpy(dtype=float),
            "ret_1d": close.pct_change(1).to_numpy(dtype=float),
            "ret_5d": close.pct_change(5).to_numpy(dtype=float),
            "ret_21d": close.pct_change(21).to_numpy(dtype=float),
            "ret_63d": close.pct_change(63).to_numpy(dtype=float),
            "dist_sma_20": (close / close.rolling(20).mean() - 1.0).to_numpy(dtype=float),
            "dist_sma_50": (close / close.rolling(50).mean() - 1.0).to_numpy(dtype=float),
            "vol_21d": close.pct_change().rolling(21).std().to_numpy(dtype=float),
            "vol_63d": close.pct_change().rolling(63).std().to_numpy(dtype=float),
            "volume_z_21d": (
                (volume - volume.rolling(21).mean()) / volume.rolling(21).std()
            ).to_numpy(dtype=float),
        }
    )
    forward_return = close.shift(-int(horizon_days)) / close - 1.0
    out["target"] = (forward_return > 0.0).astype(float).to_numpy()
    out.loc[forward_return.isna().to_numpy(), "target"] = np.nan
    return out.dropna(subset=["close"])


def find_prediction_artifact(path: str | None = None) -> Path:
    if path:
        candidate = Path(path).expanduser().resolve()
        if not candidate.exists():
            raise FileNotFoundError(candidate)
        return candidate

    if OPTIONS_NOTEBOOK_SCORED.exists():
        return OPTIONS_NOTEBOOK_SCORED

    candidates = []
    for candidate in DEFAULT_ARTIFACT_DIR.glob("ml_predictions_*.csv"):
        try:
            probe = pd.read_csv(candidate, usecols=["date", "symbol"])
        except Exception:
            continue
        candidates.append((probe["symbol"].nunique(), len(probe), probe["date"].nunique(), candidate))
    if not candidates:
        raise FileNotFoundError(f"No ml_predictions_*.csv files found in {DEFAULT_ARTIFACT_DIR}")
    return sorted(candidates, reverse=True)[0][-1]


def load_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_pickle(path) if path.suffix.lower() == ".pkl" else pd.read_csv(path)
    if "symbol" not in df.columns:
        df = df.reset_index()
    required = {"date", "symbol"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.tz_localize(None).dt.normalize()
    out["symbol"] = out["symbol"].astype(str).str.strip().str.upper()
    score_col = next(
        (
            col
            for col in (
                "prob_buy",
                "buy_score_mean_raw_pct6",
                "signal_score",
                "prediction_score",
                "prediction",
                "raw_prediction",
            )
            if col in out
        ),
        None,
    )
    if score_col is None:
        raise ValueError(f"{path} has no supported score column")
    out["score"] = pd.to_numeric(out[score_col], errors="coerce")
    out["side"] = out.get("side", pd.Series("", index=out.index)).astype(str).str.lower()
    out = out.loc[out["date"].notna() & out["symbol"].ne("") & out["score"].notna()].copy()
    return out.sort_values(["date", "score"], ascending=[True, False])


def build_equity_targets(
    predictions: pd.DataFrame,
    *,
    top_k: int,
    gross_exposure: float,
    symbols: Sequence[str] | None = None,
) -> pd.DataFrame:
    rows = []
    top_k = max(1, int(top_k))
    gross_exposure = max(0.0, min(float(gross_exposure), 1.0))
    allowed_symbols = set(_normalize_symbol_list(symbols) or ())
    if allowed_symbols:
        predictions = predictions.loc[predictions["symbol"].isin(allowed_symbols)].copy()

    for date, day in predictions.groupby("date", sort=True):
        long_candidates = day.loc[~day["side"].eq("short")].sort_values("score", ascending=False)
        selected = long_candidates.head(top_k)
        if selected.empty:
            continue
        weight = gross_exposure / float(len(selected))
        for symbol in selected["symbol"]:
            rows.append({"date": date, "symbol": symbol, "weight": weight})

    if not rows:
        raise ValueError("No equity long candidates were selected from the prediction artifact")
    target_frame = pd.DataFrame(rows)
    return target_frame.pivot(index="date", columns="symbol", values="weight").fillna(0.0)


def load_prices_for_targets(
    targets: pd.DataFrame,
    *,
    provider: str,
    end: str | None = None,
) -> dict[str, pd.DataFrame]:
    start = str(pd.Timestamp(targets.index.min()).date())
    prices = {}
    missing = []
    for symbol in targets.columns:
        try:
            frame = load_ohlcv(symbol, provider=provider, start=start, end=end)
        except Exception:
            missing.append(symbol)
            continue
        if frame.empty:
            missing.append(symbol)
            continue
        prices[symbol] = frame
    if not prices:
        raise ValueError(
            f"No target symbols had {provider} prices in Quant Warehouse for {start} to {end}"
        )
    return prices


def summarize_equity(
    *,
    framework: str,
    equity: pd.Series,
    trades: int,
    elapsed_seconds: float,
    artifact: Path,
    symbols: int,
    signal_universe: int,
) -> pd.DataFrame:
    returns = equity.pct_change().fillna(0.0)
    drawdown = equity / equity.cummax() - 1.0
    return pd.DataFrame(
        [
            {
                "framework": framework,
                "strategy": "trading_app_equity",
                "artifact": artifact.name,
                "signal_universe": int(signal_universe),
                "symbols": int(symbols),
                "start": str(equity.index[0].date()),
                "end": str(equity.index[-1].date()),
                "bars": int(len(equity)),
                "trades": int(trades),
                "final_equity": round(float(equity.iloc[-1]), 2),
                "total_return": round(float(equity.iloc[-1] / equity.iloc[0] - 1.0), 4),
                "max_drawdown": round(float(drawdown.min()), 4),
                "daily_vol": round(float(returns.std()), 4),
                "elapsed_seconds": round(float(elapsed_seconds), 4),
            }
        ]
    )


def target_weights_to_equity(
    *,
    targets: pd.DataFrame,
    prices: dict[str, pd.DataFrame],
    capital_base: float,
) -> tuple[pd.Series, int]:
    close = pd.DataFrame({symbol: df["close"] for symbol, df in prices.items()}).sort_index()
    close.index = close.index.tz_convert(None).normalize()
    first_target = pd.Timestamp(targets.index.min()).tz_localize(None).normalize()
    close = close.loc[close.index >= first_target].copy()
    weights = targets.reindex(close.index).reindex(columns=close.columns).ffill().fillna(0.0)

    cash = float(capital_base)
    shares = pd.Series(0, index=close.columns, dtype=float)
    values = []
    trades = 0
    for date, row in close.iterrows():
        target_value = weights.loc[date] * float(capital_base)
        target_shares = (target_value / row).replace([float("inf"), -float("inf")], 0.0).fillna(0.0)
        target_shares = target_shares.astype(int).astype(float)
        delta = target_shares - shares
        trades += int(delta.ne(0).sum())
        cash -= float((delta * row).sum())
        shares = target_shares
        values.append(cash + float((shares * row).sum()))
    return pd.Series(values, index=close.index, name="portfolio_value"), trades


def run_trading_app_zipline(
    *,
    prediction_artifact: str | None,
    provider: str,
    top_k: int,
    gross_exposure: float,
    capital_base: float,
    end: str | None = None,
    symbols: Sequence[str] | None = None,
) -> pd.DataFrame:
    from zipline import run_algorithm
    from zipline.api import order_target_percent, record, symbol as zipline_symbol
    from zipline.data import bundles
    from zipline.data.bundles.csvdir import csvdir_equities
    from zipline.data.bundles.core import UnknownBundle
    from zipline.utils.calendar_utils import get_calendar

    started = perf_counter()
    artifact = find_prediction_artifact(prediction_artifact)
    predictions = load_predictions(artifact)
    signal_universe = int(predictions["symbol"].nunique())
    targets = build_equity_targets(
        predictions,
        top_k=top_k,
        gross_exposure=gross_exposure,
        symbols=symbols,
    )
    prices = load_prices_for_targets(targets, provider=provider, end=end)
    targets = targets.reindex(columns=prices.keys()).fillna(0.0)

    with tempfile_directory("quant-orchestrator-trading-app-zipline-") as tmp:
        csv_root = tmp / "csv"
        for symbol, frame in prices.items():
            write_zipline_csv(symbol, frame, csv_root)

        bundle_name = "quant_warehouse_trading_app_equity"
        try:
            bundles.unregister(bundle_name)
        except (KeyError, UnknownBundle):
            pass
        bundles.register(bundle_name, csvdir_equities(["daily"], str(csv_root)), calendar_name="XNYS")
        bundles.ingest(bundle_name)

        close_index = pd.DataFrame({symbol: frame["close"] for symbol, frame in prices.items()}).index
        close_index = close_index.tz_convert(None).normalize()
        target_schedule = targets.reindex(close_index).ffill().fillna(0.0)
        target_lookup = {
            pd.Timestamp(date).normalize(): row.dropna().to_dict()
            for date, row in target_schedule.iterrows()
        }

        def initialize(context):
            context.assets = {symbol: zipline_symbol(symbol) for symbol in targets.columns}

        def handle_data(context, data):
            today = pd.Timestamp(data.current_dt).tz_convert("UTC").tz_localize(None).normalize()
            weights = target_lookup.get(today, {})
            for symbol, asset in context.assets.items():
                if not data.can_trade(asset):
                    continue
                order_target_percent(asset, float(weights.get(symbol, 0.0)))
            record(active=len([w for w in weights.values() if float(w) > 0]))

        run_algorithm(
            start=pd.Timestamp(target_schedule.index.min()),
            end=pd.Timestamp(target_schedule.index.max()),
            initialize=initialize,
            handle_data=handle_data,
            capital_base=capital_base,
            data_frequency="daily",
            bundle=bundle_name,
            trading_calendar=get_calendar("XNYS"),
            default_extension=False,
        )

    equity, trades = target_weights_to_equity(targets=targets, prices=prices, capital_base=capital_base)
    return summarize_equity(
        framework="zipline",
        equity=equity,
        trades=trades,
        elapsed_seconds=perf_counter() - started,
        artifact=artifact,
        symbols=len(targets.columns),
        signal_universe=signal_universe,
    )


def run_trading_app_nautilus(
    *,
    prediction_artifact: str | None,
    provider: str,
    top_k: int,
    gross_exposure: float,
    capital_base: float,
    end: str | None = None,
    symbols: Sequence[str] | None = None,
) -> pd.DataFrame:
    from nautilus_trader.backtest.engine import BacktestEngine
    from nautilus_trader.config import BacktestEngineConfig, LoggingConfig, StrategyConfig
    from nautilus_trader.model.currencies import USD
    from nautilus_trader.model.data import BarType
    from nautilus_trader.model.enums import AccountType, OmsType, OrderSide, TimeInForce
    from nautilus_trader.model.identifiers import Venue
    from nautilus_trader.model.objects import Money, Quantity
    from nautilus_trader.persistence.wranglers import BarDataWrangler
    from nautilus_trader.test_kit.providers import TestInstrumentProvider
    from nautilus_trader.trading.strategy import Strategy

    started = perf_counter()
    artifact = find_prediction_artifact(prediction_artifact)
    predictions = load_predictions(artifact)
    signal_universe = int(predictions["symbol"].nunique())
    targets = build_equity_targets(
        predictions,
        top_k=top_k,
        gross_exposure=gross_exposure,
        symbols=symbols,
    )
    prices = load_prices_for_targets(targets, provider=provider, end=end)
    targets = targets.reindex(columns=prices.keys()).fillna(0.0)
    close = pd.DataFrame({symbol: frame["close"] for symbol, frame in prices.items()})
    close.index = close.index.tz_convert(None).normalize()
    target_shares = (targets.reindex(close.index).ffill().fillna(0.0) * capital_base / close).fillna(0.0)
    target_shares = target_shares.astype(int)

    class TargetConfig(StrategyConfig, frozen=True):
        target_shares: object
        bar_types: object

    class TargetWeightStrategy(Strategy):
        def __init__(self, config: TargetConfig):
            super().__init__(config)
            self.current = {symbol: 0 for symbol in config.target_shares.columns}

        def on_start(self) -> None:
            for bar_type in self.config.bar_types.values():
                self.subscribe_bars(bar_type)

        def on_bar(self, bar) -> None:
            symbol = str(bar.bar_type.instrument_id.symbol)
            date = pd.Timestamp(bar.ts_event, unit="ns", tz="UTC").tz_localize(None).normalize()
            if date not in self.config.target_shares.index or symbol not in self.current:
                return
            target = int(self.config.target_shares.loc[date, symbol])
            delta = target - int(self.current[symbol])
            if delta == 0:
                return
            order = self.order_factory.market(
                instrument_id=bar.bar_type.instrument_id,
                order_side=OrderSide.BUY if delta > 0 else OrderSide.SELL,
                quantity=Quantity.from_int(abs(delta)),
                time_in_force=TimeInForce.IOC,
            )
            self.submit_order(order)
            self.current[symbol] = target

    engine = BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level="ERROR")))
    instruments = {}
    bar_types = {}
    for symbol, frame in prices.items():
        instrument = TestInstrumentProvider.equity(symbol=symbol)
        instruments[symbol] = instrument
        venue = Venue(str(instrument.id.venue))
        if venue not in {Venue(str(inst.id.venue)) for inst in list(instruments.values())[:-1]}:
            engine.add_venue(
                venue=venue,
                oms_type=OmsType.NETTING,
                account_type=AccountType.MARGIN,
                starting_balances=[Money(capital_base, USD)],
                base_currency=USD,
            )
        bar_type = BarType.from_str(f"{instrument.id}-1-DAY-LAST-EXTERNAL")
        bar_types[symbol] = bar_type
        engine.add_instrument(instrument)
        engine.add_data(BarDataWrangler(bar_type, instrument).process(frame))

    engine.add_strategy(TargetWeightStrategy(TargetConfig(target_shares=target_shares, bar_types=bar_types)))
    engine.run()
    fills_report = engine.trader.generate_order_fills_report()
    engine.dispose()

    equity, _trades = target_weights_to_equity(targets=targets, prices=prices, capital_base=capital_base)
    return summarize_equity(
        framework="nautilus",
        equity=equity,
        trades=len(fills_report),
        elapsed_seconds=perf_counter() - started,
        artifact=artifact,
        symbols=len(targets.columns),
        signal_universe=signal_universe,
    )


class tempfile_directory:
    def __init__(self, prefix: str):
        import tempfile

        self._tmp = tempfile.TemporaryDirectory(prefix=prefix)
        self.path: Path | None = None

    def __enter__(self) -> Path:
        self.path = Path(self._tmp.__enter__())
        return self.path

    def __exit__(self, exc_type, exc, tb):
        return self._tmp.__exit__(exc_type, exc, tb)


def _normalize_symbol_list(symbols: Sequence[str] | None) -> list[str]:
    if not symbols:
        return []
    return sorted({str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()})
