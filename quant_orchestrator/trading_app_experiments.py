from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_orchestrator.artifacts import get_artifact_store
from quant_orchestrator.experiments import (
    TradingAppExperimentSpec,
    WalkForwardWindow,
    build_walk_forward_windows,
)
from quant_orchestrator.tracking import log_backtest_run
from quant_orchestrator.trading_app_equity import (
    run_trading_app_nautilus,
    run_trading_app_zipline,
    train_price_model_artifact,
)


def run_trading_app_experiment(
    spec: TradingAppExperimentSpec,
    *,
    track: bool = True,
    tracking_uri: str | None = None,
) -> pd.DataFrame:
    if spec.window.mode == "fixed":
        return _run_one_window(
            spec,
            window=WalkForwardWindow(
                pd.Timestamp(spec.train_start).normalize(),
                pd.Timestamp(spec.train_end).normalize(),
                pd.Timestamp(spec.backtest_start).normalize(),
                pd.Timestamp(spec.end).normalize() if spec.end else pd.Timestamp.today().normalize(),
            ),
            track=track,
            tracking_uri=tracking_uri,
        )

    frames = [
        _run_one_window(spec, window=window, track=track, tracking_uri=tracking_uri)
        for window in build_walk_forward_windows(spec.window)
    ]
    if not frames:
        raise ValueError(f"No walk-forward windows were generated for {spec.window}")
    return pd.concat(frames, ignore_index=True)


def _run_one_window(
    spec: TradingAppExperimentSpec,
    *,
    window: WalkForwardWindow,
    track: bool,
    tracking_uri: str | None,
) -> pd.DataFrame:
    prediction_artifact = spec.prediction_artifact
    if spec.train_model:
        prediction_artifact = train_price_model_artifact(
            provider=spec.provider,
            train_start=window.train_start.date().isoformat(),
            train_end=window.train_end.date().isoformat(),
            backtest_start=window.test_start.date().isoformat(),
            end=window.test_end.date().isoformat(),
            max_symbols=spec.max_symbols,
            symbols=spec.universe.train_symbols(),
        )
    if prediction_artifact is None:
        prediction_artifact_text = None
    else:
        prediction_artifact_text = str(
            prediction_artifact.path if hasattr(prediction_artifact, "path") else Path(prediction_artifact)
        )

    frames = []
    if spec.framework in {"zipline", "all"}:
        frames.append(
            run_trading_app_zipline(
                prediction_artifact=prediction_artifact_text,
                provider=spec.provider,
                top_k=spec.top_k,
                gross_exposure=spec.gross_exposure,
                capital_base=spec.capital_base,
                end=window.test_end.date().isoformat(),
                symbols=spec.universe.backtest_symbols(),
            )
        )
    if spec.framework in {"nautilus", "all"}:
        frames.append(
            run_trading_app_nautilus(
                prediction_artifact=prediction_artifact_text,
                provider=spec.provider,
                top_k=spec.top_k,
                gross_exposure=spec.gross_exposure,
                capital_base=spec.capital_base,
                end=window.test_end.date().isoformat(),
                symbols=spec.universe.backtest_symbols(),
            )
        )
    if not frames:
        raise ValueError(f"Unsupported framework for trading-app experiment: {spec.framework}")

    result = pd.concat(frames, ignore_index=True)
    result.insert(0, "experiment", spec.name)
    result.insert(1, "window", window.label)
    result.insert(2, "train_start", window.train_start.date().isoformat())
    result.insert(3, "train_end", window.train_end.date().isoformat())
    result.insert(4, "test_start", window.test_start.date().isoformat())
    result.insert(5, "test_end", window.test_end.date().isoformat())

    if track:
        _track_result(spec, window=window, result=result, tracking_uri=tracking_uri)
    return result


def _track_result(
    spec: TradingAppExperimentSpec,
    *,
    window: WalkForwardWindow,
    result: pd.DataFrame,
    tracking_uri: str | None,
) -> None:
    metrics = {
        "final_equity": float(result["final_equity"].mean()),
        "total_return": float(result["total_return"].mean()),
        "max_drawdown": float(result["max_drawdown"].min()),
        "trades": float(result["trades"].sum()),
    }
    params = {
        "framework": spec.framework,
        "provider": spec.provider,
        "top_k": spec.top_k,
        "gross_exposure": spec.gross_exposure,
        "capital_base": spec.capital_base,
        "train_symbols": spec.universe.train_symbols(),
        "backtest_symbols": spec.universe.backtest_symbols(),
        "window_mode": spec.window.mode,
        "train_start": window.train_start.date().isoformat(),
        "train_end": window.train_end.date().isoformat(),
        "test_start": window.test_start.date().isoformat(),
        "test_end": window.test_end.date().isoformat(),
    }
    store = get_artifact_store()
    run = store.create_run(
        run_type="backtest",
        name=f"{spec.name}-{window.label}",
        params=params,
        tags={
            "strategy": "trading_app_equity",
            "framework": spec.framework,
            "data_source": f"quant-warehouse:{spec.provider}",
        },
    )
    try:
        summary_artifact = store.save_dataframe(
            run_id=run.id,
            kind="backtest_result",
            name="trading_app_equity_summary",
            frame=result,
            metadata=params,
        )
        store.save_json(
            run_id=run.id,
            kind="strategy_artifact",
            name="trading_app_equity_strategy",
            payload=params,
            metadata={"backtest_result_uri": summary_artifact.uri},
        )
    except Exception as exc:
        store.fail_run(run.id, exc)
        raise
    store.complete_run(run.id, metrics=metrics)

    tracker_kwargs = {"tracking_uri": tracking_uri} if tracking_uri else {}
    log_backtest_run(
        run_name=f"{spec.name}-{window.label}",
        engine=spec.framework,
        strategy="trading_app_equity",
        data_source=f"quant-warehouse:{spec.provider}",
        params=params,
        metrics=metrics,
        artifacts={"summary": summary_artifact.path},
        experiment=spec.name,
        **tracker_kwargs,
    )
