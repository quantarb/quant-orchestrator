"""Full-history warehouse training followed by latest-date equity scoring.

Uses the shared warehouse model, objectives, scheduler and prediction adapter.
These same-date fitted scores are deployment inputs, not out-of-sample results.
"""
from datetime import datetime, timedelta
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import polars as pl
import torch

from quant_warehouse.warehouse.api import Warehouse


def _recent_compatible_run(parent, *, score_date, max_age_hours, expected):
    """Return the newest complete, configuration-matched live run within the age limit."""
    if max_age_hours <= 0 or not parent.exists():
        return None
    now = datetime.now().timestamp()
    candidates = []
    for run in parent.iterdir():
        if not run.is_dir():
            continue
        required = {
            "status": run / "status.json",
            "configuration": run / "configuration.json",
            "checkpoint": run / "checkpoint_latest.pt",
            "predictions": run / "latest_predictions.parquet",
            "prices": run / "latest_prices.parquet",
        }
        if not all(path.is_file() for path in required.values()):
            continue
        age_hours = (now - required["checkpoint"].stat().st_mtime) / 3600.0
        if age_hours < 0 or age_hours >= max_age_hours:
            continue
        try:
            status = json.loads(required["status"].read_text())
            configuration = json.loads(required["configuration"].read_text())
        except (OSError, ValueError, TypeError):
            continue
        if status.get("stage") != "complete" or status.get("score_date") != score_date:
            continue
        if any(configuration.get(name) != value for name, value in expected.items()):
            continue
        if status.get("prediction_path") != str(required["predictions"].resolve()):
            continue
        if status.get("prices_path") != str(required["prices"].resolve()):
            continue
        if status.get("checkpoint") != str(required["checkpoint"].resolve()):
            continue
        candidates.append((required["checkpoint"].stat().st_mtime, age_hours, run, status))
    if not candidates:
        return None
    _, age_hours, run, status = max(candidates, key=lambda item: item[0])
    return run, age_hours, status


def latest_warehouse_equity_date(min_market_cap, *, warehouse=None):
    """Latest finite equity close in the requested stored US stock universe."""
    if min_market_cap <= 0:
        raise ValueError('min_market_cap must be positive')
    warehouse = warehouse or Warehouse()
    profiles = warehouse.catalog.query_symbol_profiles(provider='fmp', min_market_cap=min_market_cap,
        country='US', exchanges=['NASDAQ','NYSE'], exclude_etf=True, exclude_fund=True)
    dates = []
    for profile in profiles:
        prices = warehouse.read_prices(profile.symbol, provider='fmp', start='1900-01-01')
        if prices.height < 2:
            continue
        last = prices.filter(pl.col('close').is_finite() & (pl.col('close') > 0))['date'].max()
        if last is not None:
            dates.append(last)
    if not dates:
        raise ValueError('No eligible stored equity price history in the requested universe')
    return max(dates).strftime('%Y-%m-%d')


def train_latest_warehouse_model(output_dir, *, min_market_cap=10_000_000_000,
        epochs=1, batch_size=64, d_model=64, num_heads=4, layers=2, device='cuda', seed=0,
        reconstruction_weight=0.1, checkpoint_every_batches=10, progress_updates_per_epoch=10,
        reuse_max_age_hours=24.0, warehouse=None):
    """Train fresh weights on all stored history, including the partial latest year.

    The architecture/objectives match the completed warehouse equity model.
    Run output must be new. No options training, historical backtest, data refresh,
    broker connection or order submission is performed.
    """
    from .warehouse_multirate_training import run_warehouse_training

    output = Path(output_dir).resolve()
    warehouse = warehouse or Warehouse()
    score_date = latest_warehouse_equity_date(min_market_cap, warehouse=warehouse)
    cutoff = (datetime.fromisoformat(score_date) + timedelta(days=1)).date().isoformat()
    expected = dict(
        min_market_cap=min_market_cap, epochs=epochs, batch_size=batch_size,
        d_model=d_model, num_heads=num_heads, layers=layers, seed=seed,
        reconstruction_weight=reconstruction_weight, train_end_date=cutoff,
        prediction_start_date=score_date, prediction_end_date=score_date,
        options_per_side=0, self_supervision='both',
    )
    recent = _recent_compatible_run(
        output.parent, score_date=score_date,
        max_age_hours=float(reuse_max_age_hours), expected=expected,
    )
    if recent is not None:
        run, age_hours, status = recent
        result = dict(status)
        result.update(reused=True, reuse_age_hours=age_hours,
                      source_output_dir=str(run.resolve()), requested_output_dir=str(output))
        print(f'[warehouse-live] reusing={run} checkpoint_age_hours={age_hours:.2f} score_date={score_date}', flush=True)
        return result
    if output.exists():
        raise FileExistsError(output)
    args = SimpleNamespace(output_dir=output, min_market_cap=min_market_cap,
        warehouse_start_date='1900-01-01', warehouse_option_start_date=None,
        train_end_date=cutoff, prediction_start_date=score_date, prediction_end_date=score_date,
        options_per_side=0, epochs=epochs, batch_size=batch_size, d_model=d_model,
        num_heads=num_heads, layers=layers, device=device, seed=seed,
        self_supervision='both', reconstruction_weight=reconstruction_weight,
        checkpoint_every_batches=checkpoint_every_batches, progress_updates_per_epoch=progress_updates_per_epoch,
        checkpoint=None, resume_training=False, inference_only=False, sequence_mode='annual_memory',
        mixed_precision=False, fp8=False, compile_model=False, max_samples=0, validation_fraction=0,
        issuer_context='full', legacy_rate_fusion=False, attention_backend='pytorch', optimizer='adamw',
        grad_accumulation_steps=1, training_sequence_stride=0, mrl_dimensions='', disable_document_tasks=True,
        skip_predictions=False, learned_aggregation_gate=False)
    print(f'[warehouse-live] score_date={score_date} training_cutoff_exclusive={cutoff} options=0', flush=True)
    try:
        return run_warehouse_training(args, latest_only=True, warehouse=warehouse)
    except Exception as exc:
        if getattr(args, '_warehouse_run_started', False):
            (output/'status.json').write_text(json.dumps(dict(stage='failed', error=str(exc)), indent=2))
        raise


def score_latest_equities(model, stream, args):
    """Retain current-year context, but emit only the common latest stored date."""
    from .annual_memory import AnnualMemory
    from .warehouse_multirate_training import equity_inference_batches, predict_batch
    from quant_orchestrator.platforms.ml_frameworks.torch.models.transformers.multirate.temporal_tasks import SUPERVISED_TARGET_TASK_NAMES

    date = datetime.fromisoformat(args.prediction_end_date)
    start = datetime(date.year,1,1)
    symbols, prices, missing = [], [], []
    for symbol, frame in sorted(stream.prices.items()):
        current = frame.filter((pl.col('date') == date) & pl.col('close').is_finite() & (pl.col('close') > 0))
        if current.is_empty():
            missing.append(symbol)
        else:
            symbols.append(symbol)
            prices.append(current.select('date','close').with_columns(pl.lit(symbol).alias('symbol')))
    if not symbols:
        raise ValueError('No equity prices on the latest scoring date')
    model.eval()
    rows=[]; memory=AnnualMemory(); began=perf_counter(); waiting=began
    wait_seconds=prediction_seconds=0.
    with torch.inference_mode():
        for batch_number, batch in enumerate(equity_inference_batches(stream,args.batch_size,start,date,symbols=symbols),1):
            wait_seconds += perf_counter()-waiting
            step = perf_counter()
            rows.extend(predict_batch(model,batch,memory,stream.layout,score_date=date))
            prediction_seconds += perf_counter()-step
            print(f'[warehouse-latest] batch={batch_number} scored={len(rows)}/{len(symbols)} seconds={perf_counter()-began:.1f} batch_wait_seconds={wait_seconds:.1f} prediction_seconds={prediction_seconds:.1f}',flush=True)
            waiting=perf_counter()
    scores=pl.DataFrame(rows, infer_schema_length=None)
    if scores.height != len(symbols) or sorted(scores['symbol'].to_list()) != symbols or scores['date'].unique().to_list() != [date]:
        raise ValueError('Latest-date scoring must produce exactly one row per priced equity')
    if scores.select(pl.any_horizontal([~pl.col(c).is_finite() | pl.col(c).is_null() for c in SUPERVISED_TARGET_TASK_NAMES]).any()).item():
        raise ValueError('Nonfinite latest-date model predictions')
    prediction_path = args.output_dir/'latest_predictions.parquet'
    prices_path = args.output_dir/'latest_prices.parquet'
    scores.write_parquet(prediction_path)
    pl.concat(prices).write_parquet(prices_path)
    result=dict(score_date=date.date().isoformat(), predictions=len(rows),
        prediction_path=str(prediction_path.resolve()), prices_path=str(prices_path.resolve()),
        checkpoint=str((args.output_dir/'checkpoint_latest.pt').resolve()),
        inference_seconds=perf_counter()-began, batch_wait_seconds=wait_seconds, prediction_seconds=prediction_seconds,
        trained_equities=len(stream.prices), missing_latest_prices=missing,
        evaluation_mode='latest_date_in_sample', inference_initialization='empty_memory_current_year_context',
        options_per_side=0)
    (args.output_dir/'latest_prediction_summary.json').write_text(json.dumps(result,indent=2))
    return result
