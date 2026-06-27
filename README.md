# Quant Orchestrator

Opinionated Dagster and MLflow research orchestration around data stored in `quant-warehouse`.

## Environment

```bash
conda env create -f environment.yml
conda activate quant-orchestrator
```

The environment installs `quant-warehouse` from GitHub, Dagster, MLflow, plus Zipline Reloaded and NautilusTrader.

## Install

For a package install:

```bash
pip install "git+https://github.com/quantarb/quant-orchestrator.git"
```

For the full local research/backtesting environment:

```bash
pip install -e ".[all,dev]"
```

For just ThetaData-backed options backtests through Optopsy:

```bash
pip install -e ".[options]"
```

For CUDA-first PyTorch workflows, install the CUDA extra and use the PyTorch wheel index that matches the host driver:

```bash
pip install -e ".[cuda]"
# Example for CUDA 12.4:
pip install --index-url https://download.pytorch.org/whl/cu124 torch
```

## Provider Platform

`quant-orchestrator` follows an OpenBB-style provider layout for model and backtesting extension categories:

- `quant_orchestrator.platforms.ml_frameworks`
- `quant_orchestrator.platforms.backtesting_frameworks`

Installed packages can register providers through entry points:

```toml
[project.entry-points."quant_orchestrator.ml_framework"]
my_framework = "my_package.ml_framework:provider"

[project.entry-points."quant_orchestrator.backtesting_framework"]
my_engine = "my_package.backtesting_framework:provider"
```

At runtime, providers are resolved from the registry:

```python
from quant_orchestrator.platform import registry

registry.list("backtesting_framework")
engine_cls = registry.adapter("backtesting_framework", "optopsy")
engine = engine_cls()
```

## Experiment Tracking

MLflow is the built-in experiment tracker. Use it through the orchestrator tracking interface so Dagster jobs, backtests, and model training code have one consistent tracking API:

```python
from quant_orchestrator.tracking import log_backtest_run

log_backtest_run(
    run_name="optopsy-tsla-2025q1",
    engine="optopsy",
    strategy="long_calls",
    data_source="quant-warehouse:thetadata",
    params={"delta_min": 0.25, "delta_max": 0.45},
    metrics={"sharpe": 1.2, "max_drawdown": -0.08},
    artifacts={"trades": "artifacts/trades.csv", "equity": "artifacts/equity.csv"},
)
```

By default, tracking uses `sqlite:///artifacts/mlflow/mlflow.db`. Override it
with `tracking_uri=...`, `QUANT_ORCHESTRATOR_MLFLOW_TRACKING_URI`, or `MLFLOW_TRACKING_URI`.

## Artifact Registry

`quant-orchestrator` owns ML training, backtest, model, prediction, and strategy artifacts. Apps such as `optimal_trader` should ask the orchestrator to train or backtest, then load the returned artifact URI or path instead of maintaining separate research storage.

The registry is intentionally schema-light: sklearn, PyTorch, Flair NLP, Zipline, NautilusTrader, and other frameworks can save their native files, directories, dataframes, JSON, text reports, or pickled objects without forcing every output into one common report shape.

```python
from quant_orchestrator.artifacts import get_artifact_store

store = get_artifact_store()
run = store.create_run(
    run_type="ml_training",
    name="flair-news-classifier",
    params={"dataset": "warehouse:event_labels"},
)
report = store.register_file(
    run_id=run.id,
    kind="ml_report",
    name="flair-output",
    path="outputs/flair-training-run",
)
store.complete_run(run.id, metrics={"validation_accuracy": 0.73})

print(report.uri, report.path)
```

By default artifacts are written under `artifacts/orchestrator`. Override this with `QUANT_ORCHESTRATOR_ARTIFACT_ROOT`.

## Notebook Boundary

Use `notebooks/` for one-off research workflows that consume prepared datasets from `quant-warehouse`: model training, backtesting, walk-forward analysis, Monte Carlo analysis, and equity-curve analysis. Do not build feature families, labels, warehouse refreshes, or vendor data pulls in this repo; implement those in `quant-warehouse` first and consume the resulting dataset here.

## Load data

The examples read prices already stored in Quant Warehouse:

```bash
quant-orchestrator --framework pandas --symbols AAPL MSFT --start 2020-01-01
```

If the warehouse has no prices for a symbol, refresh it from the Quant Warehouse repo/env first:

```bash
conda activate quant-warehouse
quant-warehouse refresh AAPL --sections prices --providers yfinance
```

## Run examples

```bash
quant-orchestrator --framework all --symbols AAPL --start 2023-01-01 --fast-window 5 --slow-window 10
quant-orchestrator --framework pandas --symbols AAPL MSFT --start 2020-01-01
quant-orchestrator --framework zipline --symbols AAPL --start 2020-01-01
quant-orchestrator --framework nautilus --symbols AAPL --start 2020-01-01
```

Zipline Reloaded uses `run_algorithm()` with a temporary CSV bundle built from Quant Warehouse prices. NautilusTrader uses `BacktestEngine` and `BarDataWrangler` to convert the same OHLCV frame into Nautilus bar objects.

The printed table separates normalized strategy performance from framework runtime. All frameworks are scored with the same daily close, long/flat, fixed-share SMA crossover model so return, drawdown, volatility, and trade counts are comparable. Runtime still includes each framework adapter's setup and backtest execution.

## Trading App Equity

The `trading-app-equity` strategy ports the equity-only variant of the `optimal_trader` trading app signal into Zipline Reloaded and NautilusTrader. It loads orchestrator-registered ML prediction artifacts, selects top `prob_buy` long candidates, and runs equal-weight equity targets.

```bash
quant-orchestrator --strategy trading-app-equity --framework all --top-k 40 --gross-exposure 0.95
```

Use `--prediction-artifact /path/to/ml_predictions.csv` or `--prediction-artifact artifact:<id>` to pin a historical prediction artifact. If no artifact is provided, the latest registered `trading_app_equity_predictions` artifact is used.

To train the simple equity variant on 2020 Quant Warehouse price features and backtest from 2021 onward over the same options-notebook universe:

```bash
quant-orchestrator --strategy trading-app-equity --framework all --train-model \
  --train-start 2020-01-01 --train-end 2020-12-31 --backtest-start 2021-01-01 \
  --top-k 40 --gross-exposure 0.75
```

Add `--end 2021-12-31` to run only the 2021 calendar year.

Train and backtest universes can be separated:

```bash
quant-orchestrator --strategy trading-app-equity --framework zipline --train-model \
  --train-symbols AAPL MSFT NVDA AMZN GOOGL META TSLA \
  --backtest-symbols SPY QQQ IWM \
  --train-start 2020-01-01 --train-end 2021-12-31 --backtest-start 2022-01-01
```

## Scheduled Orchestration

Dagster definitions live in `quant_orchestrator.dagster_defs`:

```bash
dagster dev -m quant_orchestrator.dagster_defs
```

The module exposes `trading_app_experiment_job` plus daily and weekday schedules. The job
uses `TradingAppExperimentSpec`, can train on one symbol universe and backtest on another,
and logs summaries through MLflow.

## Walk-Forward and Monte Carlo

The experiment layer supports fixed, anchored, and rolling windows:

```python
from quant_orchestrator.experiments import TradingAppExperimentSpec, UniverseSplit, WindowSpec
from quant_orchestrator.trading_app_experiments import run_trading_app_experiment

spec = TradingAppExperimentSpec(
    name="anchored-trading-app",
    train_model=True,
    framework="zipline",
    universe=UniverseSplit(
        train=("AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA"),
        backtest=("SPY", "QQQ", "IWM"),
    ),
    window=WindowSpec(
        mode="anchored",
        train_start="2020-01-01",
        train_end="2020-12-31",
        test_start="2021-01-01",
        test_end="2022-12-31",
        step="30D",
        test_length="30D",
    ),
)

summary = run_trading_app_experiment(spec)
```

Monte Carlo simulations are available as reusable primitives:

```python
from quant_orchestrator.monte_carlo import simulate_return_paths

simulation = simulate_return_paths(returns, iterations=1000, horizon=252, block_size=5)
```
