# Quant Orchestrator

Composable Dagster and MLflow research orchestration around data stored in `quant-warehouse`.

`quant-orchestrator` should coordinate research workflows without assuming a single shape. A run can be ML training only, backtesting only, training followed by backtesting, or an external-engine strategy run that uses warehouse data and stores the native outputs.

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

## Platform Capabilities

The platform contract is intentionally small:

- resolve prepared datasets from `quant-warehouse`
- run ML framework adapters when a workflow needs model training or inference
- run backtesting framework adapters when a workflow needs strategy evaluation
- track runs in MLflow
- store native model, report, prediction, backtest, and strategy artifacts in the artifact registry

These capabilities are independent. A workflow does not need an ML model to run a backtest, and it does not need a backtest to train a model.

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

Backtesting framework providers are adapters around native engines, not strategy assumptions. For example, a QuantConnect strategy should be exposed through a backtesting framework adapter or runner that receives prepared `quant-warehouse` data, runs the native QuantConnect strategy, and registers whatever files/reports that engine emits.

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

`quant-orchestrator` owns ML training, backtest, model, prediction, and strategy artifacts. Downstream apps should ask the orchestrator to train, infer, backtest, or run external strategy evaluations, then load the returned artifact URI or path instead of maintaining separate research storage.

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

Use `notebooks/` for one-off research workflows and demonstrations that consume prepared datasets from `quant-warehouse`. If a workflow becomes a repeated platform capability, move the reusable part into package code and keep the notebook as an example. Do not build feature families, labels, warehouse refreshes, or vendor data pulls in this repo; implement those in `quant-warehouse` first and consume the resulting dataset here.

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

## Built-In Examples

```bash
quant-orchestrator --framework all --symbols AAPL --start 2023-01-01 --fast-window 5 --slow-window 10
quant-orchestrator --framework pandas --symbols AAPL MSFT --start 2020-01-01
quant-orchestrator --framework zipline --symbols AAPL --start 2020-01-01
quant-orchestrator --framework nautilus --symbols AAPL --start 2020-01-01
```

Zipline Reloaded uses `run_algorithm()` with a temporary CSV bundle built from Quant Warehouse prices. NautilusTrader uses `BacktestEngine` and `BarDataWrangler` to convert the same OHLCV frame into Nautilus bar objects.

The CLI examples are smoke tests and adapter demonstrations, not the platform contract. They use a simple SMA strategy so framework runtime and output shape can be compared without implying that every workflow is equity-only or ML-driven.

## Scheduled Orchestration

Dagster definitions live in `quant_orchestrator.dagster_defs`:

```bash
dagster dev -m quant_orchestrator.dagster_defs
```

The current module exposes an example scheduled experiment job. Production research jobs should compose the same platform capabilities: load warehouse data, run the selected ML framework or backtesting framework only when needed, and register native artifacts.

## Optional Experiment Primitives

Walk-forward windows and Monte Carlo simulation are reusable primitives, not required workflow assumptions. Use them when the experiment needs them; train-only and backtest-only workflows can ignore them.

```python
from quant_orchestrator.experiments import WindowSpec, build_walk_forward_windows

windows = build_walk_forward_windows(
    WindowSpec(
        mode="rolling",
        train_start="2020-01-01",
        train_end="2020-12-31",
        test_start="2021-01-01",
        test_end="2021-12-31",
    )
)
```

```python
from quant_orchestrator.monte_carlo import simulate_return_paths

simulation = simulate_return_paths(returns, iterations=1000, horizon=252, block_size=5)
```
