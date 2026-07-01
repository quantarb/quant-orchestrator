# Quant Orchestrator

Composable Dagster and MLflow research orchestration around data, features, and labels stored in `quant-warehouse`.

`quant-orchestrator` coordinates research workflows without assuming a single shape. A run can be ML training only, backtesting only, training followed by backtesting, cross-framework validation, Monte Carlo analysis, portfolio construction, or an external-engine strategy run that uses warehouse data and stores the native outputs.

## Motivation

Quant research is fragile because the same trading ideas can behave differently depending on the data vendor, feature and label pipeline, and backtesting framework used to test them. Price adjustments, missing rows, corporate-action handling, trading calendars, order timing, fill simulation, fees, slippage, and framework-specific accounting can all change measured performance. `quant-orchestrator` exists to help find the right research and validation stack for your quant trading strategies, not to crown one universally best data vendor or backtesting framework.

The framework is also meant to help you move faster by reusing existing work from mature backtesting ecosystems. Popular frameworks already have documented examples, community strategies, and native implementation patterns. Starting from those implementations is usually better than rewriting every strategy from scratch, especially when you are validating an idea quickly. Those native implementations can also serve as grounding references when the same strategy is ported to another framework for comparison or additional realism.

The point is also not to test every possible vendor/framework combination just because the platform can. More vendors and engines add data cost, compute cost, code complexity, and live-trading operational risk. The useful workflow is to test enough combinations to understand where your strategies are sensitive, narrow the candidate stacks, and then validate the small number that matter with paper or real PnL. For example, you might run the same strategy in two separate live accounts using two different data vendors, track realized performance over time, and decide from evidence whether keeping both stacks is worth the added complexity.

## Environment

```bash
conda env create -f environment.yml
conda activate quant-orchestrator
```

The environment installs `quant-warehouse` from GitHub, Dagster, MLflow, plus the research/backtesting dependencies used by the notebooks.

## Install

For a package install:

```bash
pip install "git+https://github.com/quantarb/quant-orchestrator.git"
```

For the full local research/backtesting/ML environment:

```bash
pip install -e ".[all,dev]"
```

For just ThetaData-backed options backtests through Optopsy:

```bash
pip install -e ".[options]"
```

For CUDA-first ML workflows, install the CUDA extra. The current CUDA stack is aimed at PyTorch CUDA plus RAPIDS cuML/CuPy CUDA 13:

```bash
pip install -e ".[cuda]"
```

If PyTorch needs a host-specific wheel, install the wheel that matches the local driver before running the notebooks.

## Platform Capabilities

The platform contract is intentionally small:

- compose in-memory research pipelines from optional stages
- resolve prepared datasets from `quant-warehouse`
- run ML framework providers when a workflow needs model training or inference
- run backtesting framework providers, data adapters, reporting adapters, and reusable runners when a workflow needs strategy evaluation
- run reusable optimization primitives such as parameter grids, metric filters, result ranking, and portfolio weighting
- normalize common backtest summaries, equity curves, returns, and trade logs while keeping each framework's native report
- track runs in MLflow
- store native model, report, prediction, backtest, and strategy artifacts in the artifact registry

These capabilities are independent. A workflow does not need an ML model to run a backtest, and it does not need a backtest to train a model.

The pipeline core is deliberately lightweight. `PipelineContext` is an in-memory artifact store; each `FunctionStage` declares required inputs and produced outputs; `Pipeline` validates those contracts and runs stages sequentially. This is not a replacement for Dagster. Use Dagster for scheduled asset orchestration, dependency management, and production job scheduling. Use the pipeline layer inside research workflows when you need explicit artifact handoffs between data loading, model training, inference, backtesting, filtering, ranking, portfolio construction, Monte Carlo, and reporting.

Optimization helpers are reusable mechanics, not fixed workflows. `quant_orchestrator.optimization` includes parameter-grid construction, metric filtering, result ranking, returns-matrix construction, long-only Sharpe weighting, and random-search mean-variance weighting. Notebooks still choose the strategy, thresholds, train/test windows, framework handoff, and portfolio constraints.

```python
from quant_orchestrator.pipeline import FunctionStage, Pipeline, PipelineContext

pipeline = Pipeline(
    [
        FunctionStage("load_predictions", load_predictions, produced_outputs=("predictions",)),
        FunctionStage(
            "run_backtest",
            run_backtest,
            required_inputs=("predictions",),
            produced_outputs=("backtest_report",),
        ),
    ],
)
result = pipeline.run(PipelineContext())
report = result.context.require("backtest_report")
```

`quant-orchestrator` follows an OpenBB-style provider layout for model and backtesting extension categories:

- `quant_orchestrator.platforms.ml_frameworks`
- `quant_orchestrator.platforms.backtesting_frameworks`

Current built-in ML framework modules are:

- `sklearn`
- `torch`
- `transformers`

Current built-in backtesting framework modules are:

- `backtesting.py`
- Zipline Reloaded
- NautilusTrader
- Optopsy for options research paths

Installed packages can register providers through entry points:

```toml
[project.entry-points."quant_orchestrator.ml_framework"]
my_framework = "my_package.ml_framework:provider"

[project.entry-points."quant_orchestrator.backtesting_framework"]
my_engine = "my_package.backtesting_framework:provider"
```

At runtime, providers are resolved from the registry:

```python
from quant_orchestrator.platforms.registry import registry

registry.list("backtesting_framework")
engine_cls = registry.adapter("backtesting_framework", "optopsy")
engine = engine_cls()
```

Backtesting framework providers are adapters around native engines, not strategy assumptions. Data adapters should move prepared Quant Warehouse frames into each engine in memory whenever the engine allows it. Reporting adapters should expose comparable summaries, equity curves, returns, and trade logs while preserving native reports and metrics.

Current reusable backtesting code is intentionally concrete:

- `backtesting_py/data_adapter.py` converts warehouse OHLCV plus precomputed features into `backtesting.py`'s expected dataframe shape.
- `zipline/data_adapter.py` builds an in-memory Zipline daily bar reader.
- `nautilus/data_adapter.py` converts warehouse OHLCV into Nautilus bar objects.
- `zipline/runner.py` and `nautilus/runner.py` run a precomputed long/flat signal strategy through the native engines.
- `<framework>/reporting_adapter.py` normalizes common backtest outputs while keeping each framework's native report.
- `<framework>/sma_crossover.py` contains the reusable SMA example strategy wrappers used by the CLI and notebooks.

A QuantConnect strategy should be exposed through a backtesting framework provider or runner that receives prepared `quant-warehouse` data, runs the native QuantConnect strategy, and registers whatever files or reports that engine emits. QuantConnect support is not currently implemented in this repo.

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

The registry is intentionally schema-light: sklearn, PyTorch, Flair NLP, Zipline Reloaded, NautilusTrader, `backtesting.py`, and other frameworks can save their native files, directories, dataframes, JSON, text reports, or pickled objects without forcing every output into one common report shape.

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

Use `notebooks/` for one-off research workflows and demonstrations that consume prepared datasets from `quant-warehouse`. If a workflow becomes a repeated platform capability, move only the reusable part into package code and keep the notebook as an example. Do not build feature families, labels, warehouse refreshes, or vendor data pulls in this repo; implement those in `quant-warehouse` first and consume the resulting dataset here.

FMP event-pair labels consumed from `quant-warehouse` are exact event-date labels only. Notebooks must not create future-window event-pair tasks for congress, insider, analyst, or earnings labels. Event-model datasets must start from actual event rows, then inner join feature-family rows on `(symbol, date)`. A feature family must not appear in an event-training dataset unless that exact event row exists and that feature family has coverage for the event. Mirrored event-pair tasks should use only actual event dates, e.g. congress buy vs congress sell, insider buy vs insider sell, analyst upgrade vs analyst downgrade, price target raise vs price target cut, and earnings beat vs earnings miss. Future return horizons and oracle-trade labels are separate target families. Company guidance raise/cut labels are not currently supported because the warehouse does not store true company-issued guidance revision history.

FMP oracle-trade side tasks follow the same event-only rule. Train one oracle buy/sell task across all configured `k` values; do not create separate oracle tasks or models per `k`. Use oracle buy entry dates versus oracle sell entry dates only. Do not use non-entry dates as negative examples, and do not train binary tasks from the oracle `any` union target.

During refactors, the notebooks are the integration tests. Internal APIs can change aggressively when the architecture improves, but the notebook research intent should keep working after the notebooks are updated and re-executed.

Recent notebooks follow this pattern:

- data vendors, adjusted OHLCV features, and target-engineered labels come from Quant Warehouse
- notebooks keep major datasets, predictions, reports, and summaries in `PipelineContext`
- strategy examples may live in package code when reused across frameworks, but notebook-only experiment glue stays in notebooks
- framework-specific data adapters live under `quant_orchestrator/platforms/backtesting_frameworks/<framework>/data_adapter.py`
- framework-specific reporting adapters live under `quant_orchestrator/platforms/backtesting_frameworks/<framework>/reporting_adapter.py`
- framework-specific reusable signal runners live under `quant_orchestrator/platforms/backtesting_frameworks/<framework>/runner.py` when they exist

## Load Data

The examples read prices already stored in Quant Warehouse:

```bash
quant-orchestrator --strategy sma --framework all --symbols AAPL --provider yfinance --start 2020-01-01
```

If the warehouse has no prices for a symbol, refresh it from the Quant Warehouse repo/env first:

```bash
conda activate quant-warehouse
quant-warehouse refresh AAPL --sections prices --providers yfinance
```

## Built-In Examples

```bash
quant-orchestrator --strategy sma --framework all --symbols AAPL --start 2023-01-01 --fast-window 5 --slow-window 10
quant-orchestrator --strategy sma --framework zipline --symbols AAPL --start 2020-01-01
quant-orchestrator --strategy sma --framework nautilus --symbols AAPL --start 2020-01-01
```

The CLI is a smoke-test/demo surface. The SMA CLI currently runs Zipline Reloaded and NautilusTrader for the first symbol passed to `--symbols`. The richer framework-comparison notebooks cover `backtesting.py`, Zipline Reloaded, and NautilusTrader across multiple symbols and providers.

Zipline Reloaded uses `TradingAlgorithm` with in-memory daily bars. NautilusTrader uses `BacktestEngine` and `BarDataWrangler` to convert the same OHLCV frame into Nautilus bar objects. The reusable Zipline and Nautilus signal runners execute a precomputed long/flat signal column, so notebooks can inject warehouse features or ML predictions without duplicating engine ceremony. The examples use a simple SMA strategy so framework runtime and output shape can be compared without implying that every workflow is equity-only or ML-driven.

## Scheduled Orchestration

Dagster definitions live in `quant_orchestrator.dagster_defs`:

```bash
dagster dev -m quant_orchestrator.dagster_defs
```

The current module exposes a backtest-framework-comparison job. Production research jobs should compose the same platform capabilities: load warehouse data, run the selected ML framework or backtesting framework only when needed, and register native artifacts.

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
