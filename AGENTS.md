# Repository Rules

## Dependency Source Of Truth

- Use `quant-warehouse` from `https://github.com/quantarb/quant-warehouse.git@main`.
- Do not commit local editable `quant-warehouse` dependency paths.

## Data Boundary

- `quant-orchestrator` should consume data through `quant-warehouse`; it should not call OpenBB or vendor market-data APIs directly.
- If warehouse data is missing or incomplete, fix the OpenBB fork provider first, then refresh `quant-warehouse`.

## Orchestrator Responsibility

- `quant-orchestrator` is responsible for composable research workflows: ML training, inference, backtesting, external strategy evaluation, artifact storage, and orchestration.
- `quant-orchestrator` owns CRUD/storage for ML training runs, backtest runs, ML artifacts, backtest artifacts, and strategy artifacts. Downstream apps should request work here and load returned artifact URIs/paths instead of keeping their own research storage.
- The repo is opinionated toward Dagster for orchestration and MLflow for experiment tracking.
- Do not add live trading, broker order submission, or broker account mutation code here.
- If a workflow needs market data, features, labels, or warehouse refreshes, call `quant-warehouse` rather than OpenBB or vendor APIs directly.
- Artifact storage should be schema-light. Different ML frameworks and backtesting frameworks may emit incompatible reports, models, plots, directories, or binary objects; store the native outputs with minimal metadata rather than forcing one universal report shape.
- Do not assume every workflow is ML-driven, equity-only, or backtest-driven. Train-only, inference-only, backtest-only, train-then-backtest, and external-engine strategy runs should all fit the platform model.

## Compatibility Policy

- This repo is new and rapidly changing. Do not add backward-compatibility wrappers, legacy aliases, or duplicate old APIs.
- When package structure changes, update imports and notebooks directly.
- Do not preserve old provider categories, old entry-point groups, or old module paths.

## Build Vs Buy Policy

- Prefer widely used, actively maintained third-party packages or small forks of proven projects over custom implementations.
- For ML frameworks, backtesting frameworks, experiment tracking, orchestration, model serialization, metrics, reports, and simulations, use battle-tested libraries when they fit the repo boundary.
- Build from scratch only when no reliable package fits the requirement or this repo needs a thin opinionated wrapper around a proven dependency; document that reason in the change.

## Platform Policy

- Keep provider-style extension points only for `platforms/ml_frameworks` and `platforms/backtesting_frameworks`.
- Do not add provider abstractions for orchestration or experiment tracking. Use Dagster and MLflow directly through the repo's opinionated interfaces.
- Do not add broker platforms or live-trading adapters.
- Backtesting framework providers should be thin adapters around native engines and caller-provided strategies/runners. For engines such as QuantConnect, keep the native strategy implementation intact and adapt warehouse inputs plus artifact outputs around it.

## Notebook Policy

- Use `notebooks/` for one-off research workflows: model training experiments, backtest experiments, walk-forward experiments, Monte Carlo experiments, and equity-curve analysis.
- If a notebook workflow becomes a repeated capability, move the reusable API into package code and leave the notebook as an example.
- Notebooks in this repo must not implement feature engineering, target engineering, or warehouse refresh logic. Pull prepared datasets from `quant-warehouse`.
- If a notebook needs a new feature family or label, implement it in `quant-warehouse` first, then consume it here.

## CUDA Policy

- Optimize model training and simulation code for CUDA-first execution when the selected ML framework supports it.
- Prefer GPU-native libraries such as PyTorch CUDA and RAPIDS/CuPy where they fit the workflow.
- Do not keep slow CPU compatibility paths unless they are the only practical path for a required dependency.
