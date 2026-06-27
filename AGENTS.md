# Repository Rules

## Dependency Source Of Truth

- Use `quant-warehouse` from `https://github.com/quantarb/quant-warehouse.git@main`.
- Do not commit local editable `quant-warehouse` dependency paths.

## Data Boundary

- `quant-orchestrator` should consume data through `quant-warehouse`; it should not call OpenBB or vendor market-data APIs directly.
- If warehouse data is missing or incomplete, fix the OpenBB fork provider first, then refresh `quant-warehouse`.

## Orchestrator Responsibility

- `quant-orchestrator` is responsible for research workflows: training ML models and backtesting those models.
- The repo is opinionated toward Dagster for orchestration and MLflow for experiment tracking.
- Do not add live trading, broker order submission, or broker account mutation code here.
- If a workflow needs market data, features, labels, or warehouse refreshes, call `quant-warehouse` rather than OpenBB or vendor APIs directly.

## Compatibility Policy

- This repo is new and rapidly changing. Do not add backward-compatibility wrappers, legacy aliases, or duplicate old APIs.
- When package structure changes, update imports and notebooks directly.
- Do not preserve old provider categories, old entry-point groups, or old module paths.

## Platform Policy

- Keep provider-style extension points only for `platforms/ml_frameworks` and `platforms/backtesting_frameworks`.
- Do not add provider abstractions for orchestration or experiment tracking. Use Dagster and MLflow directly through the repo's opinionated interfaces.
- Do not add broker platforms or live-trading adapters.

## Notebook Policy

- Use `notebooks/` for one-off research workflows: model training experiments, backtest experiments, walk-forward experiments, Monte Carlo experiments, and equity-curve analysis.
- Notebooks in this repo must not implement feature engineering, target engineering, or warehouse refresh logic. Pull prepared datasets from `quant-warehouse`.
- If a notebook needs a new feature family or label, implement it in `quant-warehouse` first, then consume it here.

## CUDA Policy

- Optimize model training and simulation code for CUDA-first execution when the selected ML framework supports it.
- Prefer GPU-native libraries such as PyTorch CUDA and RAPIDS/CuPy where they fit the workflow.
- Do not keep slow CPU compatibility paths unless they are the only practical path for a required dependency.
