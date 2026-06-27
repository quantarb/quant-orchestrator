# Repository Rules

## Dependency Source Of Truth

- Use `quant-warehouse` from `https://github.com/quantarb/quant-warehouse.git@main`.
- Do not commit local editable `quant-warehouse` dependency paths.

## Data Boundary

- `quant-orchestrator` should consume data through `quant-warehouse`; it should not call OpenBB or vendor market-data APIs directly.
- If warehouse data is missing or incomplete, fix the OpenBB fork provider first, then refresh `quant-warehouse`.

## Orchestrator Responsibility

- `quant-orchestrator` is responsible for research workflows: training ML models across different ML frameworks and backtesting those models across different backtesting frameworks.
- Do not add live trading, broker order submission, or broker account mutation code here.
- If a workflow needs market data, features, labels, or warehouse refreshes, call `quant-warehouse` rather than OpenBB or vendor APIs directly.
