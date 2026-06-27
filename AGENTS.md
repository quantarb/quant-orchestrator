# Repository Rules

## Dependency Source Of Truth

- Use `quant-warehouse` from `https://github.com/quantarb/quant-warehouse.git@main`.
- Do not commit local editable `quant-warehouse` dependency paths.

## Data Boundary

- `quant-orchestrator` should consume data through `quant-warehouse`; it should not call OpenBB or vendor market-data APIs directly.
- If warehouse data is missing or incomplete, fix the OpenBB fork provider first, then refresh `quant-warehouse`.
