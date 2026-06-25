# Quant Orchestrator Provider Template

This template mirrors the OpenBB extension/provider idea in a smaller form.

Provider packages expose one or more entry points:

- `quant_orchestrator.ml_framework`
- `quant_orchestrator.backtest_engine`
- `quant_orchestrator.broker`

Each entry point should return a `ProviderManifest`.

## Example `pyproject.toml`

```toml
[project.entry-points."quant_orchestrator.backtest_engine"]
my_engine = "my_quant_provider.backtest_engine:provider"
```

## Shape

```text
my-quant-provider/
  pyproject.toml
  my_quant_provider/
    __init__.py
    backtest_engine.py
    broker.py
    ml_framework.py
```
