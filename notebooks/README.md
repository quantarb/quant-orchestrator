# Quant Orchestrator Notebook Examples

These examples show the intended platform shape as small, composable workflows.

They are examples, not required paths:

- `workflow_train_only.py` demonstrates an ML-only job that trains and registers artifacts.
- `workflow_backtest_only.py` demonstrates a backtest-only job that consumes prepared data and emits native backtest output.
- `workflow_chain_example.py` demonstrates train -> predict -> strategy -> Monte Carlo composition.

The notebooks should stay focused on orchestration patterns. They should not become the place where reusable platform code lives.
