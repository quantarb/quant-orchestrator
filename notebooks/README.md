# Quant Orchestrator Notebook Examples

These examples show the intended platform shape as small, composable workflows.

They are examples, not required paths:

- `workflow_train_only.ipynb` demonstrates an ML-only job that trains and registers artifacts.
- `workflow_backtest_only.ipynb` demonstrates a backtest-only job that consumes prepared data and emits native backtest output.
- `workflow_chain_example.ipynb` demonstrates train -> predict -> strategy -> Monte Carlo composition.
- `multi_backtest_frameworks/sample_strategy_comparsion.ipynb` calls the shared framework-comparison helper to compare the same SMA crossover strategy across `backtesting.py`, Zipline Reloaded, and NautilusTrader on `yfinance` and `fmp` data, then decomposes whether vendor or framework differences dominate.
- `mag7_sma_crossover_comparison.ipynb` demonstrates a shared in-memory MAG7 SMA crossover run across `backtesting.py`, Zipline Reloaded, and NautilusTrader using Quant Warehouse features.
- `mag7_sma_crossover_monte_carlo.ipynb` demonstrates the same MAG7 SMA crossover run plus a second Monte Carlo robustness job on the resulting equity curves.
- `multi_vendor_backtesting_py_sma_crossover.ipynb` calls the `quant-orchestrator` backtest job that compares `fmp` and `yfinance` with a fixed `backtesting.py` SMA crossover.
- `cross_framework_sma_search_monte_carlo.ipynb` calls the `quant-orchestrator` search job that runs vectorbt parameter search, Monte Carlo, and forward testing across `fmp` and `yfinance`.
- `wfo_mag7_sma_optimization.ipynb` demonstrates a fixed-window walk-forward optimization over SMA parameters with a 2020-2025 train window and 2026 test window.
- `multi_backtest_frameworks/sample_strategy_validation.ipynb` demonstrates provider-specific SMA parameter optimization with `backtesting.py`, then independently forward-tests the selected parameters on Zipline Reloaded and NautilusTrader.
- `tutorial_backtesting_py.ipynb` is a framework-specific tutorial for `backtesting.py` covering multi-vendor single-symbol backtests, multi-symbol backtests, Monte Carlo, walk-forward optimization, equity-curve analysis, and portfolio optimization.
- `tutorial_zipline.ipynb` is the same tutorial shape for Zipline Reloaded.
- `tutorial_nautilus.ipynb` is the same tutorial shape for NautilusTrader.

The notebooks should stay focused on orchestration patterns. They should not become the place where reusable platform code lives.
Framework-specific SMA crossover examples now live under `quant_orchestrator/platforms/backtesting_frameworks/*/sma_crossover.py`.
