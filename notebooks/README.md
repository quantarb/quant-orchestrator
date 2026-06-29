# Quant Orchestrator Notebook Examples

These examples show the intended platform shape as small, composable workflows. They consume data, features, and labels from Quant Warehouse and then demonstrate how `quant-orchestrator` stitches ML frameworks, backtesting frameworks, reports, and artifacts together.

The comparison notebooks are meant to show how sensitive a strategy can be to data vendor and backtesting framework choices. They are not a recommendation to test every vendor/framework combination by default. Use them to understand sensitivity, narrow the candidate stack, and decide what is worth validating with real or paper PnL.

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
- `mult-ml-frameworks/sample_model_training.ipynb` demonstrates CUDA-first toy model training across MAG7, `yfinance`, and `fmp` using Quant Warehouse adjusted OHLCV features and optimal-trading labels: RAPIDS cuML RandomForest, PyTorch autoencoder, and FlairNLP multitask text classification/regression with a tiny pretrained transformer.
- `multi_backtest_frameworks/sample_strategy_validation.ipynb` demonstrates provider-specific SMA parameter optimization with `backtesting.py`, then independently forward-tests the selected parameters on Zipline Reloaded and NautilusTrader.
- `tutorial_backtesting_py.ipynb` is a framework-specific tutorial for `backtesting.py` covering multi-vendor single-symbol backtests, multi-symbol backtests, Monte Carlo, walk-forward optimization, equity-curve analysis, and portfolio optimization.
- `tutorial_zipline.ipynb` is the same tutorial shape for Zipline Reloaded.
- `tutorial_nautilus.ipynb` is the same tutorial shape for NautilusTrader.

The notebooks should stay focused on orchestration patterns. They should not become the place where reusable platform code lives, and the notebook directory should contain notebook files only.

Current reusable code placement:

- Framework-specific data adapters live under `quant_orchestrator/platforms/backtesting_frameworks/<framework>/data_adapter.py`.
- Framework-specific reporting adapters live under `quant_orchestrator/platforms/backtesting_frameworks/<framework>/reporting_adapter.py`.
- Framework-specific SMA crossover examples now live under `quant_orchestrator/platforms/backtesting_frameworks/<framework>/sma_crossover.py`.
- Shared FlairNLP classification/regression multitask helpers live under `quant_orchestrator/platforms/ml_frameworks/flair/shared.py`.

Notebook-only experiment glue should stay in the notebook until the same pattern is reused enough to justify package code.
