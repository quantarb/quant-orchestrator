# Quant Orchestrator Notebook Examples

These examples show the current platform shape as small, composable workflows. They consume data, features, and labels from Quant Warehouse and then demonstrate how `quant-orchestrator` stitches ML frameworks, backtesting frameworks, reports, and artifacts together.

The comparison notebooks are meant to show how sensitive strategies can be to data vendor and backtesting framework choices, while also making it easy to start from existing native examples in popular frameworks. They are not a recommendation to test every vendor/framework combination by default. Use them to understand sensitivity, reuse proven examples, narrow the candidate stack, and decide what is worth validating with real or paper PnL.

They are examples, not required paths:

- `multi_backtest_frameworks/sample_strategy_comparsion.ipynb` calls the shared framework-comparison helper to compare the same SMA crossover strategy across `backtesting.py`, Zipline Reloaded, and NautilusTrader on `yfinance` and `fmp` data, then decomposes whether vendor or framework differences dominate.
- `multi_backtest_frameworks/sample_strategy_validation.ipynb` demonstrates provider-specific SMA parameter optimization with `backtesting.py`, then independently forward-tests the selected parameters on Zipline Reloaded and NautilusTrader.
- `mult-ml-frameworks/sample_model_training.ipynb` demonstrates CUDA-first toy model training across MAG7, `yfinance`, and `fmp` using Quant Warehouse adjusted OHLCV features and optimal-trading labels: RAPIDS cuML RandomForest for trade-side classification, PyTorch autoencoder, and FlairNLP's native multitask model for trade-side classification plus return-percentile regression with a tiny pretrained transformer.
- `ml_trading/ml_filtered_sma_trading.ipynb` trains a pre-2020 CUDA cuML optimal-side classifier, injects fixed 2020+ ML predictions into `backtesting.py`, runs yearly anchored WFO over SMA variants, portfolio-optimizes profitable variants, and runs Monte Carlo on out-of-sample trade contributions.

The notebooks should stay focused on orchestration patterns. They should not become the place where reusable platform code lives, and the notebook directory should contain notebook files only.

Current reusable code placement:

- Framework-specific data adapters live under `quant_orchestrator/platforms/backtesting_frameworks/<framework>/data_adapter.py`.
- Framework-specific reporting adapters live under `quant_orchestrator/platforms/backtesting_frameworks/<framework>/reporting_adapter.py`.
- Framework-specific reusable signal runners live under `quant_orchestrator/platforms/backtesting_frameworks/<framework>/runner.py` when they exist. Current runners exist for Zipline Reloaded and NautilusTrader.
- Framework-specific SMA crossover examples now live under `quant_orchestrator/platforms/backtesting_frameworks/<framework>/sma_crossover.py`.
- FlairNLP helper functions used by the current multi-ML notebook live under `quant_orchestrator/platforms/ml_frameworks/flair/shared.py`.

Notebook-only experiment glue should stay in the notebook until the same pattern is reused enough to justify package code.
