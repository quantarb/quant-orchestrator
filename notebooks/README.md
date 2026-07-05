# Quant Orchestrator Notebook Examples

These examples show the current platform shape as small, composable workflows. They consume data, features, and labels from Quant Warehouse and then demonstrate how `quant-orchestrator` stitches ML frameworks, backtesting frameworks, reports, and artifacts together.

The comparison notebooks are meant to show how sensitive strategies can be to data vendor and backtesting framework choices, while also making it easy to start from existing native examples in popular frameworks. They are not a recommendation to test every vendor/framework combination by default. Use them to understand sensitivity, reuse proven examples, narrow the candidate stack, and decide what is worth validating with real or paper PnL.

They are examples, not required paths:

- `multi_backtest_frameworks/sample_strategy_comparsion.ipynb` calls the shared framework-comparison helper to compare the same SMA crossover strategy across `backtesting.py`, Zipline Reloaded, and NautilusTrader on `yfinance` and `fmp` data, then decomposes whether vendor or framework differences dominate.
- `multi_backtest_frameworks/sample_strategy_validation.ipynb` demonstrates provider-specific SMA parameter optimization with `backtesting.py`, then independently forward-tests the selected parameters on Zipline Reloaded and NautilusTrader.
- `mult-ml-frameworks/sample_model_training.ipynb` demonstrates CUDA-first toy model training across MAG7, `yfinance`, and `fmp` using Quant Warehouse adjusted OHLCV features and optimal-trading labels: RAPIDS cuML RandomForest for trade-side classification, PyTorch autoencoder, and FlairNLP's native multitask model for trade-side classification plus return-percentile regression with a tiny pretrained transformer.
- `ml_trading/ml_filtered_sma_trading.ipynb` trains a pre-2020 CUDA cuML optimal-side classifier, injects fixed 2020+ ML predictions into `backtesting.py`, runs yearly anchored WFO over SMA variants, portfolio-optimizes profitable variants, and runs Monte Carlo on out-of-sample trade contributions.
- `ml_trading/optimal_trader_trading_app_contract_replay.ipynb` replays saved optimal_trader trading-app artifacts historically without importing live-trading code, then writes the standard strategy artifact contract.
- `ml_trading/optimal_trader_moe_paper_contract_replay.ipynb` replays saved MoE paper-strategy artifacts or a historical MoE feature/scored panel and writes the same contract.
- `ml_trading/classifier_1t_options_signal_backtest.ipynb`, `ml_trading/classifier_1t_feature_family_option_windows.ipynb`, and `ml_trading/traditional_ml_synthetic_options_backtest.ipynb` are contract producers. They convert scored panels into `action_tape` and the canonical `trade_list`; downstream option replay should consume those artifacts instead of embedding option mechanics in the notebook.

The notebooks should stay focused on orchestration patterns. They should not become the place where reusable platform code lives, and the notebook directory should contain notebook files only.

The durable boundary is the produced artifact, not the ingredients used to create it. A strategy may come from a native backtesting framework, a notebook, saved optimal_trader models, or an external research artifact. Quant Orchestrator should not require those producers to share one input pipeline. It should require the reusable outputs to follow stable contracts:

- `trade_list`: the canonical list of closed equity trades consumed by Monte Carlo, walk-forward summaries, equity-curve analysis, option-equivalent replay, and trade mixing/ensembling. It is written as `trade_list.parquet` and exposed in manifests as `trade_list`.
- `scored_panel`: an optional upstream artifact for strategies that naturally produce full-universe daily scores before translating them into trades.
- `action_tape`: an optional execution-intent artifact for strategies that need entry/exit auditability before they become closed trades.
- ML notebooks should standardize produced model outputs and prediction tables only when they are reused downstream. They should not be forced into the strategy artifact contract until they emit a tradable `scored_panel` or `trade_list`.

Current reusable code placement:

- Framework-specific data adapters live under `quant_orchestrator/platforms/backtesting_frameworks/<framework>/data_adapter.py`.
- Framework-specific reporting adapters live under `quant_orchestrator/platforms/backtesting_frameworks/<framework>/reporting_adapter.py`.
- Framework-specific reusable signal runners live under `quant_orchestrator/platforms/backtesting_frameworks/<framework>/runner.py` when they exist. Current runners exist for Zipline Reloaded and NautilusTrader.
- Standard produced-artifact helpers live in `quant_orchestrator/artifact_contracts.py`. Downstream consumers should prefer `read_trade_list_artifact`, `write_trade_list_artifact`, `normalize_trade_list`, and `combine_trade_lists` when they only need closed trades.
- Generic scored-panel top-k replay lives in `quant_orchestrator/platforms/backtesting_frameworks/scored_panel_replay.py`.
- optimal_trader historical replay helpers live under `quant_orchestrator/platforms/backtesting_frameworks/optimal_trader/`; live trading and broker code should stay in optimal_trader, not here.
- Strategy-specific SMA crossover examples live in notebook-facing helpers under `quant_orchestrator/backtests/` until they prove a more durable home.
- Strategy-specific backtesting.py ML-score helpers live under `quant_orchestrator/backtests/`.
- Synthetic and real-quote option research helpers still exist under `quant_orchestrator/research_tools/`, but new option-equivalent backtests should start from standard `trade_list` artifacts. The optimal_trader vectorized engine and option-return primitives live under `quant_orchestrator/platforms/backtesting_frameworks/optimal_trader/`.
- FlairNLP helper functions used by the current multi-ML notebook live under `quant_orchestrator/platforms/ml_frameworks/flair/shared.py`.

Notebook-only experiment glue should stay in the notebook until the same pattern is reused enough to justify package code.
