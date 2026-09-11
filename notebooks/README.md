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
- Reusable classifier-family scores are produced by `research_tools.family_score_pipeline`. Its streaming runner trains, scores, persists, and releases one raw feature-family model at a time. Strategies can use `FamilyScoreStore.read_scores` to select model/date subsets and `build_score_ensemble` to create a mean ensemble without retraining or loading model objects.
- Context dimensions from `quant_warehouse.research_tools.build_security_context_panel` stay separate from model features. Use `attribute_model_scores` for score quality by year/sector/industry/regime and `attribute_strategy_returns` for additive gross, cost, and net contribution by the same dimensions.

Notebook-only experiment glue should stay in the notebook until the same pattern is reused enough to justify package code.

The multi-rate issuer/instrument trainer uses bounded Polars windows and shared
within-step issuer encoding. Oracle/HITS are supervised labels only and never
historical model inputs. Causal elapsed-time and information-age embeddings
represent irregular gaps; NTP evaluation compares each family with persistence.
Sequence training (`--training-sequence-stride 128`) packs event dates into overlapping
Polars windows and supervises each event once. Issuer streams retain history at
the sequence start plus dated updates. The epoch monitor drains immutable epoch
checkpoints and prints separate annual backtests. `--epoch-evaluation-dir` makes
training wait for those reports before starting the next epoch;
`--resume-training` restores checkpoint weights, optimizer, and batch position.
Completed epochs can also be compared with an optional predicted-Oracle gate via
`research_tools.oracle_gate_comparison`, using the same frozen backtest inputs. Pass `mode="directional"` to require only
buy-versus-short agreement in addition to HITS; the strict threshold/exit-veto
comparison remains separately available.
Optional BF16 autocast is available for measured precision/batch-size comparisons.
Masked and next-observation objectives reconstruct individual values at token
and subtoken levels. Subtoken MTP masks features within each family; token MTP
masks whole families. NTP predicts coherent next family and combined-token observations.
See the [reconstruction contract](../docs/multirate-reconstruction.md) for the objective details, checkpoint
requirements, executed diagnostic scope, and longer-history 1T experiment.

`research_tools.multirate_source_inventory` stages broad warehouse histories
using bounded Polars reads and records missing sources, schemas, historical
coverage and ambiguous cadence. See `docs/multirate-expanded-coverage.md`; this
audit is preparation for expanded training, not a completed expanded model.


Expanded $1T smoke (September 11): `notebooks/multirate_expanded_smoke.ipynb` records the subset/build/launch recipe. The actual run is `artifacts/multirate_recovery/1T/train_expanded_smoke_v9`: one fresh epoch, 13 equity symbols / 11 issuers plus six existing option paths, 40 restored feature families and 1,379 individual numeric fields, 16 observed sparse families (no holder exits). It uses Polars streaming, the pre-2024 cutoff, and blocking original anchored-HITS backtests for 2024, 2025 and 2026 through September 9. Government and insider buy/sell heads now use exact transaction-date targets and actual opposite trades for negatives; historical event inputs retain disclosure availability. This is a pipeline smoke test, not proof of predictive performance or complete point-in-time vintage coverage. Several holder/ETF families have only post-cutoff observations. The older $100B epoch-four run is paused with its checkpoint preserved.


The expanded $100B run now uses `artifacts/multirate_recovery/100B/train_expanded_v9`, with 12 fresh epochs, batch size 32, all 40 restored numeric families and 17 sparse families. It follows the completed $1T smoke and runs 2024/2025/2026-through-September-9 anchored-HITS backtests after each epoch. The numeric corpus passed duplicate/nonfinite checks; 103 legacy congressional/insider issuer sources and Walmart dividends were refreshed with original snapshots preserved. See `notebooks/multirate_expanded_smoke.ipynb` for artifact locations.


Multi-rate loading now uses bounded Polars-to-Torch issuer indexes (`research_tools/streaming_context.py`) with read-only disk mappings and a bounded calendar-window fallback. The trainer keys caches by source fingerprints, normalization, feature layout and optional option inputs, preventing cross-fit reuse. No learned encoder outputs are cached across optimizer steps. The controlled expanded-$100B benchmark in `notebooks/multirate_indexed_loading_benchmark.ipynb` measured 72.99 → 30.19 seconds/batch including first-use index builds, and 3.52 seconds with reused indexes (2.42x / 20.76x); all three losses matched exactly. Twelve real-data window comparisons also matched exactly; peak indexing process RSS was 14.85 GiB. These are three-batch measurements, not full-epoch estimates. The existing $100B run resumed from epoch-one batch 50, retaining model/optimizer/RNG state and the same yearly epoch backtests.

Multi-rate epoch evaluation now groups consecutive dates by symbol, reuses unchanged annual/quarterly contexts, skips disabled document and unused supervised-label work, and batches NTP audit transfers with bounded duplicate tracking. Successor targets use an exact vectorized index calculation; training releases unused CUDA reservations before each blocking backtest. The expanded-$100B benchmark measured 43.46 → 15.62 seconds for 384 mixed-universe observations (2.78x) and 26.91 → 11.81 seconds for consecutive-date panels (2.28x). All 26 supervised heads agree within 1.2e-6, tested HITS rankings match, and NTP pair counts/persistence baselines match exactly. These are scoring-loop measurements, not full-backtest timings. The full evaluator scores 96,181 windows versus 14,849 spaced training windows per epoch; it records inference and portfolio durations separately. The rolling run completed epoch-one evaluation and was superseded by the fresh document run described below. See `notebooks/multirate_evaluation_speed_benchmark.ipynb` for the reproducible comparison.

The active multi-rate $100B run is now `train_documents_v10`, trained from fresh weights with calendar-quarter documents and stable as-of prefixes. Full 2024–2026 scoring preserves all 96,181 dates while reducing 96,181 rolling windows to 1,292 documents: measured inference was 66.35 seconds versus 2,435.86 seconds (36.71x), with another 2.84 seconds for the six yearly books. This was a three-step smoke-checkpoint timing test, not a learned-return comparison. Input layouts and all 42 tasks match; real-data prefix predictions agree within 8.35e-7 and 86 targeted tests pass. Training remains pre-2024, uses Polars streaming and batch 16, and runs the original adjusted-price anchored-HITS backtests after every epoch. See `docs/multirate-document-scoring.md` and `notebooks/multirate_document_scoring.ipynb` for the contract, coverage, ticker-price mapping and reproducible evidence.
