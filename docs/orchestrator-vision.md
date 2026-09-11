# Quant Orchestrator Vision

This note captures the current direction so future work can resume without reconstructing the design from scratch.

## Core Goal

`quant-orchestrator` should be a composable research orchestration layer, not a fixed ML-plus-backtest workflow.

Quant research is fragile because the same trading ideas can behave differently depending on the data vendor, feature and label pipeline, and backtesting framework used to test them. Price adjustments, missing rows, corporate-action handling, trading calendars, order timing, fill simulation, fees, slippage, and framework-specific accounting can all change measured performance. `quant-orchestrator` should help find the right research and validation stack for your quant trading strategies, not crown one universally best data vendor or backtesting framework.

The platform should also help reuse existing work from mature backtesting ecosystems. Popular frameworks already have documented examples, community strategies, and native implementation patterns. Starting from those implementations is usually better than rewriting every strategy from scratch, especially when validating an idea quickly. Those native implementations can also serve as grounding references when the same strategy is ported to another framework for comparison or additional realism.

The platform should also not encourage testing every possible vendor/framework combination just because it can. More vendors and engines add data cost, compute cost, code complexity, and live-trading operational risk. The intended end state is strategy-specific stack selection based on evidence: test enough combinations to understand sensitivity across your strategies, narrow the candidate stacks, then validate the small number that matter with paper or real PnL. A good workflow can compare candidate stacks in research, deploy a small number of them in separate live or paper accounts, and then decide from realized performance whether one stack is enough or whether maintaining multiple data/framework combinations is worth the complexity.

It should coordinate:

- ML training
- inference / prediction generation
- strategy evaluation
- external-engine strategy runs
- parameter search
- portfolio construction
- strategy artifact generation and replay
- Monte Carlo and equity-curve simulations
- artifact storage and retrieval
- normalized comparison views over native reports

## What The Platform Should Not Assume

- It should not assume every run is ML-driven.
- It should not assume every run is equity-only.
- It should not assume every run includes a backtest.
- It should not assume every workflow starts with model training.
- It should not assume the same backtesting framework is used for validation and execution.

## Design Principle

Jobs should be atomic and explicit about their inputs and outputs.

A job should declare:

- what data it can see
- what artifact it consumes
- what artifact it emits
- what time window or split it is allowed to read

That is how we prevent leakage and keep workflows flexible.

The first implementation of this idea is intentionally small:

- `PipelineContext` is the shared in-memory artifact store.
- `FunctionStage` wraps a Python callable with required input and produced output declarations.
- `Pipeline` validates stage contracts and executes stages in order.

This layer should stay lightweight. It is not a scheduler, a DAG engine, or a Dagster replacement. Dagster should still own scheduled jobs, ETL assets, dependency management, and data validation. Quant Orchestrator pipelines are for research-time composition and explicit artifact handoffs.

## Reusable Primitives

The intended platform primitives are:

- train a model
- generate predictions
- run a strategy
- build parameter grids
- filter and rank result tables
- optimize parameters through a supplied runner
- construct portfolio weights from strategy return streams
- simulate returns or trade sequences
- compare runs

Monte Carlo and walk-forward optimization are primitives or workflow patterns, not hardcoded assumptions about every run.

The package should own reusable mechanics such as grid construction, metric filtering, ranking, returns-matrix construction, and generic portfolio weighting. Notebooks should continue to own research choices such as the exact strategy, thresholds, universe, train/test dates, framework handoff, and analysis text until those choices repeat enough to become stable platform stages.

## Artifact Model

The artifact registry exists and should remain schema-light.

It should store native outputs from:

- sklearn
- PyTorch
- Flair or other NLP frameworks
- `backtesting.py`
- Zipline
- NautilusTrader
- JSON, CSV, text, directories, pickles, or bytes

Different frameworks should be allowed to produce different native reports or file layouts.

Common reporting should be additive, not destructive. Backtesting reporting adapters should expose comparable summaries, equity curves, returns, and trade logs where possible, while preserving each framework's unique native metrics and artifacts.

There is one deliberate exception to the schema-light rule: reusable strategy handoffs now use a small standard artifact contract. `quant_orchestrator.platforms.backtesting_frameworks.strategy_artifacts` reads and writes:

- `feature_panel.parquet`
- `scored_panel.parquet`
- `action_tape.parquet`
- `trade_windows.parquet`
- `summary.json`
- `strategy_artifacts_manifest.json`

This contract is not meant to replace native framework outputs. It is an additive bridge for downstream workflows that should not care whether the original strategy came from a notebook, a framework runner, optimal_trader artifacts, or a future external engine. `trade_windows.parquet` is the key handoff for option-equivalent replay, walk-forward summaries, Monte Carlo over trade outcomes, and cross-framework comparison.

## Backtesting Model

Backtesting adapters should stay thin. Current code separates three concerns:

- data adapters convert prepared warehouse data into a framework-native in-memory input
- runners own repeated framework ceremony for reusable signal-style workflows
- reporting adapters normalize common summaries while preserving native outputs

Data adapters should bridge Quant Warehouse frames into each native engine without duplicating datasets. Prefer in-memory adapters when the framework supports them, as the current Zipline Reloaded, NautilusTrader, and `backtesting.py` examples do.

Current Zipline Reloaded and NautilusTrader runners execute a precomputed long/flat signal column through the native engine. That is useful for SMA examples and ML-prediction filters, but it should not become the only strategy model. Future external-engine support should be able to run native strategy implementations with prepared warehouse inputs and native artifacts.

Example strategies can live in package code when they are reused across notebooks for framework comparison. Notebook-specific experiment orchestration should stay in notebooks until it becomes a repeated platform capability.

The current repeated strategy handoff is:

`scored_panel -> action_tape -> trade_windows -> strategy_artifacts_manifest.json`

`scored_panel_replay.py` implements a generic shifted top-k replay for daily score panels. optimal_trader replay modules use their own historical strategy logic but write the same standard contract. This should stay concrete until more strategy families prove that a richer abstraction is needed.

## Options Model

The preferred option-equivalent backtest path is trade-window based. Equity strategies are the decision makers: if an equity strategy buys a symbol on an entry date and exits on an exit date, the option workflow uses that same symbol/date window and the same equity capital budget to select and price the option equivalent.

The option workflow should:

- load `trade_windows` from the standard strategy artifact contract
- select option candidates from the full chain on the equity entry date
- price only selected option contracts forward to the equity exit date or option expiration
- apply expiration intrinsic value when an option expires before the equity trade exits
- not roll into a new option unless a strategy explicitly defines a roll rule
- log trade-level failures such as missing chain data, no eligible option, missing selected path, or expired worthless

This is intentionally simpler and more inspectable than relooping over every date/symbol/option candidate in a notebook. Each equity trade is an independent unit and can be parallelized. ThetaData and FMP synthetic option handling can differ internally because their input data differ, but both should consume the same equity trade-window contract.

## ML Framework Model

ML framework helpers should stay close to the native framework while removing repeated integration friction.

Current examples:

- RAPIDS cuML provides a CUDA-backed sklearn-style RandomForest path.
- PyTorch uses CUDA auto-detection for tensor models.
- FlairNLP can be used for native multitask learning. Current helper functions for the notebook live under `quant_orchestrator.platforms.ml_frameworks.flair.shared`.

Reusable raw classifier-family training uses a bounded lifecycle: load one feature family, train its classifier, materialize standardized score partitions, persist the native model, and release CPU/GPU memory before loading the next family. The score artifact—not a process-wide model dictionary—is the reusable handoff into ensembles, meta-models, and trading strategies. Autoencoder representations remain a separate optional experiment path and are not part of the raw classifier configuration.

Score publication requires immutable Quant Warehouse input-lineage manifests. Their combined fingerprint is stored in every score row and the family-score run manifest. Consumers can require an expected lineage fingerprint, and ensembles reject mixed or missing lineage rather than silently combining predictions produced from different datasets or recipes.

ML outputs should remain native unless there is a clear reason to normalize them. A common metrics table is useful for comparison, but the platform should still store framework-specific reports and artifacts.

## Intended Workflow Examples

- Train one model in one ML framework, then feed its predictions into a strategy.
- Train multiple models in multiple frameworks, then compare their downstream strategy outputs.
- Convert a scored panel into standard strategy artifacts, then run option-equivalent replay from the resulting trade windows.
- Replay saved optimal_trader artifacts historically without importing live-trading code, then compare the emitted trade windows to another framework.
- Future external-engine path: backtest a QuantConnect-style strategy on warehouse data, then optionally replay its equity curve in another engine. QuantConnect support is not currently implemented.
- Optimize parameters in a fast engine, then validate the chosen parameters in a slower or more realistic engine.
- Run Monte Carlo on a backtest result without treating Monte Carlo as a strategy.

## Current Repo State

Already present:

- `quant_orchestrator.artifacts.ArtifactStore`
- `quant_orchestrator.pipeline.PipelineContext`
- `quant_orchestrator.pipeline.FunctionStage`
- `quant_orchestrator.pipeline.Pipeline`
- `quant_orchestrator.optimization` primitives for grids, filters, ranking, returns matrices, and portfolio weights
- ML and backtesting provider contracts
- MLflow tracking helpers
- Dagster entry points
- walk-forward window utilities
- Monte Carlo utilities
- in-memory data adapters for the current backtesting examples
- reusable signal runners for Zipline Reloaded and NautilusTrader
- normalized backtesting reports for common summaries, equity curves, returns, and trade logs
- standard strategy artifact helpers and a generic scored-panel top-k replay helper
- optimal_trader historical artifact replay helpers that avoid live-trading imports
- sample framework-specific SMA crossover strategies for `backtesting.py`, Zipline Reloaded, and NautilusTrader
- executed notebooks covering multi-provider, multi-backtesting-framework, WFO, Monte Carlo, cross-framework validation, and multi-ML-framework MAG7 workflows
- notebooks as integration tests for the current research workflows

Still missing:

- leakage-aware dataset visibility controls
- generic job wrappers for strategy execution, parameter optimization, portfolio combination, and simulation beyond the current concrete replay helpers
- a generic external-engine adapter example

## Next Implementation Steps

1. Promote repeated concrete helpers into atomic train, predict, run, optimize, combine, and simulate stages only after multiple notebooks share the same artifact handoff.
2. Add leakage-aware dataset visibility controls around context artifacts.
3. Make Dagster jobs call these primitives where scheduled execution is needed, without moving research scheduling into Quant Orchestrator.
4. Add one external-engine proof path, such as a QuantConnect-style adapter.
5. Keep notebook workflows as integration tests and examples of composition.

## Non-Goals

- Do not turn this repo into a generic workflow engine.
- Do not add live broker execution here.
- Do not make the platform dependent on one ML framework or one backtesting engine.
- Do not require every strategy to be implemented in every engine.

The multi-rate issuer/instrument trainer uses bounded Polars windows and shared
within-step issuer encoding. Oracle/HITS are supervised labels only and never
historical model inputs. Causal elapsed-time and information-age embeddings
represent irregular gaps; NTP evaluation compares each family with persistence.
The epoch monitor prints fixed post-cutoff evaluation trends without restarting training.
Optional BF16 autocast is available for measured precision/batch-size comparisons.
Masked and next-observation objectives reconstruct individual values at token
and subtoken levels. Subtoken MTP masks features within each family; token MTP
masks whole families. NTP predicts coherent next family and combined-token observations.
See the [reconstruction contract](multirate-reconstruction.md) for the objective details, checkpoint
requirements, executed diagnostic scope, and longer-history 1T experiment.

`platforms/backtesting_frameworks/anchored_hits_replay.py` provides bounded
Polars replay of the older HITS daily-percentile equity policy, with separate
long/short books and explicit next-session-close execution. The executed v6
comparison and its fixed-weight accounting limits are documented in
[multirate-v6-backtest.md](multirate-v6-backtest.md).
The epoch monitor can additionally score the full validation calendar and print
anchored HITS long/short portfolio returns after each immutable epoch checkpoint.
For the user-selected original transformer strategy, epoch evaluation now uses
`existing_multirate_backtest.py`, a thin adapter calling the existing score-policy
and shared-book functions. Data preparation stays in Polars; only one bounded
annual panel crosses the original engine's pandas interface. The replacement
0.80 authority-threshold replay is not the reference strategy.

The multi-rate epoch monitor also supports a separate frozen inference corpus
and independent calendar-year backtests (`--inference-corpus`,
`--backtest-by-year`). `research_tools.epoch_evaluation.yearly_epoch_backtests`
filters each year's full score calendar, resets capital, and keeps annual
price snapshots and epoch comparisons separate while reusing the original
transformer trading policy and shared-book engine.

Multi-rate sample metadata preparation groups anchors by issuer/instrument/date
before visiting the bounded streaming caches. Feature histories remain lazy;
the current trainer still builds its sample metadata list before fitting and
prints preparation progress every 25,000 anchors.

`research_tools.sequence_training` builds overlapping training sequences from a
Polars symbol/date index and joins bounded event labels once per window. Each
supervised date is owned once. Streaming contexts retain history at the first
supervised date and subsequent updates; variable lengths are padded per batch.
This reuses the older sequence-level training approach without restoring pandas
or full-corpus feature tensors. The epoch monitor queues immutable checkpoints
so training and annual backtest reporting can progress at different speeds.

Training can opt into synchronous epoch evaluation with
`--epoch-evaluation-dir`: both the model-evaluation report and portfolio report
must exist before the next epoch starts. `--resume-training` restores the saved
weights, optimizer and batch cursor; immutable epoch reports are retained.

`research_tools.oracle_gate_comparison` compares predicted-Oracle entry/exit
permission against completed epoch baselines using their exact frozen price
snapshots and score calendars. The optional gate is implemented in the existing
backtest score adapter; original trading-engine logic and training remain intact.

The comparison accepts `mode="directional"` for relative buy/short agreement
without an absolute Oracle entry threshold or sell/cover veto. Directional
artifacts use `oracle_directional_gate` paths, preserving strict and baseline
results. The fresh v8 workflow watches completed epochs for both comparisons.

`research_tools.multirate_source_inventory` stages broad warehouse histories
using bounded Polars reads and records missing sources, schemas, historical
coverage and ambiguous cadence. See `docs/multirate-expanded-coverage.md`; this
audit is preparation for expanded training, not a completed expanded model.


Expanded $1T smoke (September 11): `notebooks/multirate_expanded_smoke.ipynb` records the subset/build/launch recipe. The actual run is `artifacts/multirate_recovery/1T/train_expanded_smoke_v9`: one fresh epoch, 13 equity symbols / 11 issuers plus six existing option paths, 40 restored feature families and 1,379 individual numeric fields, 16 observed sparse families (no holder exits). It uses Polars streaming, the pre-2024 cutoff, and blocking original anchored-HITS backtests for 2024, 2025 and 2026 through September 9. Government and insider buy/sell heads now use exact transaction-date targets and actual opposite trades for negatives; historical event inputs retain disclosure availability. This is a pipeline smoke test, not proof of predictive performance or complete point-in-time vintage coverage. Several holder/ETF families have only post-cutoff observations. The older $100B epoch-four run is paused with its checkpoint preserved.


The expanded $100B run now uses `artifacts/multirate_recovery/100B/train_expanded_v9`, with 12 fresh epochs, batch size 32, all 40 restored numeric families and 17 sparse families. It follows the completed $1T smoke and runs 2024/2025/2026-through-September-9 anchored-HITS backtests after each epoch. The numeric corpus passed duplicate/nonfinite checks; 103 legacy congressional/insider issuer sources and Walmart dividends were refreshed with original snapshots preserved. See `notebooks/multirate_expanded_smoke.ipynb` for artifact locations.


Multi-rate loading now uses bounded Polars-to-Torch issuer indexes (`research_tools/streaming_context.py`) with read-only disk mappings and a bounded calendar-window fallback. The trainer keys caches by source fingerprints, normalization, feature layout and optional option inputs, preventing cross-fit reuse. No learned encoder outputs are cached across optimizer steps. The controlled expanded-$100B benchmark in `notebooks/multirate_indexed_loading_benchmark.ipynb` measured 72.99 → 30.19 seconds/batch including first-use index builds, and 3.52 seconds with reused indexes (2.42x / 20.76x); all three losses matched exactly. Twelve real-data window comparisons also matched exactly; peak indexing process RSS was 14.85 GiB. These are three-batch measurements, not full-epoch estimates. The existing $100B run resumed from epoch-one batch 50, retaining model/optimizer/RNG state and the same yearly epoch backtests.

Multi-rate epoch evaluation now groups consecutive dates by symbol, reuses unchanged annual/quarterly contexts, skips disabled document and unused supervised-label work, and batches NTP audit transfers with bounded duplicate tracking. Successor targets use an exact vectorized index calculation; training releases unused CUDA reservations before each blocking backtest. The expanded-$100B benchmark measured 43.46 → 15.62 seconds for 384 mixed-universe observations (2.78x) and 26.91 → 11.81 seconds for consecutive-date panels (2.28x). All 26 supervised heads agree within 1.2e-6, tested HITS rankings match, and NTP pair counts/persistence baselines match exactly. These are scoring-loop measurements, not full-backtest timings. The full evaluator scores 96,181 windows versus 14,849 spaced training windows per epoch; it records inference and portfolio durations separately. The rolling run completed epoch-one evaluation and was superseded by the fresh document run described below. See `notebooks/multirate_evaluation_speed_benchmark.ipynb` for the reproducible comparison.

The active multi-rate $100B run is now `train_documents_v10`, trained from fresh weights with calendar-quarter documents and stable as-of prefixes. Full 2024–2026 scoring preserves all 96,181 dates while reducing 96,181 rolling windows to 1,292 documents: measured inference was 66.35 seconds versus 2,435.86 seconds (36.71x), with another 2.84 seconds for the six yearly books. This was a three-step smoke-checkpoint timing test, not a learned-return comparison. Input layouts and all 42 tasks match; real-data prefix predictions agree within 8.35e-7 and 86 targeted tests pass. Training remains pre-2024, uses Polars streaming and batch 16, and runs the original adjusted-price anchored-HITS backtests after every epoch. See `docs/multirate-document-scoring.md` and `notebooks/multirate_document_scoring.ipynb` for the contract, coverage, ticker-price mapping and reproducible evidence.
