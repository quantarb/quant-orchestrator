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

The artifact registry already exists and should remain schema-light.

It should store native outputs from:

- sklearn
- PyTorch
- Flair or other NLP frameworks
- Zipline
- NautilusTrader
- VectorBT
- QuantConnect or similar external engines
- JSON, CSV, text, directories, pickles, or bytes

Different frameworks should be allowed to produce different native reports or file layouts.

Common reporting should be additive, not destructive. Backtesting reporting adapters should expose comparable summaries, equity curves, returns, and trade logs where possible, while preserving each framework's unique native metrics and artifacts.

## Backtesting Model

Backtesting adapters should stay thin.

They should accept:

- a strategy object or callable
- prepared warehouse data
- optional runner-specific parameters

This allows external strategies to be replayed without rewriting the strategy itself for every engine.

Data adapters should bridge Quant Warehouse frames into each native engine without duplicating datasets. Prefer in-memory adapters when the framework supports them, as the current Zipline Reloaded, NautilusTrader, and `backtesting.py` examples do.

Example strategies can live in package code when they are reused across notebooks for framework comparison. Notebook-specific experiment orchestration should stay in notebooks until it becomes a repeated platform capability.

## ML Framework Model

ML framework helpers should stay close to the native framework while removing repeated integration friction.

Current examples:

- RAPIDS cuML provides a CUDA-backed sklearn-style RandomForest path.
- PyTorch uses CUDA auto-detection for tensor models.
- FlairNLP can be used for native multitask learning. Any local Flair compatibility patch should be treated as temporary integration code, not a long-term platform abstraction.

ML outputs should remain native unless there is a clear reason to normalize them. A common metrics table is useful for comparison, but the platform should still store framework-specific reports and artifacts.

## Intended Workflow Examples

- Train one model in one ML framework, then feed its predictions into a strategy.
- Train multiple models in multiple frameworks, then compare their downstream strategy outputs.
- Backtest a QuantConnect strategy on warehouse data, then optionally replay its equity curve in another engine.
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
- normalized backtesting reports for common summaries, equity curves, returns, and trade logs
- sample framework-specific SMA crossover strategies for `backtesting.py`, Zipline Reloaded, and NautilusTrader
- a temporary local FlairNLP evaluation compatibility patch used by the current multi-ML notebook
- executed notebooks covering multi-provider, multi-backtesting-framework, WFO, Monte Carlo, cross-framework validation, and multi-ML-framework MAG7 workflows
- notebooks as integration tests for the current research workflows

Still missing:

- explicit input/output wiring for jobs
- leakage-aware dataset visibility controls
- generic strategy execution jobs
- generic parameter optimization jobs
- portfolio combination jobs
- a generic external-engine adapter example

## Next Implementation Steps

1. Split repeated concrete helpers into atomic train, predict, run, optimize, combine, and simulate stages when multiple notebooks share the same artifact handoff.
2. Add leakage-aware dataset visibility controls around context artifacts.
3. Make Dagster jobs call these primitives where scheduled execution is needed, without moving research scheduling into Quant Orchestrator.
4. Add one external-engine proof path, such as a QuantConnect-style adapter.
5. Keep notebook workflows as integration tests and examples of composition.

## Non-Goals

- Do not turn this repo into a generic workflow engine.
- Do not add live broker execution here.
- Do not make the platform dependent on one ML framework or one backtesting engine.
- Do not require every strategy to be implemented in every engine.
