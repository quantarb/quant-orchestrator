# Quant Orchestrator Vision

This note captures the current direction so future work can resume without reconstructing the design from scratch.

## Core Goal

`quant-orchestrator` should be a composable research orchestration layer, not a fixed ML-plus-backtest workflow.

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

## Reusable Primitives

The intended platform primitives are:

- train a model
- generate predictions
- run a strategy
- optimize parameters
- construct a portfolio from strategy outputs
- simulate returns or trade sequences
- compare runs

Monte Carlo and walk-forward optimization are primitives or workflow patterns, not hardcoded assumptions about every run.

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
- FlairNLP has a shared helper for mixed classification/regression multitask training because Flair 0.15.x needs a small evaluation patch for `TextClassifier` plus `TextRegressor` jobs.

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
- ML and backtesting provider contracts
- MLflow tracking helpers
- Dagster entry points
- walk-forward window utilities
- Monte Carlo utilities
- in-memory data adapters for the current backtesting examples
- normalized backtesting reports for common summaries, equity curves, returns, and trade logs
- sample framework-specific SMA crossover strategies for `backtesting.py`, Zipline Reloaded, and NautilusTrader
- FlairNLP shared helper for mixed classification/regression MTL
- executed notebooks covering multi-provider, multi-backtesting-framework, WFO, Monte Carlo, cross-framework validation, and multi-ML-framework MAG7 workflows

Still missing:

- a first-class job graph model
- explicit input/output wiring for jobs
- leakage-aware dataset visibility controls
- generic strategy execution jobs
- generic parameter optimization jobs
- portfolio combination jobs
- a generic external-engine adapter example

## Next Implementation Steps

1. Add small job and artifact reference types.
2. Split existing concrete helpers into atomic train, predict, run, optimize, combine, and simulate steps.
3. Make Dagster compose those primitives instead of assuming one workflow shape.
4. Add one external-engine proof path, such as a QuantConnect-style adapter.
5. Keep notebook examples as examples, not the primary contract.

## Non-Goals

- Do not turn this repo into a generic workflow engine.
- Do not add live broker execution here.
- Do not make the platform dependent on one ML framework or one backtesting engine.
- Do not require every strategy to be implemented in every engine.
