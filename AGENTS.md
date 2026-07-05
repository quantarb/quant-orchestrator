# Repository Rules

## Dependency Source Of Truth

- Use `quant-warehouse` from `https://github.com/quantarb/quant-warehouse.git@main`.
- Do not commit local editable `quant-warehouse` dependency paths.

## Data Boundary

- `quant-orchestrator` should consume data through `quant-warehouse`; it should not call OpenBB or vendor market-data APIs directly.
- If warehouse data is missing or incomplete, fix the OpenBB fork provider first, then refresh `quant-warehouse`.
- ThetaData options backfills must preserve full daily chains for each selected underlying symbol. Do not add or keep DTE, strike-range, moneyness, bid/ask, minimum-ask, liquidity, or contract-level filters that can exclude contracts from a symbol's daily chain. Universe filters such as `min_market_cap` may select which underlying symbols to backfill, but once a symbol/day is selected the download and storage path should keep every option contract returned for that day.

## Orchestrator Responsibility

- `quant-orchestrator` is responsible for composable research workflows: ML training, inference, backtesting, external strategy evaluation, artifact storage, and orchestration.
- `quant-orchestrator` owns CRUD/storage for ML training runs, backtest runs, ML artifacts, backtest artifacts, and strategy artifacts. Downstream apps should request work here and load returned artifact URIs/paths instead of keeping their own research storage.
- The repo is opinionated toward Dagster for orchestration and MLflow for experiment tracking.
- Do not add live trading, broker order submission, or broker account mutation code here.
- If a workflow needs market data, features, labels, or warehouse refreshes, call `quant-warehouse` rather than OpenBB or vendor APIs directly.
- Artifact storage should be schema-light. Different ML frameworks and backtesting frameworks may emit incompatible reports, models, plots, directories, or binary objects; store the native outputs with minimal metadata rather than forcing one universal report shape.
- Standard strategy artifacts are the exception to the schema-light rule. Reusable strategy/backtest handoffs should use `StrategyArtifactBundle` from `quant_orchestrator.platforms.backtesting_frameworks.strategy_artifacts` with `scored_panel`, `action_tape`, `trade_windows`, `summary`, and `strategy_artifacts_manifest.json` when downstream workflows need to consume the result.
- `trade_windows` is the preferred handoff into option-equivalent replay, WFO summaries, Monte Carlo over trades, and cross-framework comparison. Keep the required columns stable: `trade_id`, `symbol`, `side`, `entry_date`, and `exit_date`. Preserve `equity_entry_notional` when available so option replay can use the same capital budget as the equity trade.
- Do not assume every workflow is ML-driven, equity-only, or backtest-driven. Train-only, inference-only, backtest-only, train-then-backtest, and external-engine strategy runs should all fit the platform model.

## ML Training Vs Strategy Backtesting Boundary

- ML model training and ML model evaluation datasets must stay event-only. Do not add synthetic `no_event`, non-event, daily filler, or unlabeled feature-panel rows to classifier training/evaluation labels unless the user explicitly asks for a different modeling problem.
- Event-only ML datasets may contain event labels, oracle labels, and event-derived targets. They should not be expanded merely to make the later trading calendar dense.
- ML inference/scoring for a trading strategy must run over the full intended scoring universe and calendar, not only over event-labeled rows. A model can be trained on event rows and then scored on every eligible symbol/date row needed by the strategy.
- Trading strategy backtests, option-window generation, portfolio simulation, and strategy parameter optimization must use the full scored strategy dataset for the intended universe/calendar. Do not restrict strategy backtests to ML label rows, event rows, or target rows unless the strategy itself explicitly trades only on those event rows.
- Keep these dataframes separate in code and artifacts: event-only model training/evaluation frames, full-universe score frames, and full-universe strategy/backtest frames.
- ML strategy comparisons should vary the model prediction source, not the trading planner. Single feature-family models, MoE family experts, classifier/regressor/AE composites, and mean ensembles should all be normalized into the same daily score schema and traded through the optimal_trader-compatible capacity planner: hold until classifier direction flips, fill freed capacity with the highest-scored eligible opportunities, and avoid time-based/random rotations. For the current optimal_trader-compatible ML option strategy, classifiers control direction and exits: new entries require selected classifier agreement for the side, and existing positions exit when any selected classifier stops agreeing with the held side. Regressor/ranking and AE/familiarity signals contribute to entry filtering/ranking, not exits, unless a future strategy explicitly defines different permissions.
- For generic scored-panel strategy experiments, use `scored_panel_replay.py` to convert daily scores into `action_tape` and `trade_windows` through the shifted top-k rule instead of rebuilding that loop in notebooks.

## Option Backtesting Boundary

- Equity strategies are the decision makers for option-equivalent backtests. Generate or load equity `trade_windows`, then select and price options for each trade window.
- Do not implement option-equivalent backtests as a daily loop over every date, symbol, and option candidate unless a strategy specifically requires daily option rebalancing.
- Option selection should pull the full option chain only on the equity entry date. After selecting contracts, load or price only the selected option time series through the equity exit date or option expiration.
- If the selected option expires before the equity trade exits, apply expiration value including intrinsic value. Do not roll into a new option unless the strategy explicitly defines a roll rule.
- Option replay should log trade-level statuses such as missing chain, no eligible option, missing selected path, expired worthless, and priced. This makes failures inspectable and keeps parallel trade execution safe.
- New option strategy notebooks should consume `trade_windows.parquet` or `strategy_artifacts_manifest.json`. They should not rebuild classifier signal windows, option candidate panels, and portfolio accounting in the same notebook.

## Compatibility Policy

- This repo is new and rapidly changing. Do not add backward-compatibility wrappers, legacy aliases, or duplicate old APIs.
- When package structure changes, update imports and notebooks directly.
- Do not preserve old provider categories, old entry-point groups, or old module paths.

## Build Vs Buy Policy

- Prefer widely used, actively maintained third-party packages or small forks of proven projects over custom implementations.
- For ML frameworks, backtesting frameworks, experiment tracking, orchestration, model serialization, metrics, reports, and simulations, use battle-tested libraries when they fit the repo boundary.
- Build from scratch only when no reliable package fits the requirement or this repo needs a thin opinionated wrapper around a proven dependency; document that reason in the change.

## Platform Policy

- Keep provider-style extension points only for `platforms/ml_frameworks` and `platforms/backtesting_frameworks`.
- Do not add provider abstractions for orchestration or experiment tracking. Use Dagster and MLflow directly through the repo's opinionated interfaces.
- Do not add broker platforms or live-trading adapters.
- Backtesting framework providers should be thin adapters around native engines and caller-provided strategies/runners. For engines such as QuantConnect, keep the native strategy implementation intact and adapt warehouse inputs plus artifact outputs around it.
- Framework-specific historical replay code belongs under `quant_orchestrator/platforms/backtesting_frameworks/<framework>/`. Current optimal_trader replay code belongs under `platforms/backtesting_frameworks/optimal_trader/` and must remain historical/backtesting-only.

## Notebook Policy

- Use `notebooks/` for one-off research workflows: model training experiments, backtest experiments, walk-forward experiments, Monte Carlo experiments, and equity-curve analysis.
- `notebooks/` should contain notebook files only. Do not add standalone `.py` scripts in that directory.
- Keep exploratory, one-off, and notebook-only code inside the notebook itself so people can edit and rerun it interactively.
- If a notebook workflow becomes a repeated capability, move the reusable API into package code and leave the notebook as an example.
- Notebooks in this repo must not implement feature engineering, target engineering, or warehouse refresh logic. Pull prepared datasets from `quant-warehouse`.
- If a notebook needs a new feature family or label, implement it in `quant-warehouse` first, then consume it here.
- Data adapters, reporting adapters, and repeated framework runners are reusable platform code and should live under the relevant framework module, not inside notebooks.
- Repeated strategy contract writers and readers belong in package code. Notebooks may choose the strategy, score columns, thresholds, universe, and output directories, but should call package helpers to write the standard artifact contract.
- Notebook documentation and saved outputs should be updated when notebook behavior materially changes.

## CUDA Policy

- Optimize model training and simulation code for CUDA-first execution when the selected ML framework supports it.
- Prefer GPU-native libraries such as PyTorch CUDA and RAPIDS/CuPy where they fit the workflow.
- For sklearn-style GPU training, prefer RAPIDS cuML instead of CPU scikit-learn when the environment supports it.
- Do not keep slow CPU compatibility paths unless they are the only practical path for a required dependency.

## Documentation Policy

- Keep `README.md`, `notebooks/README.md`, and `docs/orchestrator-vision.md` aligned with current code.
- Code and executed notebooks are the source of truth. Remove stale claims instead of documenting intended future behavior as if it exists.
- When adding a new repeated platform helper, document where it lives and which notebook or workflow uses it.

## Resume Point

- Before extending the platform shape, read `docs/orchestrator-vision.md` and keep the current repository code as the source of truth for any claims.

## Git Hygiene

- This workspace is maintained by a single author. Push completed changes directly to the remote branch; do not create or wait on pull requests unless the user explicitly asks for one.
- Use normal branch pushes for published work. Use `--force-with-lease` only when intentionally rewriting the current branch history.
