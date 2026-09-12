# Expanded multi-rate source audit

The existing fresh v8 corpus is narrower than the historical 40-feature-family
plus 17-event-family inventories. `research_tools.multirate_source_inventory`
now snapshots warehouse sources with bounded Polars reads, one issuer/source at
a time. It records field schemas, finite numeric fields, date ranges, pre-2024
coverage, missing reads and failures. Snapshots resume without repeating successful
reads; failed reads are retried. This is staging data, not a completed training
corpus, and it does not change the active checkpoint or its training inputs.

The executed 100B inventory is `artifacts/multirate_recovery/100B/expanded_sources_v9`.
Across 114 issuers, 3,881 source reads returned observations, 451 were empty, and
zero errors remain after correcting timezone-aware institutional read slicing.
These are source-read counts, not model family counts.

All 114 issuers have pre-2024 ratios and metrics in period-unspecified storage.
Explicit annual/quarter paths contain no pre-2024 ratio/metric observations.
AAPL has 166 historical observations for each, spanning 1985-09-30 to 2026-07-06.
The older fiscal_period values inspected are NaN: history must not be silently
assigned to an annual/quarterly family based on guessed cadence.

Macro facade delegation was also repaired in quant-warehouse. The inventory
includes actual economic history (10 series over 10,021 dates) and Treasury
history (12 series over 9,158 dates), rather than empty schema placeholders.

The historical government-trade event export used transaction dates even when
raw warehouse records contain later disclosure dates. For example AAPL's
2014-02-10 transaction was disclosed 2014-02-27. Restore this input from raw
records using disclosure availability; do not copy that export into historical
inputs unchanged. Oracle/HITS remain supervised-only.

Remaining expansion work: map source snapshots to explicit family adapters,
retain individual numeric/text features, restore sparse events with availability
provenance, handle unknown cadence without fabricated labels, audit usable
pre-cutoff coverage, and train/validate a separate expanded model. The inventory
alone does not establish that every older family is populated or learnable.


Expanded $1T smoke (September 11): `notebooks/multirate_expanded_smoke.ipynb` records the subset/build/launch recipe. The actual run is `artifacts/multirate_recovery/1T/train_expanded_smoke_v9`: one fresh epoch, 13 equity symbols / 11 issuers plus six existing option paths, 40 restored feature families and 1,379 individual numeric fields, 16 observed sparse families (no holder exits). It uses Polars streaming, the pre-2024 cutoff, and blocking original anchored-HITS backtests for 2024, 2025 and 2026 through September 9. Government and insider buy/sell heads now use exact transaction-date targets and actual opposite trades for negatives; historical event inputs retain disclosure availability. This is a pipeline smoke test, not proof of predictive performance or complete point-in-time vintage coverage. Several holder/ETF families have only post-cutoff observations. The older $100B epoch-four run is paused with its checkpoint preserved.


The expanded $100B run now uses `artifacts/multirate_recovery/100B/train_expanded_v9`, with 12 fresh epochs, batch size 32, all 40 restored numeric families and 17 sparse families. It follows the completed $1T smoke and runs 2024/2025/2026-through-September-9 anchored-HITS backtests after each epoch. The numeric corpus passed duplicate/nonfinite checks; 103 legacy congressional/insider issuer sources and Walmart dividends were refreshed with original snapshots preserved. See `notebooks/multirate_expanded_smoke.ipynb` for artifact locations.


Multi-rate loading now uses bounded Polars-to-Torch issuer indexes (`research_tools/streaming_context.py`) with read-only disk mappings and a bounded calendar-window fallback. The trainer keys caches by source fingerprints, normalization, feature layout and optional option inputs, preventing cross-fit reuse. No learned encoder outputs are cached across optimizer steps. The controlled expanded-$100B benchmark in `notebooks/multirate_indexed_loading_benchmark.ipynb` measured 72.99 → 30.19 seconds/batch including first-use index builds, and 3.52 seconds with reused indexes (2.42x / 20.76x); all three losses matched exactly. Twelve real-data window comparisons also matched exactly; peak indexing process RSS was 14.85 GiB. These are three-batch measurements, not full-epoch estimates. The existing $100B run resumed from epoch-one batch 50, retaining model/optimizer/RNG state and the same yearly epoch backtests.

Multi-rate epoch evaluation now groups consecutive dates by symbol, reuses unchanged annual/quarterly contexts, skips disabled document and unused supervised-label work, and batches NTP audit transfers with bounded duplicate tracking. Successor targets use an exact vectorized index calculation; training releases unused CUDA reservations before each blocking backtest. The expanded-$100B benchmark measured 43.46 → 15.62 seconds for 384 mixed-universe observations (2.78x) and 26.91 → 11.81 seconds for consecutive-date panels (2.28x). All 26 supervised heads agree within 1.2e-6, tested HITS rankings match, and NTP pair counts/persistence baselines match exactly. These are scoring-loop measurements, not full-backtest timings. The full evaluator scores 96,181 windows versus 14,849 spaced training windows per epoch; it records inference and portfolio durations separately. The rolling run completed epoch-one evaluation and was superseded by the fresh document run described below. See `notebooks/multirate_evaluation_speed_benchmark.ipynb` for the reproducible comparison.

The active multi-rate $100B run is now `train_documents_v10`, trained from fresh weights with calendar-quarter documents and stable as-of prefixes. Full 2024–2026 scoring preserves all 96,181 dates while reducing 96,181 rolling windows to 1,292 documents: measured inference was 66.35 seconds versus 2,435.86 seconds (36.71x), with another 2.84 seconds for the six yearly books. This was a three-step smoke-checkpoint timing test, not a learned-return comparison. Input layouts and all 42 tasks match; real-data prefix predictions agree within 8.35e-7 and 86 targeted tests pass. Training remains pre-2024, uses Polars streaming and batch 16, and runs the original adjusted-price anchored-HITS backtests after every epoch. See `docs/multirate-document-scoring.md` and `notebooks/multirate_document_scoring.ipynb` for the contract, coverage, ticker-price mapping and reproducible evidence.

## $10B document startup

The September 12 run under `artifacts/multirate_recovery/10B/train_documents_v10` passed source preparation for all 791 issuers, event validation, and duplicate-row checks. Its 793 equities and eight option paths retain all 40 expanded feature families, with 1,751 numeric fields including the base corpus. Source repairs preserve original snapshots under `source_repairs/` and `event_repairs/`. Empty optional ratio histories and unknown transaction directions stay missing.

The first optimizer step and reloadable model/optimizer/RNG checkpoint were verified. The trainer now saves every ten batches and retains yearly blocking backtests for 2024, 2025, and 2026 through September 9. The disk-index limit is 256 MiB per issuer, with bounded-window fallback above that limit; the first-batch loss matched exactly after this change. Full epoch and backtest results remain pending. Inspect the live logs and `training_start_verification.json`.

The $10B training label path now writes one sorted event-only snapshot per training invocation and uses a bounded 128 MiB per-symbol exact-date label index (`research_tools/supervision_index.py`). Source cutoffs, disabled tasks, document ownership, missing labels, and batch order are preserved. A five-batch checkpoint-resume comparison measured 3.32 → 1.47 seconds/batch after excluding the first batch; all five losses and final model tensors matched exactly. This is a short benchmark, not a full-epoch estimate. That benchmark resumed saved optimizer/RNG state; the corrected run below starts fresh and saves checkpoints every ten batches. Evidence is under `artifacts/multirate_recovery/10B/speed_benchmark/`.


The corrected $10B run is `artifacts/multirate_recovery/10B/train_documents_v11`. Training selects calendar quarters with finite issuer observations (prices, issuer financial features, or sparse events); shared macro, sector/industry aggregates, and calendar fields cannot create documents by themselves. All context histories remain available inside valid documents, and inference retains its full calendar. The audited selection falls from 332,482 to 101,681 documents, including valid report/event-only quarters and actual supervised target dates, under the pre-2024 cutoff. The run starts fresh model/optimizer state because the old batch cursor belongs to a different sample set; the v10 artifacts remain preserved. `training_document_contract=issuer_observation_quarters_v1` prevents incompatible training resumes. See `issuer_document_audit.json` and `training_documents_v11.log` under the $10B root.


Annual recurrent documents are available with `--sequence-mode annual_memory`. Calendar years carry detached learned token/family state forward; chronological instrument batches and checkpointed memory preserve the handoff during training and resume. Inference starts with empty memory at the requested start date, processes no earlier observations, and carries memory only within the requested interval. The two-epoch $10B subset smoke completed all 2024/2025/2026 yearly backtests and full score-coverage checks; its first annual epoch took 3.0 seconds versus 6.6 seconds for quarterly sampling. This is a subset measurement. See `docs/multirate-annual-memory.md` and `artifacts/multirate_recovery/10B/annual_memory_smoke/`. The corrected quarterly v11 run is paused with checkpoints preserved while annual v12 is launched.


Annual backtests now run without historical warmup. The first document is clipped to the requested start date, including for mid-year or last-day inference. Trading scores require an observed price; macro-only rows remain model context but cannot create trades for inactive symbols. Training and its recurrent-memory checkpoint contract are unchanged.


The full $10B epoch-one no-warmup evaluation completed: 2,385 documents, 511,815 price-date scores, no missing/duplicate/unexpected/nonfinite predictions, and all 2024/2025/2026-through-September-9 long/short books. Scoring took 200.42 seconds versus 1,234.02 seconds with historical replay; the portfolio backtests took 13.75 seconds. The previous replay attempt failed on a macro-only ABMD trading calendar; observed-price scoring removes that invalid calendar. Evidence: `artifacts/multirate_recovery/10B/no_warmup_backtest_verification.json`.
