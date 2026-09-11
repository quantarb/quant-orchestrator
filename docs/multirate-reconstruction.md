# Individual-value reconstruction

The multi-rate trainer uses the `features_only_time_v6` objective
contract. Oracle/HITS remain the instrument-specific supervised tasks.

Each token head predicts the complete ordered numeric feature vector for its
rate. Raw fields from the same source endpoint share one family adapter.
The wider 1T corpus has 132 fields: balance (53), cash flow (39), income (31),
instrument terms (4), and OHLCV (5), plus the separate sparse stream. Each subtoken head predicts the channels of one feature family. Unequal
family widths are padded only in the output layout; those padded channels never
contribute to the loss. Values are normalized using the training cutoff and
training universe. Neither token nor subtoken targets average unrelated values.

For next-subtoken prediction, each family targets its next observed vector on a
strictly later date, retaining that observation's missingness. A field absent
at that next family observation is excluded, rather than borrowed from a later
date. For next-token prediction, the target is the combined vector from the
next dated observation. Families may therefore have different next dates while
a token successor always represents a single date. Same-date rows cannot serve
as future targets.

MTP uses two independent corruption passes. The subtoken pass selects observed
individual features with probability 15%, retaining at least one observed
sibling per family. A family with only one observed feature has no within-family
masked target. The token pass selects entire observed families with probability
15%. Natural missingness, padding, and visible values never contribute to either
masked loss. Shared annual/quarterly contexts receive identical corruption within
each pass. Per-family valid-target counts are recorded for each objective.

The model preserves original family presence for deliberately hidden values.
Learned per-feature missingness embeddings identify exactly which input is
missing, even when other observed values are zero. Both next-subtoken and
masked-subtoken heads receive their own family's causal temporal representation.
Token heads receive encoded rate context combining families and history. Future
dates remain blocked. With both objectives enabled, NTP and supervised heads
share the feature-corrupted pass; token MTP uses its separate family-corrupted
pass. Reconstruction losses use per-value MSE over valid targets and default to
weight 0.1 relative to weight 1 for each supervised task.
`--self-supervision both|next|masked|none` enables controlled comparisons.
The `next` and `none` settings do not corrupt inputs.

Oracle and HITS are supervised targets only. Their rows are retained in the
separate supervised scan, then removed before sparse input normalization,
window selection, family adapters, NTP/MTP targets, and issuer context. Their
values are never fed back into the model, even after the outcome is known.
Other sparse feature families retain 16 observations each, merged on equal
dates. Families with fewer than two observations cannot contribute NTP pairs.
An empty sparse feature stream uses a fully missing placeholder, not outcomes.

The family encoder embeds log elapsed days since its previous observation,
log age of its latest observed value, and flags distinguishing known history
from no history. Explicit dates use Unix nanoseconds. Slow-family information
age is also computed at each daily query before instrument fusion; this allows
cached slow representations to remain reusable while their age changes.
Padding and future observations cannot define a clock input. No next-event
interval is provided to the model, and no training dates or labels are shifted.

`ntp_evaluation.py` evaluates clean-input NTP against the last known value of
each individual channel on identical target masks, grouped by rate, family,
and subtoken/token level. A bounded SQLite cache deduplicates overlapping
windows by instrument, family, source date, and target date; the first window
in chronological evaluation order is retained. Values without a historical
persistence estimate are excluded from both compared errors. The report includes
counts, forecast-horizon range, model MSE, persistence MSE and skill
`1 - model_error / persistence_error`. Skill is null when persistence error is
zero. Errors use training-normalized units and value weighting. Evaluation
intervals filter target dates. Reports assess available pairs within model
windows, not observations absent from the corpus or outside those windows.

The checkpoint records the objective contract. Checkpoints trained with the
previous scalar-average objectives must be retrained; the trainer rejects them
instead of silently loading incompatible reconstruction heads.

## Data and experiment scope

`build_multirate_mtl_corpus.py` reads stored warehouse data with Polars, defaults
to a 1900 start request, and retains each source's available history. Annual and
quarterly sections on the same date are merged at the context-read boundary.
Missing early statement history does not remove earlier price observations.
Known internal source gaps require an explicit matching audit when building via
`build_fresh_corpus(audited_statement_gaps=...)`; an audit documents gaps, it
does not fill them or prove vendor completeness.

The repair's four-issuer corpus covers equities, actual option contracts,
exchange-listed debt, and preferred shares. Its earlier two-epoch 2024/2025
full-context and no-context diagnostic runs completed using the old objectives.
They establish pipeline execution, not predictive usefulness. In the first
2024 full-context evaluation, HITS errors exceeded the training-mean baseline
and the fixed long-only rule stayed in cash.

The longer 1T corpus contains 13 US equity symbols grouped into 11 issuers,
plus six actual AAPL option contracts for 2023–2025. Available prices begin in
1970. It is a retrospective screened universe, not historical point-in-time
market-cap membership. Training stops before 2024; 2024–2025 are reserved for
evaluation. The expanded 100B preparation remains separate. The 1T corpus does
not establish broad multi-asset or unseen-issuer generalization.

## Verification

`tests/test_multirate_reconstruction.py` checks channel preservation when means
coincide, coherent family and token successors, both masking modes, padding
exclusion, and dependence on past observations without future access for every
rate. The broader multi-rate, supervision, corpus, and replay selection passes
93 tests. Run the real-data CUDA/checkpoint checks before claiming a training
run is ready; unit tests alone do not establish model quality.

## Bounded streaming performance

`StreamingContext` caches at most 32 blocks per rate, each with at most the
requested window length plus 1,024 rows. Sparse contexts maintain this
bounded cache separately for each configured family. It converts each bounded block to a
CPU tensor once and copies only the requested as-of window into a batch.
Date-only version caches contain at most 4,096 dates for each of 32 symbols;
older requests fall back to an exact bounded query. Dense intrayear requests
also fall back when a block cannot cover the requested historical window.
Caches do not expose later observations to the model. Tests compare cached
and uncached results, cover empty histories, old anchors, eviction and date
unions. No full-corpus dataframe or tensor is constructed.

The configured full 1T run uses 86,090 historical training samples, batch size 128,
64-dimensional states, two layers, four heads and 12 epochs. Before the v5 hierarchy repair, the 2,048-sample
CUDA check completed one epoch with finite losses; its peak process RSS was
3,302 MiB and peak CUDA allocation was 5,270 MiB before the final tensor-block
conversion optimization. These are execution checks, not evidence of useful
held-out trading predictions. Completed evaluation results must be reported
separately from implementation and training status.

## Assessment and next experiments

Held-out trading utility remains unproven. Use the NTP report to establish
whether temporal reconstruction beats persistence, then compare supervised-only,
NTP, MTP and combined runs under identical trading rules. Eleven issuers are
an execution trial, not broad generalization evidence. Event-level Oracle/HITS
input experiments are excluded: the user requires these to remain supervised
labels only. The older v5 diagnostics below describe an archived model that
included target-derived context; its long run was stopped after that correction.
For a model trained before 2024, both 2024 and 2025 baseline statistics must use
that same training cutoff, separately from the evaluation interval.

The v5 hierarchy CUDA check completed 2,048 samples in one epoch with finite
loss (4.207217), nonzero gradients for every rate encoder and instrument fusion,
and nonzero valid targets for HITS, Oracle, and insider families in all four
reconstruction objectives. Peak process RSS was 3,342 MiB; peak CUDA allocation
was 8,121 MiB. Counts include overlapping training windows and are not counts of
unique financial events. These checks establish execution, not generalization.

## EOD information and execution dates

Prediction CSV `date` denotes the EOD information date, also written explicitly
as `information_date`. It is not the execution date. All rate windows use the
same as-of anchor, retaining older available observations for slower families.
The existing replay consumes that score on the following observed trading
session; it does not trade on the score's information date. For example, a
Friday EOD score is eligible for Monday execution when Monday is the next
session. `prediction_timing.json` records this contract beside the scores.

Per the requested training contract, supervised features and event labels stay
aligned to the same date. NTP retains its next-observation targets and MTP its
same-observation reconstruction targets. The execution delay belongs in the
backtest, not in an extra shift of model inputs or supervised labels. Current
scoring anchors follow instrument price dates, so weekend-only updates are not
separately scored. Recorded source dates remain unchanged; no publication lags are invented.

## v6 execution check

The corrected model completed a 2,048-sample CUDA epoch with finite loss
(5.279881), peak process RSS 3,374 MiB and peak CUDA allocation 8,059 MiB.
Checkpoint reload scored 135 instrument/date rows for 2024-03-25 through
2024-04-05. The only sparse input family was insider trading; HITS/Oracle
supervision remained active. The small diagnostic model did not beat persistence
in any measured family/level group. This short check is not a completed training
experiment. Annual groups had no eligible successor in that evaluation interval;
reports now explicitly include zero-coverage groups with null error/skill.

## Epoch-by-epoch monitoring

`monitor_multirate_epochs.py` attaches to a running training command without
restarting it or changing optimizer state. It copies a completed-epoch
checkpoint, evaluates a fixed bounded post-cutoff sample using the inference
path, and prints per-family/per-level model MSE, persistence MSE, skill and
change in skill from the preceding epoch. It rejects changed target counts or
baseline errors between epochs. Positive skill beats persistence; increasing
skill means improvement. Zero-error baselines and absent coverage retain null
skill instead of a misleading percentage.

The current full run trains all eligible samples before 2024, with no training
sample cap and no pre-2024 validation holdout. Its epoch monitor uses a fixed
256-anchor sample from 2024; 2025 is reserved for the final test. Monitoring does
not update gradients or select checkpoints. Each run has an `epoch_metrics.json`
and an immutable checkpoint under its epoch directory. The companion process
prints to `train_long_v6/epoch_validation.log`; it does not rewrite the active
trainer's log. The status file distinguishes waiting, evaluation and completion.

Example:

```bash
python scripts/monitor_multirate_epochs.py \
  --command-file artifacts/multirate_recovery/1T/training_command_v6.json \
  --training-log artifacts/multirate_recovery/1T/training_v6.log \
  --output-dir artifacts/multirate_recovery/1T/train_long_v6/epoch_validation \
  --validation-start 2024-01-02 --validation-end 2024-12-31 \
  --training-pid TRAINING_PID
```

Calendar coverage and a rolling model window are distinct: the training spans
all available pre-2024 history, while each sample uses bounded windows of 252
daily, 40 quarterly, 16 annual and 16 observations per sparse feature family.
This preserves Polars streaming and bounded memory.

## Batch staging overhead

`multirate_batch.py` stages each raw field once per step. Corruption and padding
repairs use clones so raw reconstruction targets remain unchanged. Diagnostic
counts transfer once per supervised task rather than once per instrument and
task. These changes preserve model architecture, objectives, dates and coverage.
A running Python trainer retains its loaded implementation; new training or
inference processes pick up the optimization.

Component benchmarks on the GB10, while the full run was active, measured
23.2 ms versus 0.36 ms for twelve task counters at batch 128, and 1.16 s versus
0.060 s for twenty reads of a 128×252×132 tensor. These measure the individual
operations, not an end-to-end training speedup. Concurrent GPU work prevents
interpreting the diagnostic run's wall time as an isolated throughput comparison.

## Precision and batch-size benchmarks

The trainer supports `--mixed-precision --autocast-dtype bfloat16` on supported
CUDA hardware. Float16 remains available through `--mixed-precision` (its
default format) with gradient scaling. BF16 uses autocast without a loss scaler;
model parameters and saved weights remain FP32. Inference currently runs FP32.
Precision selection is recorded in the command, checkpoint configuration and
training summary.

Compare precision and batch size on the same corpus, dates, sample selection,
model dimensions, objectives and optimizer settings. A larger batch performs
fewer optimizer updates per epoch, so its loss is not an equal-update comparison.
GPU sharing and data preparation affect measured wall time; use repeated baselines
and report throughput and peak allocation alongside loss and finite-gradient
checks. Existing training processes retain their original precision and batch size.

The 2,048-sample benchmark measured FP32/batch-128 at 90.9 and 91.4 seconds,
BF16/batch-128 at 89.9 seconds, and BF16/batch-256 at 88.4 seconds. Peak CUDA
allocations were 7.9, 5.6 and 11.1 GiB respectively. All runs completed with
finite losses; both BF16 checkpoints reloaded and produced finite post-cutoff
predictions. Throughput gains were modest (about 1.4% and 3.1% relative to the
repeated FP32 baseline), so the full run retained its existing configuration.
The batch-256 check performed half as many optimizer updates; its epoch loss
is not directly comparable as a quality result.

The completed v6 run has a [measured failure diagnosis and training profile](multirate-v6-diagnosis.md).
It identifies pooled price scaling, unseen constant option attributes, and
CPU sample preparation as repair priorities; it does not claim those repairs
or a full-run speedup have been completed.

The unchanged v6 checkpoint also has a [completed adjusted-price backtest](multirate-v6-backtest.md)
for separate 2024 and 2025 folds, including the fixed policy, costs, exposure,
and issuer-equity buy-and-hold comparison.

### 100B run and per-epoch portfolio evaluation

The 100B expansion uses the stored FMP market-cap snapshot, a US NASDAQ/NYSE
profile filter, and min_market_cap=100,000,000,000. The coverage/identity audit
retains 116 equity symbols representing 114 issuers. It excludes missing
pre-2024 history, missing required statements/irregular observations, and three
listed debt securities requiring separate issuer/contract taxonomy. The existing
AAPL annual option cohorts are also requested; this is not a broad option-chain
training expansion. See `artifacts/multirate_recovery/100B/universe_audit_v6.json`
for every candidate and exclusion. The retained statement gaps are recorded
explicitly; no observations are invented to fill them.

The pipeline is configured for 12 epochs with the same v6 model/objectives,
FP32 batch 128, full available stored history from the 1900 request floor, and
exclusive training cutoff 2024-01-01. It builds and audits before training, then
scores 2024–2025 after training. Check `run_status.json` for the actual stage;
configuration alone does not establish that training started or completed.

`monitor_multirate_epochs.py --backtest-anchored-hits` snapshots each completed
epoch, scores the full 2024 calendar (overriding the small NTP sample cap), and
runs the anchored percentile long and short equity books. It reports return,
maximum drawdown, entry events, mean gross exposure, and return change versus
the prior epoch. Inputs use a shared frozen snapshot of adjusted equity prices.
Each epoch stores NTP metrics, scores, equity curves, target weights, and action
tapes under `train_long_v6/epoch_validation/epoch_NNNN/`. Console tables are saved
to `epoch_validation.log`; 2025 remains outside this epoch monitor.

The portfolio uses strict rank >0.80 thresholds, top-20 capacity, allocation
1/min(20, equity universe size), and next-session-close execution. No Oracle
gate applies. These evaluation results do not update gradients or choose the
saved checkpoint; checkpoint selection still uses training loss. Full-calendar
inference/backtests add runtime and may contend with training for the GPU.

The epoch portfolio policy was subsequently corrected to the original multi-rate
transformer score adapter and shared-book engine (0.50 hub percentile plus
direction agreement), rather than the separate anchored 0.80 authority policy.
The latest section of `multirate-v6-backtest.md` records the exact call chain,
original timing/accounting, and the distinction between original total_return
and return from actual initial capital. This is the policy used for the 100B
per-epoch backtests; no model training objectives were changed.

### $100B epoch reporting: 2024, 2025, and 2026 YTD

The active run now monitors the full 2024-01-02 through 2026-09-09 scoring
calendar. `monitor_multirate_epochs.py --inference-corpus ...
--backtest-anchored-hits --backtest-by-year` uses a separate frozen inference
corpus and invokes the same original transformer policy/shared-book engine for
independent annual long and short books. Each year starts with $100,000 and uses
`1 / min(20, symbol_count)` allocation. Return, Sharpe, drawdown, entries,
exposure, and return change versus the previous epoch are printed with period
labels. Price snapshots are cached separately by date range, and scores are
filtered to each period before simulation.

Training PID 396801 continues on the original pre-2024 data. The inference
extension preserves every original corpus row, column, and taxonomy entry and
appends only stored 2026 observed price, statement, and insider data in Polars.
It generates no 2026 Oracle/HITS targets. Saved checkpoint normalization and
label mappings are used for inference. There are 115 equities with 2026 prices:
113 through September 9 and BRK-A/BRK-B through September 8; historical-only ANTM
has no 2026 scores. No synthetic last-day prices are added by the extension;
the unchanged backtest engine retains its existing price forward-fill behavior.

Run artifacts, launch script, extension script, and coverage manifest are under
`artifacts/multirate_recovery/100B/`. Epoch output is under
`train_long_v6/epoch_validation_2024_2026/epoch_NNNN/{2024,2025,2026}/`.
All three years are now repeatedly inspected evaluation periods, not untouched
holdouts. A six-anchor 2026 inference smoke check used the existing $1T model
only to check dataset/checkpoint compatibility; its output is not a $100B result.

### $100B preparation cache locality repair

The first $100B launch was active on CPU but had not reached a training batch.
Training uses bounded streamed feature windows, but prepares all sample metadata
before fitting. Unordered Polars event anchors exceeded the 32-issuer metadata
cache: a reproduction on the pre-2024 supervised anchor index counted 13,380
cache misses in the first 20,000 records. There are 1,071,066 unique supervised
anchors before the remaining sample/taxonomy filters; this is not a completed
training sample count.

`context_ordered_anchors` now groups preparation by source issuer, instrument,
and date. This retains every anchor and the bounded cache while avoiding
repeated issuer metadata scans during preparation. The trainer reports progress
every 25,000 anchors. Feature windows still load lazily in Polars; the full
sample metadata list remains an upfront cost and is not yet an iterable dataset.
Twenty-five streaming/context contract checks pass, including a 114-issuer
cache-eviction regression. The preparation process was restarted before any
batch checkpoint existed; current PIDs and the restart record live in
`100B/run_status.json` and `100B/cache_locality_restart.json`. The corpus and all
three yearly epoch backtests remain in place.

### Sequence training with Polars (v7)

`--training-sequence-stride 128` replaces event-by-event supervised anchors with
bounded overlapping daily sequences. The daily window remains 252 observations:
128 new positions plus at least 124 context positions when history is available.
The Polars date-index audit matches all 1,071,066 supervised symbol/date pairs to
9,085 sequence documents, with no unmatched dates. Existing annual/irregular
regular documents remain for self-supervision, giving 12,744 pre-2024 training
documents and 100 packed batches at batch size 128 in the $100B benchmark.
Each supervised event belongs to exactly one sequence; context-only positions
remain unlabeled. A single bounded exact-date join supplies each sequence's
labels. No synthetic daily labels or pandas training dataframes are introduced.

Slow-rate and issuer contexts retain the configured historical observations at
the first supervised date plus every source update through the sequence end.
Variable context lengths are left-padded at the batch boundary. Timestamp masks
prevent later observations from reaching earlier predictions. Daily instrument
history uses overlapping 252-observation windows; the first supervised positions
have a shorter instrument-local prefix than a separately anchored 252-day window.
This is a sequence-training change, not numerically equivalent optimizer steps or
identical loss weighting to v6. All previous history remains eligible across the
corpus, and all supervised dates remain covered. Hierarchical NTP/MTP objectives
and Oracle/HITS input exclusions remain intact. As-of inference and the original
trading engine are unchanged.

Every completed epoch writes an immutable hard-linked checkpoint under
`epoch_checkpoints/`. The monitor consumes these in order, including after the
trainer exits, so slower evaluation cannot overwrite an unevaluated epoch.
The $100B sequence run uses `training_command_v7.json`, `training_v7.log`,
`train_sequence_v7/`, and the existing frozen 2024–2026 inference corpus.

The history-preserving benchmark completed its first two batches in 55.3 seconds,
projecting roughly 46 minutes per training epoch versus six hours for v6. This
is an early throughput estimate, not a completed-epoch measurement; per-epoch
2024–2026 inference/backtesting adds time. The 57 sequence, context, masking,
causality, and epoch-monitor checks passed.

### Synchronous epoch backtests and checkpoint continuation

The $100B v7 run now uses `--resume-training --checkpoint ...
--epoch-evaluation-dir ...` to restore model parameters, optimizer state, and the
saved epoch/batch cursor. `Trainer.fit` skips already completed batches using
the original seeded epoch ordering. The restart snapshot is epoch 4, batch 70;
the trainer first waits for reports through epoch 3, then completes epoch 4.
Every later epoch waits for both its evaluation metrics and backtest results
before the next epoch starts. The existing epoch-1 scoring process is retained
until completion, after which a supervisor replaces the monitor's training PID.

The restart uses the same corpus, model, Polars sequence pipeline, and original
2024/2025/2026 YTD strategy. Artifacts include `synchronous_resume_snapshot.json`,
`synchronous_resume_command.json`, and `restart_synchronous_training.py` in the
100B run root. `training_gate.json` reports the blocking evaluation epoch.

The older checkpoint did not record RNG state, so this continuation preserves
weights and optimizer updates but is not bit-for-bit identical to uninterrupted
stochastic training. New batch checkpoints include CPU/CUDA RNG states. The
resumed partial epoch's printed loss covers remaining batches only, and best-loss
selection restarts over the continued epochs. Progress speed and ETA now use
elapsed time within the current epoch, excluding waits for evaluation.
