# On-demand equity and sampled-option training

The [step-by-step notebook](../notebooks/multirate_warehouse_training.ipynb)
explains documents, subtokens, annual memory, objectives, coverage, and backtests.
Set `MIN_MARKET_CAP` in dollars (for example, `100_000_000_000`), then Run All. Output labels such as `100B` are derived automatically. EDA and training
share this setting. The notebook launches a fresh run and updates result tables
from live process events as each backtest book completes. It does not discover
old runs or read JSON artifacts for display. Saved historical outputs are cleared;
changing settings blocks reuse of in-memory results from a previous run.

New multi-rate runs use `scripts/train_multirate_mtl.py --min-market-cap`.
The trainer queries the warehouse universe and assembles annual documents as
the optimizer requests them. It never reads a previous run's corpus, roster,
feature exports, event exports, or normalization statistics. A field-name-only
schema is packaged with the model code. A fixed `sign(x) * log1p(abs(x)) / 10`
transform makes training and inference consistent without a whole-dataset
normalization pass. Nonfinite source values remain missing.

```bash
python scripts/train_multirate_mtl.py \
  --min-market-cap 1000000000000 \
  --output-dir artifacts/multirate_recovery/1T/new_run \
  --epochs 1 --batch-size 16 --d-model 64 --num-heads 4 --layers 2 \
  --mrl-dimensions '' --disable-document-tasks --device cuda \
  --sequence-mode annual_memory --self-supervision both \
  --train-end-date 2024-01-01 \
  --prediction-start-date 2024-01-01 --prediction-end-date 2026-09-09 \
  --checkpoint-every-batches 10 --progress-updates-per-epoch 10 \
  --skip-embeddings --skip-t-sne
```

An existing output directory is rejected. `--corpus` is retained only for
recorded checkpoint evaluation and exact continuation of older runs, not as
an input to new training. Streaming checkpoints have a distinct normalization
contract and cannot be loaded by the older corpus inference path.

The notebook sets `--warehouse-option-start-date 2021-01-01` independently of
`--warehouse-start-date 1900-01-01`. This preserves earlier FMP equity and
financial history while excluding sparse pre-2021 option history. The option
start must be January 1 before the training cutoff. Omitting the option flag
uses the warehouse start date.

## Inputs and annual documents

`research_tools/warehouse_multirate.py` reads adjusted equity prices, warehouse
financial feature families, macro and peer context, and disclosure-dated issuer
events. Source tables use a bounded issuer cache; one CPU batch is prefetched
while the GPU runs. The shared token/subtoken objectives live in
`multirate_training_step.py`. Native observations retain their dates; no daily
filler labels are added. Equity documents preserve chronological annual memory.
Each sampled option contract has a distinct instrument identity.

Options are discovered separately for every equity symbol, including separate
share classes. `universe.json` records equities, stored options availability, and
excluded equities without enough pre-cutoff price history. `option_coverage.json`
records source dates by year, first-session failures, selected contracts, training
price observations, and documents containing multiple price observations.
An epoch fails if a filter-eligible option underlying was omitted. Zero-survivor
underlying/years are recorded separately from missing source history.

`sampled_options.py` applies same-year expiration, positive terminal moneyness,
positive first-ask/last-bid profit, at least 20 valid quote days and 80% coverage,
and the median profit separately for each underlying’s calls and puts. Valid quotes
have finite bid > 0, ask > 0, and ask >= bid. Up to the configured number of calls and puts each per
underlying are sampled independently with seed 0. Fewer survivors stay fewer.
Each real contract has its own identity and historical series; strikes are never
averaged. Both training and backtests intentionally use this hindsight-selected
fixed annual universe. Uncompleted expirations cannot pass outcome filters.

Training finishes one issuer’s chronological documents before preparing the next
issuer’s option histories. Selection is computed only for the requested
underlying/year, with per-underlying/right percentiles.
`sampled_contracts/<year>/<symbol>/` stores run-local audits, selections, and
selected price paths. No prior run’s selection is reused. Shared equity-price
and market-context initialization still covers the selected universe.
Forward splits conserve exposure per original contract unit. Expiration settlement
uses exact-session underlying marks and intrinsic value. Unsupported splits and
unavailable outcomes are explicitly excluded in coverage records.

Oracle trades and return/speed HITS graphs are computed independently from each
instrument's own prices, including each call and put. Issuer information is context,
not a substitute target price series. The final supervised fusion order is annual,
quarterly, issuer daily, sparse, instrument. Its changed dimensions require fresh
training; old checkpoints cannot be loaded into this architecture.

## Backtests and artifacts

After an epoch, the model scores the requested years from empty memory, without
replaying training history as an inference warmup. Coverage checks compare
predictions to actual priced dates.

Equity reports use the existing HITS policy and shared-book return engine with
separate long-only and short-only books and next-session signal execution.
Equity entries trigger option selection from the annual hindsight-filtered pool
of up to five calls/five puts per symbol. The model ranks eligible contracts by
mean predicted own-option long-return HITS hub/authority, using the prior observed
session. Buy one contract identity at ask; no daily full-chain scoring occurs.

The selected option's own Oracle predictions control exit: buy <= short or sell
>= 0.5 triggers sale at the following observed bid. Both calls and puts are held
long. Equity exits do not close options. Only held contracts require daily option
inference, without warmup. A bounded 16-document run-local raw-tensor cache
avoids rebuilding feature tables each day; day extraction excludes all other
dates, and no model predictions or learned state are cached. Expiration forces intrinsic settlement; no roll is
performed. Per-trade audits record ranking scores, exit reasons and skips.

Option replay uses whole contracts, available cash, bid/ask spreads and 5.5 bps
fees. Missing quotes carry forward only for valuation. Entered contracts without
an exit observation fail explicitly. Each year produces four independent books;
the live notebook reports 2024, 2025 and 2026 through its configured endpoint.

Financial series retain the warehouse's recorded observation dates. No reporting
lag or historical data-vintage reconstruction is applied; these reports use that
research-data timing convention.

Runs write configuration, startup timing, status, universe and options coverage,
checkpoints with objective-observation counts, and `epoch_validation/epoch_XXXX/`
with predictions, prices, yearly equity/option reports, and timing. Training
does not write or read a materialized corpus. Final prediction and price files
are backtest artifacts.

The September 12 $1T validation reached its first optimizer update in 27.46
seconds, with equities and options both present. It selected 13 equities and
11 underlyings with stored pre-2024 options; neither Berkshire share class had
option series stored in the warehouse. The completed epoch took 550.50 seconds
for 371 equity and 790 option documents, including 51,218 option price
observations. Scoring took 144.43 seconds and the 12 backtest reports took
1.74 seconds. Results and coverage are stored under
`artifacts/multirate_recovery/1T/warehouse_stream_20260912T201813Z/`.

## Sampled-contract correctness run

`artifacts/multirate_recovery/1T/sampled_contract_smoke_20260914` completed one
2023 training epoch and all four 2024 backtest books with a small 8-dimensional,
one-layer model. It trained 13 equity and 60 individual-option documents. The
checkpoint records 213 observations for each option Oracle channel and 2,660
for each option return/speed HITS channel. Inference covered 9,944 priced dates
across the equities and the fixed selection of 50 calls and 10 puts.

The first optimizer update took 23.25 seconds; total elapsed time was 78.10
seconds, including 31.86 seconds for inference/year selection and 0.55 seconds
for portfolio backtests. These timings validate the small configuration, not
the notebook's full-size default model. Its reports and actual configuration remain in that run's artifact directory;
the notebook now displays only results produced by its current execution. The option results explicitly use
hindsight selection and are not an unbiased out-of-sample performance estimate.

September 14 option-model selection/exit validation:
`artifacts/multirate_recovery/1T/warehouse_stream_20260914T194642Z_9ed990f2`
was launched by the notebook's actual live runner. Its bounded configuration used
2023 training history, 8 model dimensions, one layer, batch size 2, and one epoch.
It trained 13 equity and 60 option documents, then completed all 12 reports for
2024–2026 through September 9. First optimizer update: 19.79 seconds; total:
304.79 seconds; equity inference: 16.11 seconds; backtests including annual
selection and option inference: 234.43 seconds. The 55 option trades used 4,195
option queries and at most five entry candidates, with whole units and
nonnegative cash. This small model held every option until expiration; daily
Oracle exits are implemented and their next-quote execution is exercised by
regression tests. The executed notebook is saved as `executed_validation.ipynb`
in that run. This validates the workflow, not the predictive quality or runtime
of the notebook's default $100B, 64-dimensional full-history configuration.

Training progress is capped by `--progress-updates-per-epoch` (1–10, default 10), exposed as `PROGRESS_UPDATES_PER_EPOCH` in the notebook’s top cell. It reserves one update for completion and spaces earlier updates using a document-count upper bound from already loaded metadata. No corpus counting pass is needed; epochs with fewer surviving options can produce fewer updates.

Issuer-sequential training preserves cross-year learned memory. Batches contain
only instruments of the current issuer; older equity-only years can form
single-document batches. Raw source features are released after that issuer.
The earlier validation timings above precede this scheduling/percentile change.

Issuer-sequential smoke validation:
`artifacts/multirate_recovery/1T/issuer_sequential_validation_20260914` completed
one fresh 2023 training epoch (13 equity and 60 option documents) and all four
2024 backtest books. The small 8-dimensional, one-layer model reached its first
optimizer update in 12.08 seconds and completed in 138.19 seconds. Training
updates were interspersed with issuer-specific audits. The scheduling regression
verifies that another issuer's option preparation cannot start until the current
issuer's training batches have been consumed. This is a bounded smoke test,
not a full-history $100B timing estimate.

Experimental shared issuer encoding (September 15):
`MultiRateTransformerConfig.share_recurrent_issuer_context` defaults to false.
When enabled, the training step shares matching annual/quarterly projections and
encodings, plus issuer daily/sparse encodings, within one optimizer step. Input
values, dates, masks, modalities, family presence, and incoming recurrent memory
must match; different instrument memories stay separate. Gathered outputs retain
gradients from every instrument. Dropout is shared by grouped instruments, so
output/gradient equivalence tests disable dropout. Nothing is cached across updates.

The bounded experiment in `artifacts/multirate_recovery/shared_issuer_benchmark_v2`
uses freshly read AAPL 2023 equity and five sampled options, model width 64 with
two layers, and full supervised plus reconstruction training steps. It gives all
instruments the equity document's issuer daily context, including the equity
calendar; the production adapter still creates contract-specific calendar values.
Across eight timed trials per variant after warmup, median step time was 0.480 s
without sharing and 0.448 s with sharing (1.07x). Another training process was
using the GPU, so these noisy batch timings are not a full-epoch speed claim.
Most historical equity-only batches have no cross-instrument reuse. The notebook
default and existing running jobs remain unchanged. `benchmark.py`,
`benchmark.log`, and `benchmark.json` preserve the experiment recipe and evidence.

Compact family attention (September 15):
`auto_features.family_temporal_attention` projects with the existing
`MultiheadAttention` parameters and uses native PyTorch SDPA. A boolean
`[instrument * family, 1, date, date]` mask broadcasts over heads, replacing
the repeated per-head date mask and its expanded floating-point padding merge.
Each instrument/family retains its own sequence; there is no attention between
instruments. Same-date visibility, causal history, padding, task heads,
recurrent state, and checkpoint parameter names are preserved. This optimization
is automatic in the existing model and does not require enabling issuer sharing.
CPU and CUDA tests compare outputs and input/parameter gradients against native
MultiheadAttention and check future-date and instrument isolation.

A CUDA float32 attention-only forward/backward benchmark on GB10 with 200
instruments, eight families, 253 positions, width 64 and four heads measured
0.189 s before versus 0.089 s after, with peak allocated memory 5.92 versus
1.82 GiB. These are component timings, not full-model or full-epoch timings.

The full-step benchmark in
`artifacts/multirate_recovery/compact_instrument_benchmark` uses fresh AAPL 2023
warehouse documents and compares the old and compact attention on identical
weights, random seeds, batches and objectives. Both variants enable experimental
issuer sharing and use the underlying equity calendar for issuer daily context.
Four measured repetitions per variant follow warmup, with alternating order:

| Options plus one equity | Old step | Compact step | Old peak GPU allocation | Compact peak GPU allocation |
| --- | --- | --- | --- | --- |
| 50 | 1.645 s | 1.255 s | 24.15 GiB | 15.71 GiB |
| 100 | 3.134 s | 2.324 s | 47.54 GiB | 30.80 GiB |

The 100-option step uses 26% less time and 35% less allocated GPU memory.
These measurements include the supervised and reconstruction forward passes,
backward, clipping and AdamW update, but exclude warehouse preparation and
backtesting. They do not establish a full-epoch speedup. Only the compact
attention change is enabled automatically; issuer sharing and the benchmark's
larger option sample counts are not notebook defaults. Existing Python kernels
must restart to load the changed model code.

The larger capacity check in
`artifacts/multirate_recovery/compact_instrument_benchmark_200` rebuilt the
2023 AAPL and MSFT documents from the warehouse and trained 200 distinct real
options plus two equities in one batch. With the compact encoder and the same
experimental issuer sharing, three measured full steps after warmup had median
5.187 s and peak allocation 62.18 GiB. Both calls and puts were eligible under
the existing filters; the surviving pools for these two issuer-years were calls.
The old implementation was not benchmarked at 200 options; before/after
comparisons above are limited to 50 and 100 options. This is a capacity and
throughput check, not a completed training epoch or a new backtest result.

The training notebook forwards `OPTIONS_PER_SIDE` as `--options-per-side` (CLI default: 5). This nonnegative integer controls EDA and the per-underlying/year sample used by training and option backtests. Run configuration and selection audits record the chosen limit; fewer surviving contracts remain fewer. Other EDA filter settings do not override the trainer policy.

Set `OPTIONS_PER_SIDE = 0` in the notebook (CLI: `--options-per-side 0`) for equities-only training and evaluation. All option EDA cells, option warehouse discovery, sampling, and option backtests are skipped; completion requires only the long/short equity reports. Negative values are rejected.

Equities-only warehouse training (`OPTIONS_PER_SIDE = 0`) uses `equity_training_batches` in `research_tools/warehouse_multirate_training.py` to interleave independent equities up to `BATCH_SIZE`, preserving each equity’s chronological annual memory. Only metadata is scheduled ahead; raw documents are assembled with one CPU batch prefetched. The raw issuer-source cache holds at least one configured batch. Options-enabled runs retain issuer-sequential scheduling. Progress reports `batch_documents` and `active_issuers`; this changes optimizer grouping, so losses and weights need not match the old single-document schedule.


## Document preparation measurement (2026-09-16)

Profiling three warmed AAPL annual documents found 0.997 of 1.945 seconds in
`merge_observations`, including repeated wide-schema lookups. Preparation now
checks column membership once per merge and avoids rebuilding identical
issuer-daily and issuer-sparse tensors for equities. Option daily and issuer
daily remain distinct. Raw tensor aliases are read-only; `BatchTensors` stages
fields separately before training masks are applied.

A same-process alternating comparison on four AAPL years (2020–2023), with
POLARS_MAX_THREADS=8 and OMP_NUM_THREADS=4, measured baseline totals of
1.894/1.849/2.335 seconds and optimized totals of 1.254/1.306/1.074 seconds.
Median document preparation improved 1.51x. All sample tensors matched exactly,
including labels, masks, timestamps, and prices. This measures warmed preparation
for one issuer, not a full training epoch or a GPU speedup; the live run continued
while the comparison ran. No current-run process was interrupted.

Progress now records cumulative batch-wait, training-step wall, and checkpoint
seconds. Batch wait includes initial document setup; preparation overlapped with
GPU work is not counted as waiting. Periodic checkpoint time appears in the next
progress event. Training-step wall time includes staging, forward/backward, and
optimizer work; it is not a synchronized CUDA-kernel breakdown.

## Full equity epoch timing

Equities-only preparation keeps absent fields out of intermediate Polars frames
and scatters observed fields into the unchanged full-width tensors. Annual and
quarterly issuer families are merged once per raw-source cache entry; daily
family calendars remain separate for peer as-of joins. Peer frames are cached
within the run by sector/industry. Oracle single-k targets use quant-warehouse's
Numba solver with the original scalar recurrence and tie breaks.

`scripts/benchmark_warehouse_epoch.py --configuration <run>/configuration.json
--output-dir <new-run>` reproduces the full training configuration from fresh
weights and warehouse reads. It measures through the completed epoch checkpoint,
then stops at the evaluation boundary. It never loads the old run's corpus or
checkpoint. `epoch_benchmark.json` verifies the complete equity-document count
and reports both epoch time and wall time including setup; backtests are excluded.

Equities-only batches use four preparation workers after serially warming the
batch's source and peer caches. Only immutable frames are shared; ordered map
preserves document order, and at most one complete CPU batch is prefetched.
A controlled 24-document AAPL benchmark measured 5.97/11.08/14.11/11.86 documents
per second for 1/2/4/8 workers, respectively, with every output tensor equal.
This is a preparation benchmark; the full-epoch report is the runtime evidence.


### Verified full $10B equities-only epoch

On September 16, 2026, the complete 840-equity universe trained all 24,706 annual
documents in 424 optimizer batches. Total wall time, including source preparation,
model setup, and the completed epoch checkpoint, was **3,000.77 seconds (50m 1s)**.
The epoch timer was 2,994.66 seconds; the first optimizer update arrived at 513.51
seconds. Backtests were excluded by the benchmark's evaluation-boundary hook.

The run retained all available FMP history from the 1900 warehouse floor through
the exclusive 2024-01-01 cutoff, CUDA FP32, batch size 64, model width 64, four
attention heads, two layers, all existing supervised heads, both self-supervised
objectives, and checkpoints every ten batches. POLARS_MAX_THREADS=8,
OMP_NUM_THREADS=4, and four ordered preparation workers were used on NVIDIA GB10.
No previously built corpus or training checkpoint was loaded.

The final checkpoint reports epoch completion, contains 973 finite parameter
tensors, and accompanies the complete document-count assertion. The ten-batch
serial/parallel comparison had a loss difference of 9.54e-8 and maximum parameter
absolute difference of 1.31e-5; training weights are not claimed bitwise identical.
The measured report is [stored here](benchmarks/10b-equity-epoch-20260916.json).
Local run artifacts are under
`artifacts/multirate_recovery/10B/equity_epoch_performance_20260916_v3/`.

Use quant-warehouse main at commit `8ad61b7` or later and the optimized orchestrator
path introduced in `ee2586d`. An already running training process must be restarted
to pick up the new implementation. The benchmark retains a completed checkpoint
but intentionally does not produce evaluation or backtest reports.

## Backtest a completed warehouse checkpoint

Use `evaluate_warehouse_checkpoint` from
`quant_orchestrator.research_tools.warehouse_multirate_training` to score and
backtest saved weights without another training epoch:

```python
from quant_orchestrator.research_tools.warehouse_multirate_training import evaluate_warehouse_checkpoint

reports = evaluate_warehouse_checkpoint(
    "artifacts/multirate_recovery/10B/<run>/epoch_0001.pt",
    "artifacts/multirate_recovery/10B/<run>/saved_epoch_backtest",
    device="cuda",
)
```

The output directory must be new. Keep the original `universe.json` beside the
checkpoint. The evaluator restores the saved architecture and settings, checks
the feature schema, normalization, document contract and equity universe, then
uses the same evaluation function as training. It performs no optimizer steps
and initializes inference memory empty, just as normal epoch evaluation does.
Warehouse data is read fresh, so this is not a frozen historical data snapshot.
The checkpoint hash, status, predictions, prices, yearly long/short reports and
trade artifacts are saved in the evaluation directory. An equities-only
checkpoint skips option evaluation. Reports use the saved prediction dates;
calendar years are separate books rather than one compounded portfolio.

Equity evaluation now uses the same raw-source warming and one-batch prefetch
helpers as training, with four tensor-preparation workers. It holds one bounded
block of issuers across prediction years before moving to the next block, avoiding
annual reloads of the same issuer history. Annual memory remains isolated by
instrument and chronological within each instrument. Score export converts whole
head tensors to Python arrays instead of extracting each scalar separately.
Inference logs separate cumulative batch preparation wait from prediction time;
portfolio simulation remains the existing shared-book engine. The earlier
completed-checkpoint backtest used the previous serial evaluation path; its wall
time is not a benchmark of these new evaluation changes.

Completed saved-epoch evaluation (September 16): 537,722 daily equity scores
took 2,041.00 seconds, followed by 7.55 seconds for the six shared-book reports.
Net capital returns for long/short books were +28.38%/-30.96% in 2024,
+24.34%/-20.75% in 2025, and +9.21%/-7.38% through September 9, 2026.
Each book starts with $100,000 and uses next-session execution and 5.5 bps
modeled costs. These are research results on the selected warehouse universe,
not a point-in-time universe reconstruction. This run precedes the new inference
scheduler. Full metrics and checkpoint provenance:
[`benchmarks/10b-equity-backtest-20260916.json`](benchmarks/10b-equity-backtest-20260916.json).

## All-history latest-date deployment workflow

`research_tools.warehouse_live.train_latest_warehouse_model(output_dir,
min_market_cap=..., ...)` drives the `optimal_trader` multirate live notebook.
It discovers the latest finite positive stored equity close in the selected US
NASDAQ/NYSE stock universe, reads history from 1900-01-01, and sets the exclusive
training cutoff to the following day. The current partial year is included in
the annual scheduler and supervised labels. Standard historical backtests still
require a January 1 cutoff before their evaluation period; this deployment mode
is explicitly separate and writes `evaluation_mode=latest_date_in_sample`.

The model architecture, objectives, optimizer, source preparation and training
loop are shared with warehouse research. Options remain disabled. After the last
epoch, the live path reuses the model and stream and calls the shared optimized
equity inference scheduler for the current year. Only the latest date is exported;
symbols missing its prices are reported. There is no training-history inference
replay or historical backtest. Artifacts include `latest_predictions.parquet`,
`latest_prices.parquet`, `latest_prediction_summary.json`, and the normal epoch
checkpoints. The live function never refreshes vendor data or connects to brokers.
