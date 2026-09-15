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
have finite bid > 0, ask > 0, and ask >= bid. Up to five calls and five puts per
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

The follow-up 50-option prototype adds optional
`MultiRateTransformerConfig.cross_instrument_attention`. After issuer/instrument
fusion, `InstrumentAttention` groups observed instrument tokens by issuer and
exact timestamp and applies a multi-head attention layer with a residual and
normalization. Attention spans instruments at that timestamp, not the flattened
year-long collection of tokens. Missing quotes are padded out. Supervised heads
retain separate per-instrument targets, and tests verify cross-instrument
gradients, issuer isolation, time causality, and permutation equivariance.
The training caller supplies explicit issuer IDs. This is still an experimental
model configuration: the production notebook and equity-first single-contract
inference path do not enable it. Deployment requires joint instrument batches
at inference as well as training.

The fresh 2023 AAPL benchmark used one equity and 50 unique surviving calls
(the configured hindsight filters left no puts). Six measured trials per variant
followed warmup, alternating variant order while the existing $10B job continued.
Median full-step times / peak allocated GPU memory were:

| Shared encoder | Instrument attention | Seconds | GiB |
| --- | --- | ---: | ---: |
| No | No | 2.500 | 32.13 |
| Yes | No | 1.843 | 24.15 |
| No | Yes | 2.468 | 32.15 |
| Yes | Yes | 2.148 | 24.18 |

With attention enabled on both sides, sharing reduced median step time by 13%
and peak allocated GPU memory by 25%. Concurrent GPU work makes timing noisy;
these are batch measurements, not epoch-time or predictive-quality results.
Evidence and the executable recipe are in
`artifacts/multirate_recovery/shared_issuer_attention_50/benchmark.json`,
`benchmark.log`, and `benchmark.py`.
