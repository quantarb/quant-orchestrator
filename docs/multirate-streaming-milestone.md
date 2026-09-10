# Issuer and instrument streaming milestone

The canonical trainer is `scripts/train_multirate_mtl.py` in quant-orchestrator.
It uses the existing PyTorch model and warehouse-prepared Parquet corpora. The
custom code handles native-frequency windows, shared issuer context, and task
routing; PyTorch supplies attention, automatic differentiation, and optimization.

## Implemented contract

- Annual, quarterly, daily, and irregular observations retain per-row dates.
  Attention accepts different date vectors for different batch members.
- Annual and quarterly contexts come from the instrument's underlying issuer.
  Additional issuer daily and irregular contexts remain separate from the
  instrument's own daily and irregular observations. Asset-class adapters act
  on instrument inputs; task heads consume the fused instrument representation.
- Supervised instrument losses reach the rate encoders. Fusion pools only
  context states dated at or before each instrument observation.
- Shared annual/quarterly windows are projected and encoded once within a
  batch. Shared issuer daily/irregular contexts are likewise encoded once.
  Shared corruption patterns preserve the context identity during masking.
- Next-observation targets skip absent families and same-date observations.
  Masked-token objectives hide complete observed tokens before projection.
  Self-supervised predictions consume local states, not unmasked issuer fusion.
- Persistent rate caches are inference-only. They retain subtoken states and
  reject changed inputs/windows or changed model parameters. Training reuse
  retains the autograd graph within one forward/backward step.
- Instrument taxonomy requires an explicit `asset_class` for every symbol.
  Issuer linkage does not imply an option: `equity`, `option`, `note_bond`,
  `preferred`, and other supplied classes have separate adapters and supervision
  coverage checks. This is schema support, not evidence of training on every class.

## Instrument selection objective

The user's intended decision is to select the appropriate instrument given an
issuer's state at a date. Issuer identity is reference metadata; predicting its
identity does not establish instrument-selection skill. The current independent
Oracle/HITS predictions and self-supervised tasks are auxiliary foundations;
there is no implemented cross-instrument selection objective yet.

The next training change needs issuer/date candidate groups, an instrument
utility head, and a comparison loss within each group. Warehouse-prepared
outcomes must use a declared common horizon, capital convention, and outcome
criterion. The criterion (raw return, risk-adjusted return, or a mandate) still
needs to be settled. Do not silently compare asset-specific Oracle/HITS labels
as if they were interchangeable utilities. Candidate eligibility must use only
information available at the decision date, and label availability must precede
the training cutoff. Event-only selection groups must remain separate from the
full-calendar scoring universe.

Chronological evaluation should measure the chosen instrument's realized utility
and regret versus eligible alternatives, alongside per-asset coverage and
simple selection baselines. Actual debt/preferred instruments need their own
histories and terms; adjusted price paths alone do not validate complete coupon,
redemption, credit, or execution economics. Keep group assembly in bounded
Polars partitions and issuer-context reuse inside a gradient-preserving step.

## Memory contract

`StreamingContext` keeps lazy Polars scans and collects bounded windows only.
Independent endpoint rows on the same symbol/date are coalesced without an
inner join. `StreamingSupervision` keeps label filtering and aggregation lazy;
only a requested instrument/event date is collected for a sample. Supervised
labels remain event-only; scoring uses the requested full observation calendar.

No Pandas or NumPy bridge is used in the canonical corpus-to-training path.
Sample arrays are released after each batch. A bounded LRU retains reusable raw
windows; `--context-cache-size 0` disables it. Prediction CSVs are written one
batch at a time. Embedding accumulation is disabled. Compact sample/taxonomy
metadata still resides in memory; this is not a constant-memory iterator over
an unlimited number of instruments and dates.

The default physical batch size is four. Use gradient accumulation to increase
the effective batch size. GPU memory also depends on family count, sequence
length, and model dimensions. Lazy file scans do not bound attention memory.

Prepare option observations and targets in the corpus upstream. In-memory
`--option-panel` assembly is rejected. Use an explicit `--train-end-date`;
fraction-based validation is rejected because it does not define the fitting
cutoff early enough for preprocessing. Normalization and label vocabularies fit
training data only and are restored from checkpoints. Symbol restrictions also
apply to normalization, including the underlying issuer of training instruments.

## Reproduction

The stored real-data check uses one issuer (AAPL), its equity, and synthetic
option baskets in the existing `contract_check_v2` corpus. It is a pipeline check,
not a promoted experiment baseline or evidence of predictive performance.
Its original taxonomy predates mandatory `asset_class` metadata. Before rerunning
these historical commands, annotate the known equity and synthetic option rows
explicitly; the trainer now rejects untyped corpora rather than guessing security
types from their issuer link. The old Pandas corpus builder is not the streaming
assembly path and still needs replacement.

```bash
python scripts/train_multirate_mtl.py \
  --corpus artifacts/multirate_recovery/contract_check_v2 \
  --output-dir artifacts/multirate_recovery/issuer_instrument/final \
  --epochs 1 --batch-size 4 --d-model 16 --num-heads 2 --layers 1 \
  --mrl-dimensions '' --skip-embeddings --skip-t-sne \
  --train-end-date 2024-01-01 --prediction-start-date 2024-01-01 \
  --country '' --currency '' --exchanges '' --device cuda --max-samples 96

python scripts/train_multirate_mtl.py \
  --corpus artifacts/multirate_recovery/contract_check_v2 \
  --output-dir artifacts/multirate_recovery/issuer_instrument/final_holdout \
  --checkpoint artifacts/multirate_recovery/issuer_instrument/final/multirate_mtl_model.pt \
  --inference-only --batch-size 4 --mrl-dimensions '' \
  --skip-embeddings --skip-t-sne --prediction-start-date 2024-01-02 \
  --prediction-end-date 2024-01-05 \
  --country '' --currency '' --exchanges '' --device cuda
```

The smoke cap retains rare Oracle events and final historical contexts.
`training_diagnostics.json` records observed task counts by asset class, loss
sums, gradient magnitudes, process RSS, and peak CUDA allocation. Missing required
Oracle/HITS supervision for an asset class fails the run. Regression scores are
exported on their trained scale; classification scores use sigmoid probabilities.

## Limits and remaining experiments

The available small corpus has only one sparse-event availability date before
the training cutoff. It cannot demonstrate next-sparse learning from multiple
historical dates; that behavior is covered by regression tests. It has no bonds
and only one issuer. Broader real-data validation, more irregular source history,
frozen-encoder training/cache benchmarks, and full WFO comparisons remain
separate experiments. Existing historical checkpoints and the live notebook
have not been promoted or overwritten by this milestone.

## Recorded verification on 2026-09-10

The final CUDA check completed 96 training samples in 29 batches (one epoch,
about 56 seconds for optimizer steps). Peak process RSS was 3,129 MiB and peak
CUDA allocation was 6,852 MiB. These are measurements for this small configuration,
not a bound for larger models or corpora.

All four Oracle heads received four equity labels and 35 option labels each;
HITS heads also received labels in both asset classes. Four sparse masked-family
observations contributed; next-sparse counts were zero for the coverage reason
above. `final_holdout` reloaded the checkpoint and scored all 24 eligible rows
across 11 instruments on January 2–5, 2024, with zero missing expected rows.
Exact-event-date evaluation is recorded in `holdout_validation.json`; those few
observations do not establish predictive quality.
