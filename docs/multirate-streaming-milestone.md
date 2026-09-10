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

The trading objective is to learn each instrument's Oracle actions and HITS
scores conditional on issuer state, instrument history, and asset class. These
are the primary supervised targets. Masked-token and next-observation tasks
support representation learning. Issuer identity is metadata; predicting its
identity does not establish trading skill.

The trainer already applies Oracle binary classification losses and HITS
regression losses to the fused instrument representation. Tests demonstrate
that an instrument loss reaches the issuer encoders. The recorded smoke run
contains supervised observations for equity and synthetic option baskets only;
nonzero gradients establish an optimization path, not useful generalization or
optimal trading. Oracle optimality is relative to the label generator's
execution assumptions and constraints.

Use warehouse-generated HITS and Oracle labels from each instrument's own
history, with event-only supervision and label availability before the training
cutoff. Do not copy equity labels onto debt, preferred shares, or options. A new
utility head or common fixed-horizon target is not required by this objective.
Comparing predicted HITS scores across instruments still requires checking the
label generator's normalization and graph scope. Full-calendar scoring and
strategy evaluation remain separate from supervised event rows.

The next evidence needed is a multi-issuer, multi-asset chronological run with
per-class Oracle/HITS evaluation and issuer-context ablation. Compare the full
model with an otherwise matched model trained without issuer context; merely
observing gradients cannot show that issuer information improves predictions.
Actual debt/preferred instruments need their own histories and terms; adjusted
price paths alone do not validate complete coupon, redemption, credit, or
execution economics. Keep data assembly in bounded Polars partitions and issuer
context reuse inside a gradient-preserving step.

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
