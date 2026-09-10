# Individual-value reconstruction

The multi-rate trainer uses the `hierarchical_masks_v5` objective
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

Sparse windows reserve 16 observations per family, then union observations with
the same availability date. Frequent insider events cannot crowd out all older
HITS/Oracle observations. Target-derived context remains availability-gated;
HITS/Oracle context currently summarizes each year's events. Families or
instruments with fewer than two available observations cannot contribute NTP
pairs. Counts describe actual eligible targets, not guaranteed coverage.

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

The hierarchy is implemented, but held-out trading utility is unproven. The
highest-priority experiments are elapsed-time and forecast-horizon conditioning,
next-step persistence baselines per family, event-level HITS/Oracle context with
availability preserved, and supervised-only/NTP/MTP/combined ablations under the
same trading rules. The current family temporal encoder uses learned sequence
positions; timestamps impose causal visibility without encoding elapsed gaps.

Eleven issuers provide a useful pipeline trial, not evidence of broad issuer or
asset-class generalization. Extend both universe and held-out issuer coverage
before making that claim. Balance losses using measured per-family coverage and
validation results rather than assuming more reconstruction heads improve trading.
For a model frozen at the start of 2024, both 2024 and 2025 evaluations must fit
baseline statistics before 2024; `evaluate_predictions(training_cutoff=...)`
records and enforces that boundary separately from the evaluation interval.

The v5 hierarchy CUDA check completed 2,048 samples in one epoch with finite
loss (4.207217), nonzero gradients for every rate encoder and instrument fusion,
and nonzero valid targets for HITS, Oracle, and insider families in all four
reconstruction objectives. Peak process RSS was 3,342 MiB; peak CUDA allocation
was 8,121 MiB. Counts include overlapping training windows and are not counts of
unique financial events. These checks establish execution, not generalization.
