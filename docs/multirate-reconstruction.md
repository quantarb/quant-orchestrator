# Individual-value reconstruction

The multi-rate trainer uses the `family_temporal_cross_feature_v2` objective
contract. Oracle/HITS remain the instrument-specific supervised tasks.

Each token head predicts the complete ordered numeric feature vector for its
rate. Each subtoken head predicts the channels of one feature family. Unequal
family widths are padded only in the output layout; those padded channels never
contribute to the loss. Values are normalized using the training cutoff and
training universe. Neither token nor subtoken targets average unrelated values.

For next-subtoken prediction, each channel targets its next observed value on a
strictly later date. For next-token prediction, the target is the vector from
the next dated observation, with that observation's missingness mask. It does
not combine channels from different future dates. Same-date rows cannot serve
as future targets.

Masking selects whole observations with probability 10%. In remaining rows it
selects whole feature families with probability 10%, and then individual values
in unselected families with probability 10%. This hides approximately 27.1% of
available values overall. The three disjoint modes are recorded separately in
training diagnostics. Original missing values, padding, and visible values do
not contribute to masked losses. Shared annual/quarterly contexts receive
identical corruption within each batch.

The model preserves original family presence for deliberately hidden values.
Next-subtoken heads receive only their own family's causal temporal representation.
Masked-subtoken heads also receive encoded rate context, enabling inference
from other feature families at the same date and prior dates. Token heads
receive encoded rate context. Whole-observation queries can therefore
use historical context, and individual-value queries can use the other visible
features at their date. Future dates remain blocked. Reconstruction losses use
per-value MSE, averaged over valid values for each task.

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
coincide, missing-channel successor alignment, both masking modes, padding
exclusion, and dependence on past observations without future access for every
rate. The broader multi-rate, supervision, corpus, and replay selection passes
84 tests. Run the real-data CUDA/checkpoint checks before claiming a training
run is ready; unit tests alone do not establish model quality.
