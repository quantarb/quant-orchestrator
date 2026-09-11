# Causal multi-date document training and scoring

Fresh training defaults to `--sequence-mode documents`. The contract is
`calendar_quarter_prefix_v1`; checkpoint inference restores the recorded mode and
rejects an explicitly incompatible mode. Existing rolling checkpoints remain
rolling checkpoints. Their weights are not relabeled as document-trained models.

`research_tools/document_sequences.py` partitions compact native observation date
indexes into fixed calendar quarters. Training keeps all pre-cutoff quarters;
inference keeps every requested daily date. The feature matrices stay in the
bounded, disk-backed Polars/Torch context indexes. This restores the older
multi-date-per-forward approach without restoring its pandas corpus construction.

Each stream retains its configured history at quarter start and every subsequent
update in that quarter. Daily and issuer-daily streams retain 252 start-history
observations; annual and quarterly retain 16 and 40; sparse streams retain their
existing per-family history. A permanent first placeholder and right padding
keep observed token positions stable as the document grows. No stream is trimmed
to make a batch fit: positional overflow raises an error. The current EOD layout
fits the 512-position capacity. All rate fusion and issuer context use actual
per-token timestamps. A later report or disclosure cannot affect an earlier
prediction through that fusion.

Supervised losses apply to actual labeled dates owned by the quarter. Historical
warmup positions have no supervised labels, and missing labels are not fabricated
negatives. Congressional/insider transaction-date targets and disclosure-date
inputs are unchanged. Oracle and HITS remain targets only. All feature/subtoken/
token NTP and MTP objectives remain present, with the existing within-family and
whole-family masking contracts. NTP audits retain every unique eligible pair for
this document layout; their context differs from rolling-window audits.

Scoring exports every owned daily token with `information_date` equal to that
token's date. Full-calendar inference requires one finite score row for every
expected symbol/date, with no omissions, extras or duplicates. As-of inference
inside a quarter uses the same history origin and positions as that prefix in a
completed quarter. These checks establish causality relative to the supplied
corpus timestamps; existing source-vintage limitations are not resolved by this
change.

## Executed validation, September 11, 2026

- 86 targeted tests passed, including real-date ownership, native history retention,
  right padding, supervised date placement, future-observation isolation, NTP/MTP,
  issuer grouping, score coverage and the existing yearly portfolio strategy.
- The $1T smoke performed three optimizer steps. January's 371 score rows from
  15 instruments were compared with January inside a complete first-quarter pass.
  All 26 supervised heads agreed within 8.35e-7. The quarter exported 1,081 rows;
  both coverage checks found zero missing dates, duplicates or nonfinite scores.
- The $100B smoke performed three optimizer steps and scored the entire intended
  2024-01-02 through 2026-09-09 calendar. Its 96,181 rows exactly match the previous
  scoring universe/calendar. Input layouts match: 1,383 numeric fields, 45 grouped
  numeric adapters (including the 40 restored families), 15 sparse input families,
  and 42 tasks. The corpus's 17 sparse families include the two target-only
  Oracle/HITS families; they were not lost or reintroduced as inputs.

| Full $100B evaluation | Previous rolling run | Document smoke |
|---|---:|---:|
| Forward-pass documents/windows | 96,181 | 1,292 |
| Exported daily predictions | 96,181 | 96,181 |
| Inference subprocess, including NTP audit | 2,435.86 s | 66.35 s |
| Six yearly long/short portfolio books | Completed separately | 2.84 s |

Document inference was 36.71x faster. Total document inference plus portfolio
replay took 69.19 seconds. This is a complete-calendar timing/coverage benchmark,
not a forecast-quality comparison: the document checkpoint had only three
optimizer steps, while the rolling checkpoint had completed one epoch. Predictions
between the two models are not expected to match. See
`notebooks/multirate_document_scoring.ipynb` and
`artifacts/multirate_recovery/100B/document_run_comparison_v10.json`.

## Active run

`artifacts/multirate_recovery/100B/train_documents_v10` starts from fresh weights
and optimizer state: 12 epochs, batch 16, FP32 CUDA, all 48,142 pre-2024 quarterly
documents, and the unchanged expanded corpus. This includes more training
quarter-documents than the earlier event-strided sampler; evaluation speed is not
an assertion that an entire training epoch is shorter. The supervisor runs the
original adjusted-price anchored-HITS books for 2024, 2025 and 2026 through
September 9 after each immutable epoch checkpoint, then releases the next epoch.
The monitor can use evaluation batches of 64 when GPU headroom permits.

The previous rolling evaluation finished and its results are retained. A missing
price lookup under ANTM was resolved with an explicit dated ANTM-to-ELV price
mapping, supported by the issuer's
[June 28, 2022 announcement](https://www.elevancehealth.com/newsroom/elevance-health-rings-in-rebrand-with-nyse-opening-bell).
`--backtest-price-symbols` changes only the warehouse price lookup after the
recorded effective date, preserves model/strategy symbol identity, and pins its
provenance with the adjusted-price snapshot. A period crossing the rename date
requires a prepared continuous snapshot. The new run uses the previous run's
frozen price snapshots, original strategy, costs and capacity. The price audit
also records that BRK-A and BRK-B currently end on September 8; the existing engine
carries their last close on September 9.

Old epoch-one long-side capital returns were 34.85% (2024), 26.89% (2025), and
27.12% (2026 YTD). These are the old rolling model's results, not results for the
new document model. The new run will produce its own metrics after epoch one.
