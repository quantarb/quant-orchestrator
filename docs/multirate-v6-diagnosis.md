# Multi-rate v6 diagnosis

Executed September 10, 2026 against the completed `train_long_v6` checkpoint.
This is a diagnosis and performance profile, not a new full training run or a
claim that the model's prediction problems have been repaired.

Evidence lives under `artifacts/multirate_recovery/1T/diagnosis_v6/`.
The checkpoint, data, and production training code were not changed.

## Prediction failures

The frozen daily close normalization mean is 13,361.328125 and its scale is
57,713.671875. One pooled normalization spans ordinary equities, BRK-A, and
options. Consequently, ordinary price changes occupy a very small interval in
normalized space. An unrestricted linear decoder's modest normalized errors
become thousands of dollars when converted back to source units.

The January 5 to January 8, 2024 trace uses the actual saved checkpoint and the
ordinary inference path, with the saved training normalization. Source and
successor values were checked against the original parquet rows.

| Instrument | Current close | Next close | Subtoken NTP prediction | Token NTP prediction |
|---|---:|---:|---:|---:|
| AAPL | 179.14 | 183.47 | -7,265.20 | -8,864.11 |
| BRK-A | 554,300 | 558,780 | 363,004.57 | 342,023.28 |
| AAPL241220C00185000 | 18.85 | 21.00 | 11,355.57 | 77,561.90 |

These are raw-unit reconstructions of NTP outputs, not supervised trading
scores. BRK-A accounts for 98.65% of the full 2024–2025 subtoken price squared
error and 98.85% of token price squared error. Every other instrument still
loses to persistence on the aggregate price-family metric; removing BRK-A from
reporting would not establish useful performance.

The pre-2024 options have only one expiration and one strike: expiration day
19,706 and strike 125. Both have zero training variance, so normalization falls
back to scale 1. Evaluation expiration values become 371 or 735, and strikes
become 60 or 120, versus training targets of zero. For the traced 2024 call,
subtoken NTP reconstructs strike 128.50 when both current and next strike are
185. The current architecture predicts absolute values through learned heads;
it has no persistence residual path to preserve a known constant automatically.
This is an identifiable representation and training-coverage problem. It does
not establish that normalization is the only cause of weak learning.

The accumulated weighted scalar loss is also imbalanced: Oracle contributes
approximately 17,913, HITS 335, and reconstruction 3,749 after its 0.1 weight.
These are summed loss contributions across training, not measured gradient
shares. Per-objective gradient measurements are needed before adjusting weights.

The 19 reconstruction/time/NTP tests passed. They cover causal future exclusion,
family versus token successors, missingness, and separate masking semantics.
The real trace preserves the next trading observation across a weekend. This
check found no alignment failure; it is not a proof that every data path is
correct. NTP uses the feature-corrupted training view, while token MTP gets a
separate whole-family-corrupted view. Naturally absent fields stay excluded
from reconstruction targets. Oracle/HITS remain supervised-only.

## Training profile

A seeded, 2,048-sample, one-epoch diagnostic used the full run's FP32, batch 128,
64-dimensional model, two layers, 252 daily observations and both NTP/MTP.
It retained issuer context and streaming Polars. It completed 18 optimizer
steps with loss 5.279881, matching the prior diagnostic result.

CPU profiling and synchronized CUDA boundary timers measured 88.54 seconds
inside training, excluding process initialization and artifact output:

| Region | Seconds | Approximate training share |
|---|---:|---:|
| Sample preparation and tensor staging | 77.02 | 87.0% |
| Both model forward passes | 5.24 | 5.9% |
| Backward, gradient diagnostics, clipping, optimizer | 4.79 | 5.4% |
| Reconstruction target construction | 0.60 | 0.7% |

The target-construction region includes 0.50 seconds in the Python
next-observation loops. Those loops are not the main bottleneck.
The staging region includes 76.20 seconds in lazy sample materialization.
Nested costs must not be added together: streaming windows account for 57.76
seconds, repeated version lookup for 13.93 seconds, and supervised label lookup
for 12.10 seconds. Some version calls also occur outside the timed training loop.
The whole process made 341,842 Polars collect calls, many caused by eager,
column-by-column operations rather than separate disk reads.

Concrete hot spots:

- `StreamingContext.window`: repeated bounded queries and tensor conversion.
- `StreamingContext.version`: rebuilding a Polars Series and converting a
  cached datetime to epoch nanoseconds for each lookup.
- `StreamingSupervision.get`: querying an event label for each individual anchor.
- Gradient hooks: individual finite checks and device-to-host scalar transfers;
  the hooks accounted for 3.70 seconds nested within backward.

The profile has measurement overhead. The sparse 2,048-anchor sample also has
less window reuse than a complete epoch, so these percentages must not be
extrapolated directly to the 4.4-hour run. They identify where to optimize and
what to measure next; they do not establish a full-run speedup.

An isolated conversion prototype on a real 1,276-row, 132-column block reduced
conversion from 4.33 ms to 1.35 ms by combining casting and null handling into
one Polars projection. Outputs matched exactly, including NaNs. This is a
component benchmark only; the prototype has not been installed in production.

## Repair order

1. Batch bounded Polars projections, cache timestamp integers alongside date
   metadata, and read supervised labels in bounded symbol/date blocks. Preserve
   event-only label semantics and as-of slicing. Benchmark before/after on the
   same anchors, then measure warm-cache behavior on a longer segment.
2. Replace pooled cross-instrument price scaling with an explicit scale-aware
   representation fitted from training data only, including a defined policy
   for previously unseen option contracts. Keep invertible raw-value metrics.
   Encode contract dates and constants deliberately rather than treating a
   single observed expiration/strike as adequate normalization coverage.
3. Evaluate persistence-residual NTP decoding so known values need not be
   relearned from scratch. Preserve coherent next observations at both levels.
   Do not let a residual path reveal a hidden MTP target: each view must use
   only information actually visible under that view's mask.
4. Add per-feature/per-instrument NTP and held-out MTP baselines, measure
   objective gradient contributions, and run small fitting/ablation checks.
   Do not launch another full run solely because aggregate training loss fell.

These changes should preserve pre-2024 history, both hierarchy levels, both
self-supervised objectives, supervised-only Oracle/HITS, and streaming memory
bounds. Speed and quality improvements require new controlled measurements.

## Artifacts

- `normalization.json`, `data_audit.txt`: frozen scales, ranges, per-symbol errors.
- `sample_traces.json`, `raw_trace_rows.json`: 38 feature/level traces and source rows.
- `trace_run/`: ordinary inference artifacts for 75 anchors across 15 instruments.
- `profile.pstats`, `profile.txt`, `timings.json`, `profile_run/`: profile evidence.
- `conversion_benchmark.json`: isolated conversion timings.
- `contract_tests.txt`: 19 passing tests.
- `profile_reproduce.py`, `trace_reproduce.py`, `normalization_reproduce.py`:
  experiment-local reproduction scripts; run from the repository root with
  `PYTHONPATH=../quant-warehouse:.:scripts` and the configured Python environment.
