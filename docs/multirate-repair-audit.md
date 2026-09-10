# Multi-rate repair audit — 2026-09-10

This is an initial repair audit, not a certification of the historical models,
backtests, or live notebook. Existing uncommitted work was retained.

## Repairs verified in this pass

- Option aggregation now parses string entry dates explicitly. The Pandas to
  Polars migration left date-only strings becoming null and being discarded.
  Existing aggregation tests now exercise the actual Polars interface and
  retain their synthetic-contract and liquidity-weighted quote assertions.
- Dense-rate and sparse-signal normalization use a shared fitting helper.
  Rows at or after `train_end_date` are excluded from fitting. Sparse
  statistics are saved alongside the other rates and reused for inference.
  Checkpoint statistics avoid unnecessary refitting/scans and are checked for
  dimension and numeric validity.
- Inference rejects checkpoints lacking normalization, label vocabulary, or
  configuration instead of silently fitting preprocessing on live inputs.
- Document label IDs use the saved vocabulary during inference. A subset of
  training categories retains its original class IDs; unseen labels are
  excluded from classification accuracy.

## Checkpoint currently selected by the live notebook

`optimal_trader/notebooks/trading_app_v2_multirate_live.ipynb` defaults to:

`artifacts/multi-rate-mtl/train_from_scratch_100b_fmp_thetadata_2021_present_polars_torch/multirate_mtl_checkpoint_latest.pt`

The inspected checkpoint records epoch 3, batch 2006 and 102,328 training
samples. Its top-level keys are `state_dict`, `optimizer_state_dict`, and
`metrics`. It contains no normalization statistics, label vocabulary, or
configuration. Its target families are `["__empty_sparse_family__"]` despite
listing Oracle and HITS heads. That metadata does not establish trained
trading supervision. The current inference check deliberately rejects this
file. Exact training metadata and supervision evidence must be recovered, or
a supervised replacement trained and evaluated. The checkpoint was not altered.

## Remaining unresolved work

- The older `optimal_trader/scripts/multirate_transformer` task registry
  imports `run_feature_family_gnn_smoke.py`, which imports the removed
  `CORPORATE_EVENT_COLUMNS` warehouse API. Two model-construction tests fail
  there; 14 other targeted tests pass. Its historical task contract needs to
  be resolved before restoring or migrating it.
- Documentation describes different task sets and architectures across the
  two repositories. The desired experiment and last known working workflow
  have not yet been identified by the user.
- Existing uncommitted corpus construction, target construction, streaming,
  backtesting, and live-notebook changes have not received a complete audit.
- Normalization respects an explicit date cutoff; separate symbol holdouts
  and fraction-based validation splits still need a preprocessing audit.
- No full-corpus CUDA training, WFO benchmark, or live order execution was run.

## Verification

Targeted orchestrator tests cover the model, training contract, option
aggregation, subtoken documents, factorization pooling, and target documents.
A temporary synthetic corpus also completed a one-epoch CPU training run,
wrote a checkpoint, and completed inference using the saved checkpoint with
restored architecture and preprocessing. This verifies pipeline execution,
not predictive quality or real-data coverage.

## Follow-up

The [issuer/instrument streaming milestone](multirate-streaming-milestone.md)
supersedes the initial trainer findings above. It adds shared gradient-bearing
issuer encodings, causal instrument fusion, bounded sparse/label reads, batch
array release, symbol-restricted normalization, and a real-data CUDA/checkpoint
check. Fraction-based validation is explicitly rejected; larger corpus builders,
legacy experiments, frozen-encoder benchmarks, and full WFO remain separate work.
