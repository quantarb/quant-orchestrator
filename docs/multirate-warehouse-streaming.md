# On-demand equity and frozen-option training

The [step-by-step notebook](../notebooks/multirate_warehouse_training.ipynb)
explains documents, subtokens, annual memory, objectives, coverage, and backtests.
Select $1T, $100B, or $10B in its configuration. Review mode reads existing
outputs; train mode runs the same workflow sequentially for selected universes.

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
  --checkpoint-every-batches 10 --progress-every-batches 1 \
  --skip-embeddings --skip-t-sne
```

An existing output directory is rejected. `--corpus` is retained only for
recorded checkpoint evaluation and exact continuation of older runs, not as
an input to new training. Streaming checkpoints have a distinct normalization
contract and cannot be loaded by the older corpus inference path.

## Inputs and annual documents

`research_tools/warehouse_multirate.py` reads adjusted equity prices, warehouse
financial feature families, macro and peer context, and disclosure-dated issuer
events. Source tables use a bounded issuer cache; one CPU batch is prefetched
while the GPU runs. The shared token/subtoken objectives live in
`multirate_training_step.py`. Native observations retain their dates; no daily
filler labels are added. Equity documents preserve chronological annual memory.
Each year's newly formed option basket has a distinct instrument identity.

Options are discovered separately for every equity symbol, including separate
share classes. `universe.json` records equities, stored options availability, and
excluded equities without enough pre-cutoff price history. `option_coverage.json`
records source dates by year, first-session failures, created baskets, training
price observations, and documents containing multiple price observations.
An epoch fails if an expected option underlying was omitted or only contributed
isolated snapshots.

Up to five observed expiry/DTE cohorts per right are selected across
the available DTE range on the actual first NYSE session of each year. When fewer
than five expirations exist, all available expirations are used. Every
strike in each selected cohort receives a fixed equal weight. Later contracts
are never added. A missing constituent invalidates that day's basket quote;
remaining members are never renormalized. Older history can be sparse. Missing
first-session chains are reported without using a later fallback.

`frozen_option_adjustments.py` conserves economic exposure through forward stock
splits by increasing contract counts and reducing strikes. Actual post-split
quote identifiers are matched within strike-rounding precision. Expiration
values use unadjusted underlying prices from option quotes and split-consistent
intrinsic payoffs. Reverse-split deliverables require an explicit mapping.
Cohort-member Parquet files are audit outputs produced on demand, never inputs
reused by another training run.

## Backtests and artifacts

After an epoch, the model scores the requested years from empty memory, without
replaying training history as an inference warmup. Coverage checks compare
predictions to actual priced dates.

Equity reports use the existing HITS policy and shared-book return engine with
separate long-only and short-only books. Frozen-option reports use equity
signals to decide direction: long calls for bullish signals and long puts for
bearish signals. Option decisions execute no earlier than the next equity
trading session and only with complete basket quotes. The existing capacity
planner retains positions awaiting an executable exit. Expirations force
settlement; missing settlement values fail the backtest.

Option calculations use the shared fixed-weight engine, basket bid/ask spread
costs, and 5.5 basis points per unit of turnover. Missing marks use the last
observed midpoint for valuation only. Reports count stale valuation
position-days and open positions at period end. These are fixed-weight return
simulations, not a broker cash/margin ledger.

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
