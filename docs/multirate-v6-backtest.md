# Completed v6 adjusted-price backtest

Executed 2026-09-10 using the unchanged pre-2024 v6 checkpoint's saved full-calendar
2024–2025 supervised predictions. No thresholds were tuned on these backtests.

| Year | Model return | Model maximum drawdown | Model trades | Issuer-equity hold return | Hold maximum drawdown |
|---|---:|---:|---:|---:|---:|
| 2024 | 7.31% | -2.51% | 6 | 56.34% | -18.92% |
| 2025 | -2.22% | -8.18% | 5 | 40.91% | -24.54% |

Each year starts independently with $100,000. Final model equity is $107,310.10
and $97,783.59; baseline equity is $156,342.51 and $140,910.11 respectively.
These are separate annual folds, not a continuous two-year portfolio.

The policy is funded long-only with one instrument per issuer and a target entry
budget of portfolio value divided by 11 issuers. Prior-session predicted Oracle
buy probability must be at least 0.5 and greater than predicted short probability.
HITS long-return-hub predictions rank eligible entries. Positions exit when buy
is no greater than short, or sell reaches 0.5. Existing holdings do not rotate
on ranking changes. End-of-year positions are liquidated. Signals from one EOD
execute at the next session's close; they do not execute at the same day's close.
The first available score therefore executes on the second session of each fold.

Equities use warehouse `splits_and_dividends` adjusted prices with no separate
split/share or dividend-cash adjustment. Quantities are synthetic adjusted-price
units. Options retain bid/ask quotes and raw underlying prices for intrinsic
settlement. Fees and slippage are each 5 basis points per side. Idle cash earns
zero. Whole adjusted equity units and whole option contracts are used.

The model selected only equities: no option trades occurred. Average invested
capital was 14.31% in 2024 and 9.64% in 2025. All model trades closed by June 6,
2024 and May 21, 2025 respectively. Lower portfolio drawdown therefore should
not be interpreted as demonstrated superior risk management: cash exposure was
very different. Both policies had zero stale held-position marks.

This is evidence of underperformance for this fixed long-only policy, not an
exhaustive test of all possible strategies using the checkpoint. Short selling,
option writing, early exercise, and assignment are not simulated. The roster
is the retrospectively selected $1T universe, not a point-in-time market-cap
universe. Options are only the few AAPL contracts in the existing corpus.

Artifacts: `artifacts/multirate_recovery/1T/backtest_v6/results.json` and per-year
`*_model/` and `*_baseline/` folders contain hashed input snapshots, equity
curves, action tapes, trade lists, scored panels, and strategy manifests.
The three replay regression tests passed, including next-session/weekend
execution, raw-price option expiration valuation, and no duplicate dividend
credit when using adjusted equity prices.

## Previous HITS entry/exit policy replay

The unchanged checkpoint was also replayed with the prior HITS policy:
hub >= 0.50 to enter, authority >= 0.50 to exit, and a shared top-5 book without
an issuer cap or Oracle gating. Existing positions remain until authority exit,
expiration, or fold-end. Entry allocation is up to one fifth of NAV, cash-limited.
As in the prior DTE policy, calls and equities use long-return HITS channels;
puts use short-return HITS channels. The present training targets describe each
contract's own price path, so this inherited put-channel convention should not
be assumed equivalent to the underlying's bearish direction.

The Polars replay uses the same adjusted-price, next-session-close, fee and
slippage conventions as the Oracle comparison above. It adapts the former DTE
basket policy to individual instruments; it does not rebuild the old baskets.

| Year | Return | Maximum drawdown | Trades | Maximum entry hub score |
|---|---:|---:|---:|---:|
| 2024 | 0.00% | 0.00% | 0 | 0.225259 |
| 2025 | 0.00% | 0.00% | 0 | 0.215246 |

All 3,762 scored rows in 2024 and 3,735 in 2025 fail the 0.50 entry threshold.
Both $100,000 portfolios remain entirely in non-interest-bearing cash. This
result reflects the threshold and score scale, not a demonstrated profitable
HITS trading strategy. HITS predictions are regression outputs, not calibrated
probabilities. Thresholds were not reduced after inspecting these test periods.

Artifacts are under `artifacts/multirate_recovery/1T/backtest_v6_hits/`, including
`results.json`, `entry_audit.json`, and each year's complete replay bundle.
Four replay tests pass, including a nonzero-trade HITS scenario verifying shared
slots, ranking, prior-session authority exits, and independence from Oracle.

## Anchored HITS percentile policy

The user selected the older anchored strategy in
`optimal_trader/scripts/run_oracle_hits_anchored_wfo.py`. Its `hits_score`
function converts each day's predictions to average-tie percentile ranks,
separately for each side's hub and authority. Defaults are strict percentile
`> 0.80` for entry and exit, capacity 20, separate long-only and short-only
books, and 5.5 bps per change in portfolio weight. No Oracle gate or issuer cap
is applied. Each held symbol has signed weight 1/20; vacant capacity stays cash.
Exits are processed before entries, including same-session re-entry if both
signals qualify. Existing positions are not rotated because their hub rank falls.

`anchored_hits_replay.py` reproduces that ranking, threshold, and fixed-weight
accounting in bounded Polars day slices. It uses the unchanged pre-2024 v6
scores for 13 equities, rather than retraining the older feature-family random
forests every year. Options are excluded because the anchored strategy was an
equity strategy. Per the user's EOD requirement, signal-date ranks execute at
the following session's close, and held positions then earn subsequent returns.
The old script used signal-date weights against next returns, so this execution
delay is an intentional difference. Adjusted equity prices are reused from the
hashed inputs of the completed v6 replay.

| Year | Book | Return | Max drawdown | Entry events | Exit events | Open at year-end | Mean gross exposure |
|---|---|---:|---:|---:|---:|---:|---:|
| 2024 | Long | 16.55% | -7.97% | 328 | 319 | 9 | 40.63% |
| 2024 | Short | -15.14% | -16.64% | 374 | 367 | 7 | 29.98% |
| 2025 | Long | 14.97% | -11.16% | 238 | 230 | 8 | 37.12% |
| 2025 | Short | -9.17% | -13.81% | 367 | 361 | 6 | 24.84% |

Each book/year starts independently at $100,000. Entries are action events,
not distinct round trips: authority exits may immediately re-enter if the hub
also qualifies. The older fixed-weight cost convention charges net target-weight
changes; it does not charge an exit/re-entry with unchanged net weight or drift
rebalancing. Open positions are marked at the final close without forced terminal
liquidation, matching the older accounting. Short results exclude borrow fees
and locate constraints. These are model-policy experiments, not broker execution
simulations. With 13 equities and 20 slots, gross exposure cannot exceed 65%.

This is not an identical-cost comparison with the earlier Oracle and raw-HITS
replays: those used 10 bps per side, while this uses the anchored default of
5.5 bps on target-weight changes. It also uses constant portfolio weights rather
than fixed quantities. No rank thresholds or capacities were tuned on 2024/2025.
The positive long returns alone do not establish predictive alpha.

Artifacts: `artifacts/multirate_recovery/1T/backtest_v6_anchored_hits/` contains
four annual book folders with equity curves, target weights, action tapes and
summaries, plus `results.json`, `input_sha256.json`, and the reproduction script.
Seven tests pass across both replay modules, including ties, strict percentile
thresholds, and a Friday-to-Monday next-close execution check for long and short.

## Corrected universe-capped allocation

Per the user's sizing correction, the current anchored replay allocates
`1 / min(top_k, scoring-universe symbol count)` per held symbol. For 13 equities
and top_k=20 this is 1/13 (7.6923%), initially $7,692.31 per symbol. The denominator
is the fold's declared scoring universe, not the number of currently open
positions. Vacant positions therefore still leave cash. The prior 1/20 results
above remain historical experiment results, not the current sizing rule.

With all other settings and entry/exit events unchanged:

| Year | Long return | Long max drawdown | Short return | Short max drawdown |
|---|---:|---:|---:|---:|
| 2024 | 26.04% | -12.09% | -22.51% | -24.60% |
| 2025 | 23.28% | -16.77% | -13.92% | -20.55% |

Artifacts: `artifacts/multirate_recovery/1T/backtest_v6_anchored_hits_sized/`.
Five anchored replay tests pass, including allocation for a universe smaller
than the configured capacity. Both books remain independent annual experiments.

## Original multi-rate transformer strategy (current reference)

The sizing clue identified a different original call path than the earlier
anchored experiments: `optimal_trader/scripts/run_symbol_year_transformer_mtl.py`
(lines around 2861 and 3022) calls `build_legacy_compatible_scores` from
`scripts/multirate_transformer/trading_policy.py`, then the existing
`run_shared_book_framework_comparison` in orchestrator's `shared_book.py`.
The 0.80 authority-threshold results above are different strategies and must
not be presented as this original multi-rate strategy.

The original adapter percentile-ranks all four HITS components daily, sets
entry scores from hubs, and constructs single-model long/short agreement from
long-hub >= short-hub (long) versus short-hub > long-hub (short). The default
optimal-trader planner enters on hub percentile >0.50 with direction agreement
and exits when that agreement is lost. Although authority exit-score columns
are produced, the supplied consensus counts control this call path's exits.
Capacity is min(20, symbol count); cost is 0.5 bps commission plus 5 bps slippage.

The unchanged pre-2024 v6 checkpoint was evaluated through the actual existing
functions, not a replacement trading loop. Only prediction column names were
adapted, with bounded annual panels converted at the original pandas API boundary.
Polars still prepares data. Adjusted equity prices, original forward-return
alignment, original costs and metric calculations were retained. In particular,
there is no extra next-session-close execution shift in this reference run.

| Year | Book | Return from initial $100,000 | Final equity | Sharpe | Max drawdown | Entries | Exits |
|---|---|---:|---:|---:|---:|---:|---:|
| 2024 | Long | 16.88% | $116,883.61 | 1.349 | -10.27% | 86 | 80 |
| 2024 | Short | -22.30% | $77,695.46 | -2.631 | -24.99% | 82 | 76 |
| 2025 | Long | 24.26% | $124,256.14 | 1.406 | -15.26% | 115 | 108 |
| 2025 | Short | -13.34% | $86,655.07 | -1.398 | -18.92% | 88 | 83 |

The original `total_return` metric divides final equity by the first recorded
(post-first-return) equity value, rather than initial capital. It consequently
reports 17.1022%, -22.4952%, 23.5968%, and -12.2377% for these four rows. Those
original values are preserved; the additional `capital_return` field reports
the actual change from starting capital shown above. Sharpe uses mean daily net
return / sample standard deviation * sqrt(252), with zero risk-free rate.

These are independent annual equity books, not options or a combined long/short
portfolio. No borrow costs/locates or forced terminal liquidation are added.
Original net target-weight turnover accounting remains unchanged. This tests
the new model under the old engine; it does not reproduce old model training.

`platforms/backtesting_frameworks/existing_multirate_backtest.py` is the thin
adapter reused by the epoch monitor. It imports the existing score policy and
calls the existing engine directly; it contains no new trading loop. The 100B
epoch monitor now uses this reference strategy and prints capital return,
Sharpe, drawdown, entry events, exposure and capital-return change versus the
previous epoch. Equities with no scores in the validation calendar are recorded
as excluded from backtesting, while their older data remain eligible for training.

Artifacts: `artifacts/multirate_recovery/1T/backtest_v6_existing_strategy/`, with
verified reusable-adapter outputs under `verified/`. All four books matched the
direct original-engine invocation to 1e-10 on equity, return, Sharpe, drawdown
and event counts. Source files and input prices/scores have recorded hashes.
Twelve epoch-monitor/shared-book tests passed.

## Fresh v8 Oracle-gate comparison

The optional `oracle_gate=True` variant in `existing_multirate_backtest.py` adds
predicted Oracle permission to the same original score policy and shared-book
engine. HITS ranks are computed on the full original universe before gating.
Entries require the side's Oracle probability >= 0.5, strictly above the
opposite side, plus existing HITS eligibility. Sell/cover probability >= 0.5
vetoes entry and closes the corresponding position. Losing Oracle direction or
existing HITS agreement also closes it. A side probability falling below 0.5
alone blocks new entries but does not force an existing position out. This is
an additional gate on the current strategy, not a restoration of the earlier
Oracle replay's different sizing, issuer cap, costs, or execution timing.

`research_tools.oracle_gate_comparison.compare_epoch_oracle_gate` reuses each
completed epoch's full predictions and frozen baseline price files. Capacity,
5% sizing for this universe, adjusted prices, 5.5-bps costs, and execution remain
identical. No training data, weights, or thresholds are fitted by this comparison.
The detached watcher under the fresh run's evaluation directory compares epochs
as they complete and writes `epoch_NNNN/oracle_gate_comparison.json` plus the
native engine artifacts under `epoch_NNNN/oracle_gate/{year}/`.

Epoch-3 long results (2026 ends September 9):

| Period | Original return | Gated return | Original Sharpe | Gated Sharpe | Gated mean exposure |
|---|---:|---:|---:|---:|---:|
| 2024 | 53.63% | 7.07% | 2.66 | 2.14 | 9.98% |
| 2025 | 32.79% | 3.21% | 1.55 | 0.72 | 12.52% |
| 2026 YTD | 14.04% | 6.74% | 1.77 | 2.34 | 13.55% |

Gated maximum drawdowns were 1.12%, 3.98%, and 1.87% respectively. Epoch 1
made no gated trades on either side. Epoch 2 gated long returns were 17.32%,
2.99%, and 9.77%; gated shorts made no trades across epochs 1–3. Their 0% return
reflects cash, not a successful short selection. The 0.5 gate did not improve
long total returns in these comparisons; its lower exposure reduced drawdowns.
The ungated baseline remains the primary run; both variants are retained.

### Directional-only comparison

`compare_epoch_oracle_gate(directory, mode="directional")` adds only buy > short
for long entry/holding and short > buy for short entry/holding to the original
HITS permissions. Ties permit neither side. There is no absolute Oracle entry
threshold and sell/cover do not veto trades. HITS ranks are computed before the
gate and remain unchanged. Prices, capacity, costs and execution stay identical.
Outputs are separate under `oracle_directional_gate/{year}/` and
`oracle_directional_gate_comparison.json`. The fresh v8 directional watcher
completed epochs 1–3 and waits for subsequent completed epoch baselines.

Epoch 3 capital returns (separate $100,000 annual books, 2026 through Sept 9):

| Period | Original long | Directional long | Original short | Directional short |
|---|---:|---:|---:|---:|
| 2024 | 53.63% | 6.31% | -31.29% | -34.17% |
| 2025 | 32.79% | 7.00% | -32.43% | -26.83% |
| 2026 YTD | 14.04% | 16.61% | -12.42% | -22.67% |

Directional long exposure averaged 27.24%, 28.78%, and 36.60%, respectively;
short exposure averaged 93.93%, 94.84%, and 93.69%. Results vary substantially
across epochs; this comparison does not establish a consistent improvement.
