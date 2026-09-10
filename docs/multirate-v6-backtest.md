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
