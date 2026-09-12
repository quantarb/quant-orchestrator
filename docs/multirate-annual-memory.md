# Annual documents with recurrent memory

`--sequence-mode annual_memory` uses calendar-year documents: January 1 inclusive
through the next January 1 exclusive. Every observation in the year is retained;
there is no artificial January 1 market observation. Shared macro/peer/calendar
rows cannot create training years without issuer observations or training events.
The document contract is `calendar_year_recurrent_v1`.

The four native-rate streams and the two issuer-context streams retain learned
ending token and family states. A dedicated first position in the next document
receives those states; raw features and event labels begin at subsequent positions.
Families without a new observation retain their prior state. Cold starts use zero
memory. Memory is detached at each document boundary, making gradient propagation
bounded to the current year. This is recurrent state, not an encoded-window cache.
During training it reflects the previous forward pass, including training masking
and dropout; it is not recomputed after every optimizer update.

`research_tools.annual_memory.AnnualCorpus` packs different instruments into a
batch and visits each instrument's years in chronological order. Memory resets at
each epoch. Checkpoints save recurrent state, its processed-sample count, optimizer,
RNG state, and diagnostics. A signature rejects resumes with changed input files,
sample dates, batch size, or seed. Raw history is not repeated across annual
boundaries. The model still checks its 512-position capacity without truncation.

Inference rebuilds memory chronologically under frozen checkpoint weights,
including eligible years before the requested score interval. Training-event date
metadata before the recorded training cutoff is retained for matching warmup year
selection; target values are not model inputs. All requested scoring dates remain
in the inference calendar. A mid-year query replays the same prior years and only
the available prefix of the current year. Backtests use the existing adjusted-price
anchored-HITS engine and frozen yearly price snapshots.

Validation artifacts are under
`artifacts/multirate_recovery/10B/annual_memory_smoke/`. The smoke retains 1,751
numeric fields for AAPL, MSFT and their eight stored option paths, with data from
2020. Both epochs completed training, full 2024–2026 scoring, and separate yearly
long/short backtests. Each evaluation exported 2,963 rows with zero missing,
duplicate, unexpected or nonfinite rows. A replay ending June 30, 2025 matched
1,648 corresponding full-run score rows across all 26 heads within 7.01e-7.
The first annual training epoch took 3.0 seconds versus 6.6 seconds for quarterly
sampling on the same corpus and batch size. These are small-subset training-loop
measurements, not a full-universe epoch estimate or evidence of predictive quality.

Core and regression validation covers 106 tests, including CPU-exact model,
optimizer and recurrent-state resume; actual GPU resume is also compared with
an uninterrupted run in `resume_comparison.json`. See `prefix_comparison.json`,
`result.json`, and the per-epoch prediction/backtest reports for executable evidence.
