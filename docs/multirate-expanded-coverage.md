# Expanded multi-rate source audit

The existing fresh v8 corpus is narrower than the historical 40-feature-family
plus 17-event-family inventories. `research_tools.multirate_source_inventory`
now snapshots warehouse sources with bounded Polars reads, one issuer/source at
a time. It records field schemas, finite numeric fields, date ranges, pre-2024
coverage, missing reads and failures. Snapshots resume without repeating successful
reads; failed reads are retried. This is staging data, not a completed training
corpus, and it does not change the active checkpoint or its training inputs.

The executed 100B inventory is `artifacts/multirate_recovery/100B/expanded_sources_v9`.
Across 114 issuers, 3,881 source reads returned observations, 451 were empty, and
zero errors remain after correcting timezone-aware institutional read slicing.
These are source-read counts, not model family counts.

All 114 issuers have pre-2024 ratios and metrics in period-unspecified storage.
Explicit annual/quarter paths contain no pre-2024 ratio/metric observations.
AAPL has 166 historical observations for each, spanning 1985-09-30 to 2026-07-06.
The older fiscal_period values inspected are NaN: history must not be silently
assigned to an annual/quarterly family based on guessed cadence.

Macro facade delegation was also repaired in quant-warehouse. The inventory
includes actual economic history (10 series over 10,021 dates) and Treasury
history (12 series over 9,158 dates), rather than empty schema placeholders.

The historical government-trade event export used transaction dates even when
raw warehouse records contain later disclosure dates. For example AAPL's
2014-02-10 transaction was disclosed 2014-02-27. Restore this input from raw
records using disclosure availability; do not copy that export into historical
inputs unchanged. Oracle/HITS remain supervised-only.

Remaining expansion work: map source snapshots to explicit family adapters,
retain individual numeric/text features, restore sparse events with availability
provenance, handle unknown cadence without fabricated labels, audit usable
pre-cutoff coverage, and train/validate a separate expanded model. The inventory
alone does not establish that every older family is populated or learnable.


Expanded $1T smoke (September 11): `notebooks/multirate_expanded_smoke.ipynb` records the subset/build/launch recipe. The actual run is `artifacts/multirate_recovery/1T/train_expanded_smoke_v9`: one fresh epoch, 13 equity symbols / 11 issuers plus six existing option paths, 40 restored feature families and 1,379 individual numeric fields, 16 observed sparse families (no holder exits). It uses Polars streaming, the pre-2024 cutoff, and blocking original anchored-HITS backtests for 2024, 2025 and 2026 through September 9. Government and insider buy/sell heads now use exact transaction-date targets and actual opposite trades for negatives; historical event inputs retain disclosure availability. This is a pipeline smoke test, not proof of predictive performance or complete point-in-time vintage coverage. Several holder/ETF families have only post-cutoff observations. The older $100B epoch-four run is paused with its checkpoint preserved.
