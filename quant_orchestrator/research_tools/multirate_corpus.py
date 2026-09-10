"""Bounded Polars assembly of warehouse observations and instrument targets.

No market-data download occurs here. Option selection operates only on stored
full chains. Each reduced instrument history is persisted independently.
"""

from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import sys

import polars as pl

from quant_orchestrator.research_tools.multirate_supervision import instrument_asset_groups
from quant_orchestrator.research_tools.multirate_targets import materialize_instrument_targets

STATEMENTS = ("income", "balance", "cash")
CHANNELS = ["signal_value", *[f"text_{i}" for i in range(7)]]


def statement_fields(frame):
    """Retain raw numeric statement fields, excluding temporal metadata."""
    metadata = {"date", "period_ending", "filing_date", "accepted_date", "fiscal_year", "fiscal_period"}
    return [name for name, dtype in frame.schema.items()
            if name not in metadata and dtype.is_numeric()
            and frame[name].cast(pl.Float64).is_finite().any()]


def bounded_dates(frame, start, end, *, column="date"):
    """Assert the warehouse read contract before accepting model observations."""
    if frame.is_empty():
        return frame
    dates = frame[column].cast(pl.Datetime("ns"))
    if (
        dates.null_count()
        or dates.min() < datetime.fromisoformat(start)
        or dates.max() > datetime.fromisoformat(end)
    ):
        raise ValueError(f"Warehouse {column} values violate requested range {start}..{end}")
    return frame


def missing_statement_years(frame, *, column, start, end, minimum):
    """Check internal coverage without requiring statements before their source floor.

    This does not establish that the first stored observation is the provider's
    earliest available observation. Full-history refresh auditing owns that check.
    """
    first = frame[column].min()
    counts = dict(frame.group_by(pl.col(column).dt.year().alias("year")).len().iter_rows())
    first_year = max(int(start[:4]), first.year + 1)
    complete_end = int(end[:4]) + end.endswith("12-31")
    return [year for year in range(first_year, complete_end) if counts.get(year, 0) < minimum]


def _option_histories(warehouse, issuer, sessions, year, staging):
    from quant_warehouse.platforms.data_providers.thetadata.options import (
        read_thetadata_eod_option_chain,
    )

    first = sessions["date"].min()
    columns = [
        "snapshot_date",
        "contract_symbol",
        "expiration",
        "strike",
        "option_type",
        "bid",
        "ask",
        "volume",
        "underlying_price",
    ]

    def read(start, end):
        return read_thetadata_eod_option_chain(
            issuer, start_date=start, end_date=end, columns=columns, backend=warehouse.backend
        )

    chain = read(first, first)
    if chain.is_empty():
        raise ValueError(f"Missing stored first-session option chain for {issuer}/{year}")
    observed_spot = chain.filter(
        pl.col("underlying_price").is_finite() & (pl.col("underlying_price") > 0)
    )
    if observed_spot.is_empty():
        raise ValueError(f"Missing contemporaneous underlying quote for {issuer}/{year}")
    spot = observed_spot["underlying_price"].median()
    # Membership depends exclusively on the first session. Select one call and
    # put with the latest expiration within that calendar year, nearest spot.
    eligible = chain.filter(
        (pl.col("expiration") > first) & (pl.col("expiration").dt.year() == year)
    )
    members = (
        eligible.with_columns((pl.col("strike") - spot).abs().alias("_distance"))
        .sort(["expiration", "_distance", "contract_symbol"], descending=[True, False, False])
        .unique("option_type", keep="first", maintain_order=True)
    )
    if members.height != 2:
        raise ValueError(f"Expected a call and put in {issuer}/{year}")
    selected = members["contract_symbol"].to_list()
    pieces = []
    day = first
    last = min(sessions["date"].max(), members["expiration"].max())
    while day <= last:
        stop = min(day + timedelta(days=6), last)
        quotes = read(day, stop).filter(pl.col("contract_symbol").is_in(selected))
        if not quotes.is_empty():
            path = staging / f"option_quotes_{issuer}_{year}_{day:%Y%m%d}.parquet"
            quotes.write_parquet(path)
            pieces.append(path)
        day = stop + timedelta(days=1)
    if not pieces:
        raise ValueError(f"Missing selected option histories for {issuer}/{year}")
    for row in members.iter_rows(named=True):
        symbol = row["contract_symbol"]
        path = (
            pl.scan_parquet(pieces)
            .filter(pl.col("contract_symbol") == symbol)
            .collect(engine="streaming")
        )
        path = path.filter(
            (pl.col("bid") > 0)
            & (pl.col("ask") >= pl.col("bid"))
            & pl.col("bid").is_finite()
            & pl.col("ask").is_finite()
        )
        path = (
            path.select(
                pl.lit(symbol).alias("symbol"),
                pl.col("snapshot_date").alias("date"),
                ((pl.col("bid") + pl.col("ask")) / 2).alias("open"),
                pl.col("ask").alias("high"),
                pl.col("bid").alias("low"),
                ((pl.col("bid") + pl.col("ask")) / 2).alias("close"),
                "volume",
                "bid",
                "ask",
            )
            .unique("date")
            .sort("date")
        )
        yield row, path


def build_fresh_corpus(
    output: Path,
    *,
    roster: pl.DataFrame,
    start: str,
    end: str,
    option_issuers: tuple[str, ...] = (),
    options_start_year: int = 2023,
    audited_statement_gaps: dict[str, list[int]] | None = None,
    warehouse=None,
) -> dict:
    from quant_warehouse import Warehouse

    instrument_asset_groups(roster)
    if not {"issuer", "underlying_symbol"}.issubset(roster.columns):
        raise ValueError("Roster requires issuer and underlying_symbol metadata")
    if not set(option_issuers).issubset(set(roster["underlying_symbol"])):
        raise ValueError("Option issuers must be represented in the roster")
    if datetime.fromisoformat(start) >= datetime.fromisoformat(end):
        raise ValueError("Corpus start must precede end")
    warehouse = warehouse or Warehouse()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    staging = output / "partitions"
    staging.mkdir()
    (output / "prices").mkdir()
    paths = {rate: [] for rate in ("annual", "quarterly", "daily", "sparse_events")}
    families, target_families = (
        set(),
        {"equity.strategy.hits_graph", "equity.strategy.oracle_trades"},
    )
    coverage, taxonomy = [], []

    def save(rate, frame):
        path = staging / f"{rate}_{len(paths[rate]):06d}.parquet"
        frame.write_parquet(path)
        paths[rate].append(path)

    def add_instrument(tax, frame):
        bounded_dates(frame, start, end)
        if frame.is_empty():
            raise ValueError(f"No price history for {tax['symbol']}")
        frame = frame.with_columns(
            pl.col("date").cast(pl.Datetime("ns")), pl.lit(tax["symbol"]).alias("symbol")
        ).sort("date")
        frame.write_parquet(output / "prices" / f"{tax['symbol']}.parquet")
        features = {
            f"price.{name}": pl.col(name) for name in ("open", "high", "low", "close", "volume")
        }
        for name in ("coupon_rate", "strike", "contract_size"):
            value = tax.get(name)
            if value is not None:
                features[f"instrument.{name}"] = pl.lit(float(value))
        for name in ("expiration", "maturity"):
            value = tax.get(name)
            if value:
                value = (
                    datetime.fromisoformat(str(value)) if not isinstance(value, datetime) else value
                )
                features[f"instrument.{name}"] = pl.lit((value - datetime(1970, 1, 1)).days)
        if tax.get("option_type"):
            features["instrument.option_type"] = pl.lit(
                float(str(tax["option_type"]).lower().startswith("c"))
            )
        families.update(features)
        save(
            "daily",
            frame.select(
                "symbol",
                "date",
                *[expr.cast(pl.Float32).alias("value__" + name) for name, expr in features.items()],
            ),
        )
        save("sparse_events", materialize_instrument_targets(tax["symbol"], frame))
        taxonomy.append(tax)
        coverage.append(
            dict(
                symbol=tax["symbol"],
                rate="daily",
                rows=frame.height,
                first=str(frame["date"].min()),
                last=str(frame["date"].max()),
            )
        )

    profiles = {
        issuer: warehouse.read_profile(issuer, provider="fmp")
        for issuer in set(roster["underlying_symbol"])
    }
    for tax in roster.iter_rows(named=True):
        profile = profiles[tax["underlying_symbol"]]
        tax.update(
            sector=getattr(profile, "sector", None) or "Unknown",
            industry=getattr(profile, "industry", None) or "Unknown",
            subsector="Unknown",
        )
        print(f"[corpus] instrument {tax['symbol']}", file=sys.stderr, flush=True)
        frame = warehouse.read_prices(
            tax["symbol"], provider="fmp", start=start, end=end, output_format="polars"
        )
        add_instrument(tax, frame)
    for issuer in sorted(set(roster["underlying_symbol"])):
        for period, rate in [("annual", "annual"), ("quarter", "quarterly")]:
            for section in STATEMENTS:
                frame = warehouse.read_fundamentals(
                    issuer, section=section, period=period, provider="fmp", start=start, end=end
                )
                if frame.is_empty():
                    raise ValueError(
                        f"Missing {issuer}/{period}/{section}; refresh warehouse history first"
                    )
                date_column = "period_ending" if "period_ending" in frame.columns else "date"
                bounded_dates(frame, start, end, column=date_column)
                minimum = 1 if period == "annual" else 4
                missing_years = missing_statement_years(
                    frame, column=date_column, start=start, end=end, minimum=minimum
                )
                gap_key = f"{issuer}/{period}/{section}"
                audited_years = (audited_statement_gaps or {}).get(gap_key, [])
                if missing_years != audited_years:
                    raise ValueError(
                        f"Statement coverage differs from audit for {gap_key}: observed gaps {missing_years}, audited gaps {audited_years}"
                    )
                numeric = statement_fields(frame)
                if not numeric:
                    raise ValueError(f"No usable fields for {issuer}/{period}/{section}")
                names = {c: f"fmp.{section}.{c}" for c in numeric}
                families.update(names.values())
                save(
                    rate,
                    frame.select(
                        pl.lit(issuer).alias("symbol"),
                        pl.col(date_column).cast(pl.Datetime("ns")).alias("date"),
                        *[
                            pl.col(c).cast(pl.Float32).fill_nan(None).alias("value__" + name)
                            for c, name in names.items()
                        ],
                    ),
                )
                coverage.append(
                    dict(
                        symbol=issuer,
                        rate=rate,
                        section=section,
                        rows=frame.height,
                        fields=numeric,
                        first=str(frame[date_column].min()),
                        last=str(frame[date_column].max()),
                    )
                )
        frame = warehouse.read_fundamentals(
            issuer, section="ownership_insider_trading", start=start, end=end
        )
        if frame.is_empty():
            raise ValueError(f"No irregular issuer history for {issuer}")
        bounded_dates(frame, start, end, column="filing_date")
        family = "fmp.ownership_insider_trading"
        target_families.add(family)
        raw_fields = ["securities_owned", "securities_transacted", "transaction_price"]
        expressions = [
            pl.col(c).cast(pl.Float32, strict=False).alias(CHANNELS[i])
            for i, c in enumerate(raw_fields)
        ]
        expressions += [
            pl.lit(None, dtype=pl.Float32).alias(c) for c in CHANNELS[len(raw_fields) :]
        ]
        save(
            "sparse_events",
            frame.select(
                pl.lit(issuer).alias("symbol"),
                pl.col("filing_date").cast(pl.Datetime("ns")).alias("date"),
                pl.col("filing_date").cast(pl.Datetime("ns")).alias("event_date"),
                pl.lit(family).alias("target_family"),
                *expressions,
            ),
        )
        coverage.append(
            dict(
                symbol=issuer,
                rate="sparse",
                rows=frame.height,
                first=str(frame["filing_date"].min()),
                last=str(frame["filing_date"].max()),
            )
        )
        if issuer in option_issuers:
            prices = pl.read_parquet(output / "prices" / f"{issuer}.parquet")
            base = next(t for t in taxonomy if t["symbol"] == issuer)
            for year in range(max(options_start_year, int(start[:4])), int(end[:4]) + 1):
                sessions = prices.filter(pl.col("date").dt.year() == year)
                if sessions.is_empty():
                    continue
                for member, path in _option_histories(warehouse, issuer, sessions, year, staging):
                    print(
                        f"[corpus] option {member['contract_symbol']}", file=sys.stderr, flush=True
                    )
                    tax = {
                        **base,
                        "symbol": member["contract_symbol"],
                        "asset_class": "option",
                        "strike": member["strike"],
                        "expiration": str(member["expiration"]),
                        "option_type": member["option_type"],
                        "contract_size": 100.0,
                        "reference_url": "warehouse:thetadata/full_chain",
                    }
                    add_instrument(tax, path)
    families = sorted(families)
    for rate, files in paths.items():
        if not files:
            raise ValueError(f"Missing {rate} observations")
        scans = []
        for file in files:
            scan = pl.scan_parquet(file)
            if rate != "sparse_events":
                existing = scan.collect_schema().names()
                scan = scan.with_columns(
                    *[
                        pl.lit(None, dtype=pl.Float32).alias("value__" + family)
                        for family in families
                        if "value__" + family not in existing
                    ]
                )
                scan = scan.select("symbol", "date", *["value__" + family for family in families])
            scans.append(scan)
        pl.concat(scans, how="diagonal_relaxed").sink_parquet(
            output / f"{rate}.parquet", row_group_size=8192
        )
    pl.DataFrame(taxonomy, infer_schema_length=None).write_csv(output / "taxonomy.csv")
    manifest = dict(
        feature_families=families,
        target_families=sorted(target_families),
        start=start,
        end=end,
        coverage=coverage,
        audited_statement_gaps=audited_statement_gaps or {},
        asset_classes=sorted({t["asset_class"] for t in taxonomy}),
        options="actual fixed first-session contracts; no rolling; only positive executable quotes",
        irregular_channels=raw_fields,
        build_mode="bounded_polars",
        history_policy="retain each source's available dates; no common-start intersection; pre-source observations remain masked",
        target_contract="own instrument/year Oracle actions and HITS tails; label availability Dec 31",
        prices="FMP splits-and-dividends adjusted OHLCV; options bid/ask midpoint paths",
    )
    manifest["input_sha256"] = {}
    for name in [
        "annual.parquet",
        "quarterly.parquet",
        "daily.parquet",
        "sparse_events.parquet",
        "taxonomy.csv",
    ]:
        with (output / name).open("rb") as f:
            manifest["input_sha256"][name] = hashlib.file_digest(f, "sha256").hexdigest()
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
