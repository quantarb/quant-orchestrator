"""Train the four-rate MultiRateTransformer MTL corpus."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import subprocess
import sys
from collections import OrderedDict
from datetime import datetime, timezone
from time import perf_counter
from pathlib import Path

# Make direct execution use the checkout being trained, even when an older
# globally installed quant-orchestrator is present.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl
import torch
from torch import nn

from quant_warehouse import Warehouse

from quant_orchestrator.platforms.ml_frameworks.torch.models.transformers.multirate import (
    DOCUMENT_PROTOTYPE_STATS,
    MultiRateTransformer,
    MultiRateTransformerConfig,
    MultiRateTaskSpec,
    Task,
    Corpus,
    Trainer,
    add_subtoken_temporal_tasks,
)
from quant_orchestrator.platforms.ml_frameworks.torch.models.transformers.multirate.temporal_tasks import (
    DOCUMENT_TASK_NAMES,
    FUND_ACTIVITY_SUPERVISED_TASK_NAMES,
    HOLDER_ACTIVITY_SUPERVISED_TASK_NAMES,
    HITS_SUPERVISED_TASK_NAMES,
    ORACLE_SUPERVISED_TASK_NAMES,
    PREDICTION_TASK_NAMES,
    SUPERVISED_TARGET_TASK_NAMES,
    TEMPORAL_MTL_TASK_NAMES,
)


ANNUAL_WINDOW = 252
QUARTERLY_WINDOW = 252
DAILY_WINDOW = 252  # one trading year for daily self-supervision
DEFAULT_EPOCHS = 20
OPTION_FEATURES = (
    "observed",
    "call_count",
    "put_count",
    "call_volume",
    "put_volume",
    "call_open_interest",
    "put_open_interest",
    "mean_dte",
    "mean_abs_moneyness",
    "mean_spread_pct",
    "mean_entry_bid",
    "mean_entry_ask",
)
def matryoshka_alignment_loss(embedding: torch.Tensor, dimensions: tuple[int, ...]) -> torch.Tensor:
    """Keep nested prefixes useful at multiple retrieval dimensions.

    The full document representation remains trained by the existing MTL
    objectives.  Each smaller normalized prefix is additionally aligned with
    the corresponding prefix of the full representation, which is the MRL
    objective used by downstream retrieval/coordinate consumers.
    """
    if not dimensions:
        return embedding.sum() * 0.0
    target = nn.functional.normalize(embedding.detach(), dim=-1)
    losses = []
    for dimension in dimensions:
        prefix = nn.functional.normalize(embedding[:, :dimension], dim=-1)
        losses.append(1.0 - (prefix * target[:, :dimension]).sum(dim=-1).mean())
    return torch.stack(losses).mean()

def _as_datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    if isinstance(value, torch.Tensor):
        value = value.item()
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value) / 1_000_000_000, tz=timezone.utc).replace(tzinfo=None)
    text = str(value).replace("Z", "+00:00")
    return datetime.fromisoformat(text).replace(tzinfo=None)


def _epoch_ns(value: object) -> int:
    parsed = _as_datetime(value).replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1_000_000_000)


def _is_missing(value: object) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _encode_labels(values: pl.Series) -> tuple[torch.Tensor, list[str], dict[str, int]]:
    normalized = values.cast(pl.String).fill_null("Unknown")
    labels = sorted(normalized.unique().to_list())
    mapping = {value: index for index, value in enumerate(labels)}
    return torch.tensor([mapping[value] for value in normalized.to_list()], dtype=torch.long), labels, mapping


def _symbol_rows(table: pl.DataFrame, symbol: str) -> pl.DataFrame:
    return table.filter(pl.col("symbol").cast(pl.String).str.to_uppercase() == str(symbol).upper())


def _read_parquet_polars(path: Path, columns: list[str] | None = None) -> pl.DataFrame:
    """Read parquet with Polars streaming and keep it as a Polars table."""
    scan = pl.scan_parquet(path)
    if columns is not None:
        scan = scan.select(columns)
    return scan.collect(engine="streaming")


class _IndexedTable:
    """Columnar, per-symbol view used by the sample builder.

    The previous path performed a Pandas filter and tail for every document
    and rate.  This index keeps sorted NumPy arrays and uses searchsorted for
    O(log n) window lookup without allocating a DataFrame per sample.
    """

    def __init__(self, table: pl.DataFrame, value_columns: list[str], *, target_column: str | None = None):
        self.value_columns = tuple(value_columns)
        self.rows: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]] = {}
        ordered = table.sort(["symbol", "date"])
        for group in ordered.partition_by("symbol", maintain_order=True):
            symbol = str(group["symbol"][0]).upper()
            dates = group["date"].cast(pl.Datetime, strict=False).cast(pl.Int64).to_torch().flatten()
            values = group.select(value_columns).fill_nan(None).fill_null(float("nan")).to_torch().to(torch.float32)
            targets = group[target_column].to_torch().to(torch.long) if target_column else None
            self.rows[symbol] = (dates, values, targets)

    def save_memmap(self, directory: Path, name: str) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        symbols = sorted(self.rows)
        offsets = [0]
        date_parts = []; value_parts = []; target_parts = []
        has_targets = any(self.rows[symbol][2] is not None for symbol in symbols)
        for symbol in symbols:
            dates, values, targets = self.rows[symbol]
            date_parts.append(dates); value_parts.append(values)
            if has_targets:
                target_parts.append(targets if targets is not None else torch.full((len(dates),), -1, dtype=torch.long))
            offsets.append(offsets[-1] + len(dates))
        torch.save(torch.cat(date_parts) if date_parts else torch.empty(0, dtype=torch.long), directory / f"{name}_dates.pt")
        torch.save(torch.cat(value_parts) if value_parts else torch.empty((0, len(self.value_columns)), dtype=torch.float32), directory / f"{name}_values.pt")
        if has_targets:
            torch.save(torch.cat(target_parts), directory / f"{name}_targets.pt")
        (directory / f"{name}_index.json").write_text(json.dumps({
            "symbols": symbols, "offsets": offsets, "value_columns": list(self.value_columns), "has_targets": has_targets,
        }))

    @classmethod
    def from_memmap(cls, directory: Path, name: str) -> "_IndexedTable":
        metadata = json.loads((directory / f"{name}_index.json").read_text())
        instance = cls.__new__(cls)
        instance.value_columns = tuple(metadata["value_columns"])
        dates = torch.load(directory / f"{name}_dates.pt", weights_only=True)
        values = torch.load(directory / f"{name}_values.pt", weights_only=True)
        targets = torch.load(directory / f"{name}_targets.pt", weights_only=True) if metadata["has_targets"] else None
        offsets = metadata["offsets"]
        instance.rows = {
            symbol: (dates[offsets[i]:offsets[i + 1]], values[offsets[i]:offsets[i + 1]], targets[offsets[i]:offsets[i + 1]] if targets is not None else None)
            for i, symbol in enumerate(metadata["symbols"])
        }
        return instance

    def window(self, symbol: str, anchor: datetime, length: int):
        dates, values, targets = self.rows.get(str(symbol).upper(), (
            torch.empty(0, dtype=torch.long),
            torch.empty((0, len(self.value_columns)), dtype=torch.float32),
            None,
        ))
        stop = int(torch.searchsorted(dates, torch.tensor(_epoch_ns(anchor), dtype=torch.long), right=True))
        start = max(0, stop - length)
        selected = values[start:stop]
        output = torch.full((length, len(self.value_columns)), float("nan"), dtype=torch.float32)
        padding = torch.ones(length, dtype=torch.bool)
        if len(selected):
            output[-len(selected):] = selected
            padding[-len(selected):] = False
        selected_dates = dates[start:stop]
        return output, padding, selected_dates, (targets[start:stop] if targets is not None else None)


class _LazySample(dict):
    """Scalar sample metadata with on-demand rate-array materialization."""

    _LAZY_KEYS = frozenset({
        "annual", "annual_padding", "quarterly", "quarterly_padding",
        "daily", "daily_padding", "daily_dates", "sparse", "sparse_padding",
        "sparse_labels", "supervised_targets", "supervised_valid",
    })

    def __init__(self, metadata: dict[str, object], factory):
        super().__init__(metadata)
        self._factory = factory
        self._loaded = False

    def _materialize(self) -> None:
        if not self._loaded:
            super().update(self._factory())
            self._loaded = True

    def __getitem__(self, key):
        if key in self._LAZY_KEYS:
            self._materialize()
        return super().__getitem__(key)


def _canonical_issuer_key(profile: object | None, symbol: str) -> str:
    cik = str(getattr(profile, "cik", None) or "").strip()
    if cik and cik.lower() not in {"none", "nan"}:
        return f"cik:{cik}"
    company_name = " ".join(str(getattr(profile, "company_name", None) or "").split()).casefold()
    if company_name and company_name not in {"none", "nan"}:
        return f"name:{company_name}"
    return f"symbol:{str(symbol).strip().upper()}"


def _window(table: pl.DataFrame, symbol: str, anchor: datetime, value_columns: list[str], length: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if isinstance(table, _IndexedTable):
        values, padding, dates, _ = table.window(symbol, anchor, length)
        return values, padding, dates
    rows = _symbol_rows(table, symbol).filter(pl.col("date") <= _as_datetime(anchor)).tail(length)
    values = rows.select(value_columns).fill_nan(None).fill_null(float("nan")).to_torch().to(torch.float32) if len(rows) else torch.empty((0, len(value_columns)), dtype=torch.float32)
    dates = rows["date"].cast(pl.Datetime, strict=False).cast(pl.Int64).to_torch().flatten() if len(rows) else torch.empty(0, dtype=torch.long)
    padding = torch.ones(length, dtype=torch.bool)
    output = torch.full((length, len(value_columns)), float("nan"), dtype=torch.float32)
    if len(rows):
        output[-len(rows):] = values
        padding[-len(rows):] = False
    return output, padding, dates


def _sparse_window(table: pl.DataFrame, symbol: str, anchor: datetime, value_columns: list[str], length: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if isinstance(table, _IndexedTable):
        values, padding, dates, targets = table.window(symbol, anchor, length)
        labels = torch.full((length,), -1, dtype=torch.long)
        if targets is not None and len(targets):
            labels[-len(targets):] = targets
        return values, padding, labels, dates
    rows = _symbol_rows(table, symbol).filter(pl.col("date") <= _as_datetime(anchor)).tail(length)
    values = torch.full((length, len(value_columns)), float("nan"), dtype=torch.float32)
    labels = torch.full((length,), -1, dtype=torch.long)
    padding = torch.ones(length, dtype=torch.bool)
    dates = torch.empty(0, dtype=torch.long)
    if len(rows):
        values[-len(rows):] = rows.select(value_columns).fill_nan(None).fill_null(float("nan")).to_torch().to(torch.float32)
        labels[-len(rows):] = rows["target_id"].to_torch().to(torch.long)
        padding[-len(rows):] = False
        dates = rows["date"].cast(pl.Datetime, strict=False).cast(pl.Int64).to_torch().flatten()
    return values, padding, labels, dates


def _relative_dates(dates: torch.Tensor, length: int) -> torch.Tensor:
    # Windows are causal and left-padded. Relative ordering is shared across
    # a batch; sparse rows are aggregated to one row per availability date.
    result = torch.arange(length, dtype=torch.long)
    return result


def _add_option_state_features(
    daily: pl.DataFrame,
    option_panel: pl.DataFrame,
    *,
    max_contracts_per_type: int,
) -> tuple[pl.DataFrame, list[str]]:
    """Attach leakage-safe, as-of option state features to daily rows.

    The option panel contains entry-time chain descriptors and future outcome
    columns. Only entry-time fields are used here. State is forward-filled
    within each symbol so a missing chain day remains missing until the first
    observed chain, then represents the latest known chain as-of that date.
    """
    required = {"symbol", "entry_date", "option_type"}
    missing = required - set(option_panel.columns)
    if missing:
        raise ValueError(f"option panel missing required columns: {sorted(missing)}")
    options = option_panel
    options = options.with_columns(
        pl.col("symbol").cast(pl.String).str.to_uppercase().str.strip_chars(),
        pl.col("entry_date").cast(pl.Datetime, strict=False).dt.truncate("1d"),
        pl.col("option_type").cast(pl.String).str.to_lowercase().str.strip_chars(),
    ).filter(pl.col("symbol").is_not_null() & pl.col("entry_date").is_not_null())
    if max_contracts_per_type > 0:
        options = options.with_columns(
            pl.col("volume").cast(pl.Float64, strict=False).alias("_volume_value"),
            pl.col("open_interest").cast(pl.Float64, strict=False).alias("_open_interest_value"),
        ).sort(
            ["symbol", "entry_date", "option_type", "_volume_value", "_open_interest_value"],
            descending=[False, False, False, True, True],
            nulls_last=True,
        ).with_columns(
            pl.int_range(0, pl.len()).over(["symbol", "entry_date", "option_type"]).alias("_contract_rank")
        ).filter(pl.col("_contract_rank") < max_contracts_per_type)

    def numeric(name: str) -> pl.Expr:
        return pl.col(name).cast(pl.Float64, strict=False) if name in options.columns else pl.lit(None, dtype=pl.Float64)

    options = options.with_columns(
        numeric("volume").alias("_volume_value"),
        numeric("open_interest").alias("_open_interest_value"),
        numeric("dte").alias("_dte_value"),
        numeric("abs_moneyness").alias("_abs_moneyness_value"),
        (numeric("entry_bid") if "entry_bid" in options.columns else numeric("bid")).alias("_entry_bid_value"),
        (numeric("entry_ask") if "entry_ask" in options.columns else numeric("ask")).alias("_entry_ask_value"),
    ).with_columns(
        pl.col("_volume_value").fill_null(pl.col("_volume_value").mean().over(["symbol", "entry_date"])),
        pl.col("_open_interest_value").fill_null(pl.col("_open_interest_value").mean().over(["symbol", "entry_date"])),
    )
    groups = ["symbol", "entry_date"]

    def weighted_quote(column: str) -> pl.Expr:
        valid = pl.col(column).gt(0) & pl.col("_volume_value").gt(0)
        weighted_sum = pl.when(valid).then(pl.col(column) * pl.col("_volume_value")).otherwise(0).sum()
        weight_sum = pl.when(valid).then(pl.col("_volume_value")).otherwise(0).sum()
        fallback = pl.when(pl.col(column).gt(0)).then(pl.col(column)).otherwise(None).mean()
        return pl.when(weight_sum.gt(0)).then(weighted_sum / weight_sum).otherwise(fallback)

    weighted_bid = weighted_quote("_entry_bid_value")
    weighted_ask = weighted_quote("_entry_ask_value")
    state = options.group_by(groups, maintain_order=True).agg(
        pl.lit(1.0).alias("value__options__observed"),
        pl.col("option_type").eq("call").sum().cast(pl.Float64).alias("value__options__call_count"),
        pl.col("option_type").eq("put").sum().cast(pl.Float64).alias("value__options__put_count"),
        pl.when(pl.col("option_type").eq("call")).then(pl.col("_volume_value")).otherwise(None).sum().alias("value__options__call_volume"),
        pl.when(pl.col("option_type").eq("put")).then(pl.col("_volume_value")).otherwise(None).sum().alias("value__options__put_volume"),
        pl.when(pl.col("option_type").eq("call")).then(pl.col("_open_interest_value")).otherwise(None).sum().alias("value__options__call_open_interest"),
        pl.when(pl.col("option_type").eq("put")).then(pl.col("_open_interest_value")).otherwise(None).sum().alias("value__options__put_open_interest"),
        pl.col("_dte_value").mean().alias("value__options__mean_dte"),
        pl.col("_abs_moneyness_value").mean().alias("value__options__mean_abs_moneyness"),
        ((weighted_ask - weighted_bid) / weighted_bid.abs()).alias("value__options__mean_spread_pct"),
        weighted_bid.alias("value__options__mean_entry_bid"),
        weighted_ask.alias("value__options__mean_entry_ask"),
    ).rename({"entry_date": "date"})
    option_columns = [f"value__options__{name}" for name in OPTION_FEATURES]
    state = state.select([
        "symbol", "date",
        *[pl.col(column) if column in state.columns else pl.lit(None).alias(column) for column in option_columns],
    ])
    daily = daily.with_columns(
        pl.col("symbol").cast(pl.String).str.to_uppercase().str.strip_chars(),
        pl.col("date").cast(pl.Datetime, strict=False).dt.truncate("1d"),
    ).sort(["symbol", "date"])
    daily = daily.join(state, on=["symbol", "date"], how="left", suffix="_option")
    daily = daily.with_columns([
        pl.col(column).forward_fill().over("symbol").cast(pl.Float32).alias(column)
        for column in option_columns
    ])
    return daily, option_columns


def _issuer_dte_bin_option_panel(
    panel: pl.DataFrame,
    taxonomy: pl.DataFrame,
    *,
    bin_count: int,
) -> pl.DataFrame:
    """Freeze representative weighted DTE quantiles independently per issuer.

    ``bin_count`` representative DTE groups are selected at the interior
    quantiles of each issuer's first usable option date. For five bins this is
    Q10/Q30/Q50/Q70/Q90, matching the prior three-bin Q25/Q50/Q75 behavior.
    """
    if bin_count < 1:
        raise ValueError("option issuer DTE bin count must be at least 1")
    issuer_map = dict(zip(taxonomy["symbol"].to_list(), taxonomy["issuer"].to_list()))
    result = panel.with_columns(
        pl.col("underlying_symbol").cast(pl.String).str.to_uppercase().str.strip_chars(),
        pl.col("entry_date").cast(pl.Datetime, strict=False).dt.truncate("1d"),
        pl.col("dte").cast(pl.Int64, strict=False),
    ).with_columns(
        pl.col("underlying_symbol").replace(issuer_map, default=None).alias("_issuer"),
        (pl.col("entry_bid") if "entry_bid" in panel.columns else pl.col("bid")).cast(pl.Float64, strict=False).alias("_bid"),
        (pl.col("entry_ask") if "entry_ask" in panel.columns else pl.col("ask")).cast(pl.Float64, strict=False).alias("_ask"),
        (pl.col("dte_contract_count") if "dte_contract_count" in panel.columns else pl.lit(1.0)).cast(pl.Float64, strict=False).fill_null(1.0).alias("_weight"),
    ).drop_nulls(["_issuer", "entry_date", "dte"])
    selected: list[pl.DataFrame] = []
    for group in result.partition_by("_issuer", maintain_order=True):
        usable = group.filter((pl.col("_bid") > 0) & (pl.col("_ask") > 0))
        first_date = (usable["entry_date"].min() if len(usable) else group["entry_date"].min())
        first = group.filter((pl.col("entry_date") == first_date) & (pl.col("dte") >= 0))
        if not len(first):
            continue
        pairs = sorted((int(dte), max(float(weight), 1.0)) for dte, weight in first.select(["dte", "_weight"]).iter_rows())
        values = [pair[0] for pair in pairs]
        total_weight = sum(pair[1] for pair in pairs)
        cumulative = []
        running = 0.0
        for _, weight in pairs:
            running += weight
            cumulative.append(running / total_weight)
        quantiles = [(index + 0.5) / bin_count for index in range(bin_count)]
        targets = [next(values[index] for index, cumulative_value in enumerate(cumulative) if cumulative_value >= quantile) for quantile in quantiles]
        targets = set(targets)
        selected.append(group.filter(pl.col("dte").is_in(list(targets))))
    if not selected:
        raise ValueError("issuer-specific DTE bin selection produced no option rows")

    selected_panel = pl.concat(selected, how="vertical_relaxed")
    # Collapse the selected strike grid into one synthetic contract per
    # underlying/date/type/DTE. The model therefore sees the mean of the
    # original contract-level features during training, while live inference
    # can pass one real contract through the same feature columns unchanged.
    group_cols = ["underlying_symbol", "entry_date", "option_type", "dte"]
    if "side" in selected_panel.columns:
        group_cols.append("side")
    selected_panel = selected_panel.with_columns(pl.col("dte").round().cast(pl.Int64))
    # Preserve the executable economics of the bucket.  A plain arithmetic
    # mean lets illiquid contracts influence the synthetic quote as much as
    # liquid contracts.  Use traded volume when available and open interest
    # as a fallback; if neither exists, every valid contract gets equal
    # weight.  Bid and ask are kept separate so later labels can use the ask
    # for long entries and the bid for short entries.
    volume = pl.col("volume").cast(pl.Float64, strict=False) if "volume" in selected_panel else pl.lit(None, dtype=pl.Float64)
    open_interest = pl.col("open_interest").cast(pl.Float64, strict=False) if "open_interest" in selected_panel else pl.lit(None, dtype=pl.Float64)
    selected_panel = selected_panel.with_columns(
        pl.when(volume.fill_null(0) > 0).then(volume).otherwise(open_interest).fill_null(0).clip(lower_bound=0).alias("_quote_weight")
    ).with_columns(pl.when(pl.col("_quote_weight") > 0).then(pl.col("_quote_weight")).otherwise(1.0).alias("_quote_weight"))

    selected_pl = selected_panel
    numeric_cols = [name for name, dtype in zip(selected_pl.columns, selected_pl.dtypes) if dtype.is_numeric() and name != "dte"]
    quote_columns = [name for name in ("entry_bid", "entry_ask", "bid", "ask", "mid") if name in selected_pl.columns]
    aggregations: list[pl.Expr] = [
        pl.col(column).mean().alias(column)
        for column in numeric_cols
        if column not in {"_quote_weight", *quote_columns}
    ]
    for column in selected_pl.columns:
        if column in group_cols or column in numeric_cols or column in quote_columns or column in {"symbol", "contract_symbol", "_quote_weight"}:
            continue
        aggregations.append(pl.col(column).first().alias(column))
    for column in quote_columns:
        value = pl.col(column).cast(pl.Float64, strict=False)
        valid = value.gt(0) & pl.col("_quote_weight").gt(0)
        weighted_sum = pl.when(valid).then(value * pl.col("_quote_weight")).otherwise(0.0).sum()
        weight_sum = pl.when(valid).then(pl.col("_quote_weight")).otherwise(0.0).sum()
        aggregations.append(
            pl.when(weight_sum.gt(0)).then(weighted_sum / weight_sum)
            .otherwise(None).alias(column)
        )
    aggregated = selected_pl.group_by(group_cols, maintain_order=True).agg(aggregations)
    counts = selected_pl.group_by(group_cols, maintain_order=True).len(name="synthetic_contract_count")
    aggregated = aggregated.join(counts, on=group_cols, how="left").with_columns(
        (pl.lit("OPT_SYNTH_") + pl.col("underlying_symbol") + pl.lit("_") + pl.col("option_type").str.slice(0, 1).str.to_uppercase() + pl.lit("_DTE") + pl.col("dte").cast(pl.String)).alias("symbol")
    ).with_columns(pl.col("symbol").alias("contract_symbol"))
    if "expiration" in aggregated.columns:
        aggregated = aggregated.with_columns((pl.col("entry_date") + pl.duration(days=pl.col("dte"))).alias("expiration"))
    aggregated = aggregated.with_columns(pl.lit(True).alias("synthetic_option"))
    return aggregated


def _filter_universe(
    taxonomy: pl.DataFrame,
    *,
    country: str,
    currency: str,
    exchanges: set[str],
    allow_unresolved_profiles: bool = False,
) -> tuple[set[str], list[str]]:
    """Apply the investable US-equity universe filter using catalog profiles."""
    wanted = {str(symbol).upper() for symbol in taxonomy["symbol"].to_list()}
    profiles = Warehouse().catalog.query_symbol_profiles(
        provider="fmp", min_market_cap=0, country="", exchanges=(),
        exclude_etf=False, exclude_fund=False, limit=100_000,
    )
    by_symbol = {str(profile.symbol).strip().upper(): profile for profile in profiles}
    keep: set[str] = set()
    unresolved_currency: list[str] = []
    for symbol in wanted:
        profile = by_symbol.get(symbol)
        if profile is None:
            if allow_unresolved_profiles and not symbol.startswith("OPT_") and "." not in symbol:
                keep.add(symbol)
                unresolved_currency.append(symbol)
            continue
        profile_country = str(getattr(profile, "country", "") or "").upper().strip()
        profile_exchange = str(getattr(profile, "exchange", "") or "").upper().strip()
        if country and profile_country != country.upper():
            continue
        if exchanges and profile_exchange not in exchanges:
            continue
        profile_currency = str(getattr(profile, "currency", "") or "").upper().strip()
        if currency and profile_currency and profile_currency != currency.upper():
            continue
        if currency and not profile_currency:
            # The local FMP catalog does not expose currency for these rows;
            # US-listed common stocks are treated as USD and recorded below.
            unresolved_currency.append(symbol)
        keep.add(symbol)
    return keep, unresolved_currency


def _add_executable_option_return(options: pl.DataFrame) -> pl.DataFrame:
    """Use ask-to-bid execution for option-return supervision."""
    def number(name: str, fallback: float | None = None) -> pl.Expr:
        return pl.col(name).cast(pl.Float64, strict=False) if name in options.columns else pl.lit(fallback, dtype=pl.Float64)
    entry_mid = number("entry_mid")
    exit_mid = number("exit_mid")
    spread = number("spread_pct", 0.0).fill_null(0.0).clip(0.0, 1.0)
    entry_ask = number("entry_ask") if "entry_ask" in options.columns else entry_mid * (1.0 + spread / 2.0)
    exit_bid = number("exit_bid") if "exit_bid" in options.columns else exit_mid * (1.0 - spread / 2.0)
    return options.with_columns(
        pl.when((entry_ask > 0.0) & exit_bid.is_not_null()).then(exit_bid / entry_ask - 1.0).otherwise(None).alias("execution_return")
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, help="Load an existing multirate_mtl_model.pt for inference without optimizer steps.")
    parser.add_argument("--inference-only", action="store_true", help="Skip training and export predictions from --checkpoint.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--progress-every-batches", type=int, default=100,
        help="Print training progress every N batches; 0 disables batch progress logging.",
    )
    parser.add_argument(
        "--checkpoint-every-batches", type=int, default=100,
        help="Save a resumable checkpoint every N batches; 0 disables batch checkpoints.",
    )
    parser.add_argument("--grad-accumulation-steps", type=int, default=1)
    parser.add_argument(
        "--context-cache-size", type=int, default=256,
        help="Maximum issuer/date windows cached per rate; 0 disables the training-data LRU.",
    )
    parser.add_argument("--context-memmap-dir", type=Path, help="Optional prepared normalized context-array cache directory.")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--min-delta", type=float, default=1e-3)
    parser.add_argument(
        "--validation-fraction", type=float, default=0.0,
        help="Chronological holdout fraction; 0 (the default) trains on every sample with no validation split.",
    )
    parser.add_argument("--learned-aggregation-gate", action="store_true")
    parser.add_argument(
        "--legacy-rate-fusion",
        action="store_true",
        help="Use the original decoder-conditioned rate fusion used by exact_date artifacts.",
    )
    parser.add_argument(
        "--mrl-dimensions", default="16,32,64,128",
        help="Comma-separated nested embedding dimensions for MRL; empty disables MRL.",
    )
    parser.add_argument("--mrl-weight", type=float, default=0.25)
    parser.add_argument("--mixed-precision", action="store_true", help="Use CUDA autocast (float16 with GradScaler).")
    parser.add_argument(
        "--attention-backend", choices=("pytorch", "transformer_engine"), default="pytorch",
        help="Transformer attention implementation; transformer_engine requires quant-orchestrator[cuda-te].",
    )
    parser.add_argument(
        "--fp8", action="store_true",
        help="Use Transformer Engine FP8 autocasting; requires --attention-backend transformer_engine.",
    )
    parser.add_argument("--compile-model", action="store_true", help="Compile the model with torch.compile.")
    parser.add_argument("--optimizer", choices=("adamw", "adamw8bit"), default="adamw")
    parser.add_argument("--skip-embeddings", action="store_true", help="Do not retain or write evaluation embeddings.")
    parser.add_argument("--skip-t-sne", action="store_true", help="Skip prototype t-SNE generation.")
    parser.add_argument("--skip-predictions", action="store_true", help="Do not export daily supervised-head predictions.")
    parser.add_argument("--train-end-date", help="Train only on document anchors before this YYYY-MM-DD date.")
    parser.add_argument("--train-symbols-file", type=Path, help="CSV of symbols permitted for training.")
    parser.add_argument("--test-symbols-file", type=Path, help="CSV of symbols reserved for evaluation.")
    parser.add_argument("--prediction-start-date", help="Export daily supervised-head scores on and after this YYYY-MM-DD date.")
    parser.add_argument("--option-panel", type=Path, help="Optional entry-time option candidate panel to add as-of daily state features.")
    parser.add_argument("--option-target-events", type=Path, help="Option-native HITS/Oracle sparse targets generated from bid/ask baskets.")
    parser.add_argument("--option-max-contracts", type=int, default=32, help="Per-symbol/date/type option rows retained by volume before aggregation; 0 keeps all rows.")
    parser.add_argument("--option-start-date", default="2025-01-01", help="Earliest option entry date used for features and supervision.")
    parser.add_argument("--option-end-date", help="Optional latest option entry date used for features and supervision.")
    parser.add_argument("--option-dte", type=int, nargs="+", help="Restrict option training documents to one or more frozen DTE groups.")
    parser.add_argument(
        "--option-issuer-dte-bins",
        type=int,
        default=0,
        metavar="N",
        help="Freeze N representative weighted DTE groups independently for each issuer (5 uses Q10/Q30/Q50/Q70/Q90).",
    )
    parser.add_argument("--country", default="US", help="Issuer country filter; empty disables it.")
    parser.add_argument("--currency", default="USD", help="Trading currency filter; empty disables it.")
    parser.add_argument("--exchanges", default="NYSE,NASDAQ,AMEX", help="Comma-separated allowed exchanges; empty disables it.")
    parser.add_argument("--allow-unresolved-profiles", action="store_true", help="Keep plain source symbols missing from the local profile catalog; foreign-suffix symbols remain excluded by profile filtering.")
    parser.add_argument(
        "--disable-document-tasks",
        action="store_true",
        help="Disable document classification heads for an apples-to-apples timing benchmark.",
    )
    parser.add_argument(
        "--sample-build-only",
        action="store_true",
        help="Build and cache all samples, write cache metrics, then stop before model construction (benchmarking only).",
    )
    parser.add_argument(
        "--stream-samples", action="store_true",
        help="Keep sample metadata in memory and materialize rate arrays only per batch.",
    )
    parser.add_argument(
        "--max-samples", type=int, default=0,
        help="Optional deterministic cap on samples for a fast end-to-end smoke run.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.inference_only and args.checkpoint is None:
        parser.error("--inference-only requires --checkpoint")
    if args.skip_predictions:
        args.prediction_start_date = None
    mrl_dimensions = tuple(sorted({int(value) for value in args.mrl_dimensions.split(",") if value.strip()}, key=int))
    if mrl_dimensions and (mrl_dimensions[-1] != args.d_model or mrl_dimensions[0] < 1):
        parser.error("--mrl-dimensions must be positive and include --d-model as its largest dimension")
    if args.mrl_weight < 0:
        parser.error("--mrl-weight must be non-negative")
    enabled_document_tasks = () if args.disable_document_tasks else DOCUMENT_TASK_NAMES
    if args.grad_accumulation_steps < 1:
        parser.error("--grad-accumulation-steps must be at least 1")
    if args.context_cache_size < 0:
        parser.error("--context-cache-size must be non-negative")
    if args.mixed_precision and args.device == "cpu":
        parser.error("--mixed-precision requires a CUDA device")
    if args.attention_backend == "transformer_engine" and not args.device.startswith("cuda"):
        parser.error("--attention-backend transformer_engine requires a CUDA device")
    if args.fp8 and args.attention_backend != "transformer_engine":
        parser.error("--fp8 requires --attention-backend transformer_engine")
    option_dtes = set(args.option_dte or ())
    root = args.corpus
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root / "manifest.json").read_text())
    feature_families = list(manifest["feature_families"])
    target_families = list(manifest["target_families"])
    # The four-rate architecture keeps a sparse stream even for a fresh
    # feature-only build with no target-event parquet yet.  A neutral family
    # preserves the tensor contract without creating a supervised task or
    # contributing any labels.
    if not target_families:
        target_families = ["__empty_sparse_family__"]
    checkpoint_payload = None
    if args.inference_only and args.checkpoint is not None:
        checkpoint_payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        checkpoint_metrics = checkpoint_payload.get("metrics", {}) if isinstance(checkpoint_payload, dict) else {}
        # The checkpoint defines the sparse target schema used by its task
        # heads. Fresh live data may contain only a subset of those events.
        target_families = list(checkpoint_metrics.get("target_families", target_families))
    taxonomy = pl.read_csv(root / "taxonomy.csv").with_columns(
        pl.col("symbol").cast(pl.String).str.to_uppercase().str.strip_chars()
    )
    # Presence columns are useful for corpus diagnostics but are not model
    # inputs.  Avoid materializing them for the large 10B corpus.
    rate_columns = ["symbol", "date", *[f"value__{family}" for family in feature_families]]
    annual = _read_parquet_polars(root / "annual.parquet", rate_columns)
    quarterly = _read_parquet_polars(root / "quarterly.parquet", rate_columns)
    daily = _read_parquet_polars(root / "daily.parquet", rate_columns)
    sparse_path = root / "sparse_events.parquet"
    if sparse_path.exists():
        sparse = _read_parquet_polars(sparse_path)
    else:
        sparse = pl.DataFrame({
            "symbol": pl.Series([], dtype=pl.String),
            "date": pl.Series([], dtype=pl.Datetime),
            "event_date": pl.Series([], dtype=pl.Datetime),
            "target_family": pl.Series([], dtype=pl.String),
            "signal_value": pl.Series([], dtype=pl.Float32),
            **{f"text_{i}": pl.Series([], dtype=pl.Float32) for i in range(7)},
        })
    # Apply a frozen DTE selection before normalization/index construction.
    # Otherwise a DTE-105 run needlessly scans every synthetic option symbol
    # in the full daily table.
    issuer_quartile_panel = None
    if args.option_issuer_dte_bins < 0:
        parser.error("--option-issuer-dte-bins must be non-negative")
    if (option_dtes or args.option_issuer_dte_bins) and args.option_panel is not None:
        selected_panel = _read_parquet_polars(args.option_panel, None if args.option_issuer_dte_bins else ["symbol", "dte", "underlying_symbol"])
        if args.option_issuer_dte_bins:
            issuer_quartile_panel = _issuer_dte_bin_option_panel(
                selected_panel,
                taxonomy,
                bin_count=args.option_issuer_dte_bins,
            )
            selected_panel = issuer_quartile_panel
            selected_symbols = set(selected_panel["symbol"].cast(pl.String).str.to_uppercase().to_list())
            def keep_selected(frame: pl.DataFrame) -> pl.DataFrame:
                return frame.filter(~pl.col("symbol").cast(pl.String).str.to_uppercase().str.starts_with("OPT_") | pl.col("symbol").cast(pl.String).str.to_uppercase().is_in(list(selected_symbols)))
        else:
            selected_panel = selected_panel.with_columns(pl.col("dte").cast(pl.Int64, strict=False))
            selected_dte = selected_panel.filter(pl.col("dte").is_in(list(option_dtes)))
            selected_symbols = set(selected_dte["symbol"].cast(pl.String).str.to_uppercase().to_list())
            selected_underlyings = set(selected_dte["underlying_symbol"].cast(pl.String).str.to_uppercase().to_list())
            def keep_selected(frame: pl.DataFrame) -> pl.DataFrame:
                symbols = pl.col("symbol").cast(pl.String).str.to_uppercase()
                return frame.filter(~symbols.str.starts_with("OPT_") | symbols.is_in(list(selected_symbols | selected_underlyings)))
        annual = keep_selected(annual)
        quarterly = keep_selected(quarterly)
        daily = keep_selected(daily)
        sparse = keep_selected(sparse)
    if args.option_target_events is not None:
        option_events = _read_parquet_polars(args.option_target_events)
        if not option_events.is_empty():
            option_symbols = set(option_events["symbol"].astype(str).str.upper())
            sparse_symbols = pl.col("symbol").cast(pl.String).str.to_uppercase()
            replace_families = {"equity.strategy.hits_graph", "equity.strategy.oracle_trades"}
            sparse = sparse.filter(~(sparse_symbols.is_in(list(option_symbols)) & pl.col("target_family").is_in(list(replace_families))))
            sparse = pl.concat([sparse, option_events], how="diagonal_relaxed")
    exchanges = {value.strip().upper() for value in args.exchanges.split(",") if value.strip()}
    universe_symbols, unresolved_currency = _filter_universe(
        taxonomy, country=args.country.strip(), currency=args.currency.strip(), exchanges=exchanges,
        allow_unresolved_profiles=args.allow_unresolved_profiles,
    ) if (args.country.strip() or args.currency.strip() or exchanges) else (set(taxonomy["symbol"].to_list()), [])
    taxonomy = taxonomy.filter(pl.col("symbol").is_in(list(universe_symbols)))
    if taxonomy.is_empty():
        raise ValueError("universe filters removed every corpus symbol")
    if "issuer" not in taxonomy.columns:
        profile_rows = Warehouse().catalog.query_symbol_profiles(
            provider="fmp", min_market_cap=0, country="", exchanges=(),
            exclude_etf=False, exclude_fund=False, limit=100_000,
        )
        profiles_by_symbol = {str(profile.symbol).strip().upper(): profile for profile in profile_rows}
        taxonomy = taxonomy.with_columns(pl.Series("issuer", [
            _canonical_issuer_key(profiles_by_symbol.get(str(symbol).upper()), str(symbol))
            for symbol in taxonomy["symbol"].to_list()
        ]))
    def normalize_table(table: pl.DataFrame) -> pl.DataFrame:
        return table.with_columns(
            pl.col("symbol").cast(pl.String).str.to_uppercase().str.strip_chars(),
            pl.col("date").cast(pl.Datetime, strict=False).dt.truncate("1d"),
        )
    annual = normalize_table(annual)
    quarterly = normalize_table(quarterly)
    daily = normalize_table(daily)
    sparse = normalize_table(sparse)
    option_columns: list[str] = []
    option_target_map: dict[tuple[str, datetime], torch.Tensor] = {}
    option_document_symbols: set[str] = set()
    option_document_start_dates: dict[str, datetime] = {}
    source_symbol_by_symbol: dict[str, str] = {}
    option_entry_anchors = pl.DataFrame({"symbol": pl.Series([], dtype=pl.String), "date": pl.Series([], dtype=pl.Datetime)})
    if args.option_panel is not None:
        option_panel = issuer_quartile_panel if issuer_quartile_panel is not None else _read_parquet_polars(args.option_panel)
        option_panel = _add_executable_option_return(option_panel)
        option_panel = option_panel.with_columns(pl.col("entry_date").cast(pl.Datetime, strict=False).dt.truncate("1d"))
        option_panel = option_panel.filter(pl.col("entry_date") >= _as_datetime(args.option_start_date))
        if args.option_end_date:
            option_panel = option_panel.filter(pl.col("entry_date") <= _as_datetime(args.option_end_date))
        if option_dtes:
            option_panel = option_panel.filter(pl.col("dte").cast(pl.Int64, strict=False).is_in(list(option_dtes)))
        option_entry_anchors = option_panel.select([pl.col("symbol"), pl.col("entry_date").alias("date")]).unique()
        option_state_panel = option_panel
        if "underlying_symbol" in option_state_panel:
            option_state_panel = option_state_panel.with_columns(pl.col("underlying_symbol").alias("symbol"))
        daily, option_columns = _add_option_state_features(
            daily,
            option_state_panel,
            max_contracts_per_type=max(0, args.option_max_contracts),
        )
        annual = annual.with_columns([pl.lit(None, dtype=pl.Float32).alias(column) for column in option_columns])
        quarterly = quarterly.with_columns([pl.lit(None, dtype=pl.Float32).alias(column) for column in option_columns])
        option_panel = option_panel.with_columns(pl.col("symbol").cast(pl.String).str.to_uppercase().str.strip_chars())
        if "underlying_symbol" in option_panel:
            source_symbol_by_symbol.update({str(row[0]).upper(): str(row[1]).upper() for row in option_panel.select(["symbol", "underlying_symbol"]).drop_nulls().iter_rows()})
        option_document_symbols = set(option_panel["symbol"].drop_nulls().cast(pl.String).str.to_uppercase().to_list())
        if option_dtes and not option_document_symbols:
            raise ValueError(f"no option documents found for DTEs {sorted(option_dtes)}")
        if option_dtes:
            sparse = sparse.filter(~pl.col("symbol").cast(pl.String).str.to_uppercase().str.starts_with("OPT_") | pl.col("symbol").cast(pl.String).str.to_uppercase().is_in(list(option_document_symbols)))
        option_document_start_dates = {str(row[0]): row[1] for row in option_panel.drop_nulls(["symbol", "entry_date"]).group_by("symbol").agg(pl.col("entry_date").min()).iter_rows()}
        option_panel = option_panel.with_columns(
            pl.col("entry_date").cast(pl.Datetime, strict=False).dt.truncate("1d"),
            pl.col("side").cast(pl.String).str.to_lowercase().str.strip_chars(),
            pl.col("execution_return").cast(pl.Float64, strict=False),
        ).drop_nulls(["symbol", "entry_date", "execution_return"])
        for (symbol, date, side), group in option_panel.group_by(["symbol", "entry_date", "side"], maintain_order=True):
            if side not in {"long", "short"}:
                continue
            transformed = group["execution_return"].to_torch().to(torch.float64)
            transformed = torch.sign(transformed) * torch.log1p(torch.minimum(transformed.abs(), torch.tensor(1_000_000.0)))
            values = option_target_map.setdefault((str(symbol), _as_datetime(date)), torch.full((2,), float("nan"), dtype=torch.float32))
            values[0 if side == "long" else 1] = transformed.nanmean().to(torch.float32)
    if option_document_symbols:
        allowed_symbols = {symbol for symbol in taxonomy["symbol"].to_list() if not str(symbol).upper().startswith("OPT_")} | option_document_symbols
        taxonomy = taxonomy.filter(pl.col("symbol").is_in(list(allowed_symbols)))
        taxonomy_rows = {str(row["symbol"]): row for row in taxonomy.iter_rows(named=True)}
        synthetic_taxonomy_rows = []
        for option_symbol, underlying_symbol in source_symbol_by_symbol.items():
            if option_symbol not in taxonomy_rows and underlying_symbol in taxonomy_rows:
                row = dict(taxonomy_rows[underlying_symbol])
                row["symbol"] = option_symbol
                synthetic_taxonomy_rows.append(row)
        if synthetic_taxonomy_rows:
            taxonomy = pl.concat([taxonomy, pl.DataFrame(synthetic_taxonomy_rows)], how="vertical_relaxed")
    if "event_date" in sparse:
        sparse = sparse.with_columns(pl.col("event_date").cast(pl.Datetime, strict=False).dt.truncate("1d"))

    # Derived target labels are supervised at their original event date. Their
    # delayed availability remains in ``date`` and therefore keeps them out
    # of the input context at the prediction date.
    supervised_target_map: dict[tuple[str, datetime], dict[str, float]] = {}
    if "event_date" in sparse:
        target_channels = ["signal_value", *[f"text_{i}" for i in range(7)]]
        supervised_rows = sparse.filter(pl.col("target_family").is_in([
            "equity.strategy.hits_graph", "equity.strategy.oracle_trades",
        ])).iter_rows(named=True)
        for row in supervised_rows:
            event_date = row.get("event_date")
            if _is_missing(event_date):
                continue
            key = (str(row["symbol"]).upper(), _as_datetime(event_date))
            values = {channel: float(row[channel]) for channel in target_channels if not _is_missing(row.get(channel))}
            target_family = str(row["target_family"])
            targets = supervised_target_map.setdefault(key, {})
            if target_family == "equity.strategy.hits_graph":
                for task_name, channel in zip(HITS_SUPERVISED_TASK_NAMES, target_channels):
                    if channel in values:
                        targets[task_name] = max(targets.get(task_name, float("-inf")), values[channel])
            else:
                for task_name, channel in zip(ORACLE_SUPERVISED_TASK_NAMES, target_channels[:4]):
                    if channel in values:
                        targets[task_name] = max(targets.get(task_name, 0.0), values[channel])
            if target_family.startswith("fund_activity."):
                activity_name = target_family.removeprefix("fund_activity.")
                task_name = f"fund_activity_{activity_name}"
            if task_name in FUND_ACTIVITY_SUPERVISED_TASK_NAMES:
                targets[task_name] = max(targets.get(task_name, 0.0), values.get("signal_value", 0.0))
            if target_family.startswith("holder_activity."):
                activity_name = target_family.removeprefix("holder_activity.")
                task_name = f"holder_activity_{activity_name}"
                if task_name in HOLDER_ACTIVITY_SUPERVISED_TASK_NAMES:
                    targets[task_name] = max(targets.get(task_name, 0.0), values.get("signal_value", 0.0))

    daily_value_columns = [f"value__{family}" for family in feature_families]
    if option_columns:
        daily_value_columns.extend(option_columns)
    annual_value_columns = daily_value_columns
    quarterly_value_columns = daily_value_columns
    sparse = sparse.sort(["symbol", "date", "event_date"] if "event_date" in sparse.columns else ["symbol", "date"])
    # Multiple same-day disclosures become one sparse token while retaining
    # the first available family and the mean numeric/text representation.
    sparse = sparse.with_columns(pl.col("target_family").replace({name: i for i, name in enumerate(target_families)}, default=None).cast(pl.Int64).alias("target_id"))
    sparse_value_columns = ["signal_value", *[f"text_{i}" for i in range(7)]]
    sparse = sparse.group_by(["symbol", "date"], maintain_order=True).agg(
        pl.col("target_family").first(), pl.col("target_id").first(),
        *[pl.col(column).mean().alias(column) for column in sparse_value_columns],
    )

    # Standardize numeric rate values using the available corpus while
    # retaining NaN for coverage-aware missingness handling.
    rate_columns = {"annual": annual_value_columns, "quarterly": quarterly_value_columns, "daily": daily_value_columns}
    norms: dict[str, tuple[list[float], list[float]]] = {}
    for name, columns in rate_columns.items():
        values = pl.concat(
            [table.select(columns) for table in (annual, quarterly, daily)],
            how="vertical_relaxed",
        )
        stats = values.select(
            *[pl.col(column).fill_nan(None).mean().alias(f"mean_{column}") for column in columns],
            *[pl.col(column).fill_nan(None).std(ddof=0).alias(f"std_{column}") for column in columns],
        ).row(0)
        mean = [0.0 if value is None or not math.isfinite(float(value)) else float(value) for value in stats[:len(columns)]]
        scale = [float(value) if value is not None and math.isfinite(float(value)) and float(value) > 1e-6 else 1.0 for value in stats[len(columns):]]
        norms[name] = (mean, scale)
        # Keep normalization in Polars; the model boundary receives only the
        # resulting per-window tensors later.
        normalized = []
        for table in (annual, quarterly, daily):
            normalized.append(table.with_columns([
                ((pl.col(column).cast(pl.Float32) - float(mean[index])) / float(scale[index])).alias(column)
                for index, column in enumerate(columns)
            ]))
        annual, quarterly, daily = normalized
    sparse_stats = sparse.select(
        pl.col("signal_value").fill_nan(None).mean().alias("_sparse_mean"),
        pl.col("signal_value").fill_nan(None).std(ddof=0).alias("_sparse_std"),
    ).row(0)
    sparse_mean = float(sparse_stats[0]) if sparse_stats[0] is not None else 0.0
    sparse_scale = float(sparse_stats[1]) if sparse_stats[1] not in (None, 0.0) else 1.0
    sparse = sparse.with_columns(((pl.col("signal_value").cast(pl.Float32) - sparse_mean) / sparse_scale).alias("signal_value"))
    raw_sparse_columns = ["signal_value", *[f"text_{i}" for i in range(7)]]
    sparse_value_columns: list[str] = []
    # Construct the wide family-specific sparse matrix in Polars.  Assigning
    # hundreds of columns one at a time fragments a large tabular frame and
    # caused the 10B/two-year option run to consume tens of GB before training.
    sparse_wide = sparse.select(["target_family", *raw_sparse_columns])
    sparse_wide_columns = []
    for family in target_families:
        for column in raw_sparse_columns:
            output_column = f"{family}__{column}"
            sparse_wide_columns.append(
                pl.when(pl.col("target_family") == family)
                .then(pl.col(column))
                .otherwise(None)
                .alias(output_column)
            )
            sparse_value_columns.append(output_column)
    sparse = sparse.with_columns(sparse_wide.select(sparse_wide_columns))

    # Build immutable columnar indexes once. All subsequent sample windows use
    # these arrays instead of repeatedly filtering Pandas frames.
    if args.context_memmap_dir and (args.context_memmap_dir / "annual_index.json").exists():
        annual_index = _IndexedTable.from_memmap(args.context_memmap_dir, "annual")
        quarterly_index = _IndexedTable.from_memmap(args.context_memmap_dir, "quarterly")
        daily_index = _IndexedTable.from_memmap(args.context_memmap_dir, "daily")
        sparse_index = _IndexedTable.from_memmap(args.context_memmap_dir, "sparse")
    else:
        annual_index = _IndexedTable(annual, annual_value_columns)
        quarterly_index = _IndexedTable(quarterly, quarterly_value_columns)
        daily_index = _IndexedTable(daily, daily_value_columns)
        sparse_index = _IndexedTable(sparse, sparse_value_columns, target_column="target_id")
        if args.context_memmap_dir:
            for name, index in (("annual", annual_index), ("quarterly", quarterly_index), ("daily", daily_index), ("sparse", sparse_index)):
                index.save_memmap(args.context_memmap_dir, name)

    # A document can be anchored by a regular annual observation or by a
    # sparse event.  Using their union preserves early event history even
    # when annual fundamentals begin later for a symbol.
    anchor_parts = [annual.select(["symbol", "date"]), sparse.select(["symbol", "date"])]
    if option_document_symbols:
        anchor_parts.append(option_entry_anchors)
    if option_target_map:
        anchor_parts.append(pl.DataFrame({
            "symbol": [symbol for symbol, _ in option_target_map],
            "date": [date for _, date in option_target_map],
        }))
    anchors = pl.concat(anchor_parts, how="diagonal_relaxed").unique().sort(["symbol", "date"])
    # Exact-date inference must retain the requested daily anchor. Regular
    # training anchors are intentionally reduced to one date per symbol/year,
    # but that reduction would otherwise discard a current EOD scoring date.
    if args.inference_only and args.prediction_start_date and args.prediction_start_date != "1900-01-01":
        requested_anchor_date = _as_datetime(args.prediction_start_date)
        exact_daily_anchors = daily.filter(pl.col("date") == requested_anchor_date).select(["symbol", "date"])
        anchors = pl.concat([anchors, exact_daily_anchors], how="diagonal_relaxed").unique().sort(["symbol", "date"])
    anchors = anchors.with_columns(pl.col("date").dt.year().alias("year"))
    target_pairs = list(option_target_map)
    regular_candidates = anchors.filter(~pl.struct(["symbol", "date"]).is_in(target_pairs)) if target_pairs else anchors
    if option_document_symbols:
        option_daily_anchors = regular_candidates.filter(pl.col("symbol").is_in(list(option_document_symbols)))
        regular_candidates = regular_candidates.filter(~pl.col("symbol").is_in(list(option_document_symbols)))
    else:
        option_daily_anchors = anchors.head(0)
    regular_anchors = regular_candidates.group_by(["symbol", "year"]).agg(pl.col("date").max())
    if option_target_map:
        option_anchors = pl.DataFrame({"symbol": [symbol for symbol, _ in option_target_map], "date": [date for _, date in option_target_map]}).with_columns(pl.col("date").dt.year().alias("year"))
        anchors = pl.concat([regular_anchors, option_daily_anchors, option_anchors], how="diagonal_relaxed").unique(["symbol", "date"])
    else:
        anchors = pl.concat([regular_anchors, option_daily_anchors], how="diagonal_relaxed").unique(["symbol", "date"])
    if args.inference_only and args.prediction_start_date and args.prediction_start_date != "1900-01-01":
        requested_anchor_date = _as_datetime(args.prediction_start_date)
        exact_daily_anchors = daily.filter(pl.col("date") == requested_anchor_date).select(["symbol", "date"])
        anchors = pl.concat([anchors, exact_daily_anchors], how="diagonal_relaxed").unique(["symbol", "date"])
    empty_annual = annual.head(0)
    empty_quarterly = quarterly.head(0)
    empty_daily = daily.head(0)
    empty_sparse = sparse.head(0)
    context_caches: dict[str, OrderedDict[tuple[str, int], tuple]] = {
        rate: OrderedDict() for rate in ("annual", "quarterly", "daily", "sparse")
    }
    context_cache_hits = {rate: 0 for rate in context_caches}
    context_cache_misses = {rate: 0 for rate in context_caches}
    context_cache_build_seconds = {rate: 0.0 for rate in context_caches}

    def cached_window(rate: str, table, symbol: str, anchor: datetime, columns: list[str], length: int, source_symbol: str | None = None):
        source = str(source_symbol or symbol).upper()
        # Annual and quarterly windows only change when their source issuer
        # publishes a new row.  Keying them by the exact document date would
        # defeat reuse across daily documents and option instruments.
        version = rate_version(rate, table, source, anchor)
        key = (source, version)
        cache = context_caches[rate]
        if args.context_cache_size and key in cache:
            context_cache_hits[rate] += 1
            value = cache.pop(key)
            cache[key] = value
            return value
        context_cache_misses[rate] += 1
        started = perf_counter()
        value = _window(table, source, anchor, columns, length)
        context_cache_build_seconds[rate] += perf_counter() - started
        if args.context_cache_size:
            cache[key] = value
            while len(cache) > args.context_cache_size:
                cache.popitem(last=False)
        return value

    def rate_version(rate: str, table, source: str, anchor: datetime) -> int:
        if rate in {"annual", "quarterly"}:
            if isinstance(table, _IndexedTable):
                dates = table.rows.get(str(source).upper(), (torch.empty(0, dtype=torch.long), None, None))[0]
                stop = int(torch.searchsorted(dates, torch.tensor(_epoch_ns(anchor), dtype=torch.long), right=True))
                return int(dates[stop - 1]) if stop else -1
            source_rows = _symbol_rows(table, source)
            available = source_rows.filter(pl.col("date") <= anchor)["date"]
            return _epoch_ns(available[-1]) if len(available) else -1
        return _epoch_ns(anchor)

    def cached_sparse_window(symbol: str, anchor: datetime):
        rate = "sparse"
        key = (str(symbol).upper(), _epoch_ns(anchor))
        cache = context_caches[rate]
        if args.context_cache_size and key in cache:
            context_cache_hits[rate] += 1
            value = cache.pop(key)
            cache[key] = value
            return value
        context_cache_misses[rate] += 1
        started = perf_counter()
        value = _sparse_window(sparse_index, symbol, anchor, sparse_value_columns, 16)
        context_cache_build_seconds[rate] += perf_counter() - started
        if args.context_cache_size:
            cache[key] = value
            while len(cache) > args.context_cache_size:
                cache.popitem(last=False)
        return value

    samples: list[dict[str, object]] = []
    taxonomy_by_symbol = {str(row["symbol"]): row for row in taxonomy.iter_rows(named=True)}
    taxonomy_symbols = set(taxonomy_by_symbol)
    for row in anchors.iter_rows(named=True):
        symbol = str(row["symbol"]).upper(); anchor = _as_datetime(row["date"])
        if symbol not in taxonomy_symbols:
            continue
        source_symbol = source_symbol_by_symbol.get(symbol, symbol)
        def materialize(current_symbol=symbol, current_anchor=anchor, current_source=source_symbol):
            annual_values, annual_padding, _ = cached_window("annual", annual_index, current_symbol, current_anchor, annual_value_columns, ANNUAL_WINDOW, current_source)
            quarterly_values, quarterly_padding, _ = cached_window("quarterly", quarterly_index, current_symbol, current_anchor, quarterly_value_columns, QUARTERLY_WINDOW, current_source)
            daily_values, daily_padding, daily_dates = cached_window("daily", daily_index, current_symbol, current_anchor, daily_value_columns, DAILY_WINDOW, current_source)
            sparse_values, sparse_padding, sparse_labels, sparse_dates = cached_sparse_window(current_symbol, current_anchor)
            supervised_targets = torch.zeros((DAILY_WINDOW, len(SUPERVISED_TARGET_TASK_NAMES)), dtype=torch.float32)
            supervised_valid = torch.zeros((DAILY_WINDOW, len(SUPERVISED_TARGET_TASK_NAMES)), dtype=torch.bool)
            if len(daily_dates):
                offset = DAILY_WINDOW - len(daily_dates)
                for position, date in enumerate(daily_dates):
                    values = supervised_target_map.get((current_symbol, _as_datetime(date)), {})
                    for task_index, task_name in enumerate(SUPERVISED_TARGET_TASK_NAMES):
                        if task_name in values:
                            supervised_targets[offset + position, task_index] = values[task_name]
                            supervised_valid[offset + position, task_index] = True
            return {
                "annual": annual_values, "annual_padding": annual_padding,
                "quarterly": quarterly_values, "quarterly_padding": quarterly_padding,
                "daily": daily_values, "daily_padding": daily_padding,
                "daily_dates": [_as_datetime(date).strftime("%Y-%m-%d") for date in daily_dates],
                "sparse": sparse_values, "sparse_padding": sparse_padding, "sparse_labels": sparse_labels,
                "supervised_targets": supervised_targets, "supervised_valid": supervised_valid,
            }
        metadata = {
            "symbol": symbol, "date": anchor.strftime("%Y-%m-%d"),
            "issuer": str(taxonomy_by_symbol[symbol]["issuer"]),
            "annual_context_key": (source_symbol, rate_version("annual", annual_index, source_symbol, anchor)),
            "quarterly_context_key": (source_symbol, rate_version("quarterly", quarterly_index, source_symbol, anchor)),
            "sector": str(taxonomy_by_symbol[symbol]["sector"]), "subsector": str(taxonomy_by_symbol[symbol]["subsector"]),
            "industry": str(taxonomy_by_symbol[symbol]["industry"]),
        }
        samples.append(_LazySample(metadata, materialize) if args.stream_samples else {**metadata, **materialize()})
    # The indexes own the compact sorted arrays used by lazy samples. Release
    # the source Polars frames before model construction/training; retaining
    # both representations is the main avoidable memory spike on 100B runs.
    del annual, quarterly, daily, sparse
    if args.max_samples:
        if args.max_samples < 1:
            parser.error("--max-samples must be positive when provided")
        samples = sorted(samples, key=lambda item: (_as_datetime(item["date"]), str(item["symbol"])))[:args.max_samples]
    frame = pl.DataFrame([{key: value for key, value in sample.items() if isinstance(value, (str, int))} for sample in samples])
    label_arrays: dict[str, torch.Tensor] = {}
    label_names: dict[str, list[str]] = {}
    for task in DOCUMENT_TASK_NAMES[1:]:
        label_arrays[task], label_names[task], _ = _encode_labels(frame[task])
    if checkpoint_payload is not None and isinstance(checkpoint_payload.get("labels"), dict):
        checkpoint_labels = checkpoint_payload["labels"]
        for task in DOCUMENT_TASK_NAMES[1:]:
            if task in checkpoint_labels:
                label_names[task] = list(checkpoint_labels[task])
    if option_columns:
        feature_families = [*feature_families, "options"]
    feature_family_dimensions = {family: 1 for family in feature_families if family != "options"}
    if option_columns:
        feature_family_dimensions["options"] = len(option_columns)
    family_names = [*feature_families, *target_families]
    label_names["family"] = (
        list(checkpoint_payload["labels"]["family"])
        if checkpoint_payload is not None and isinstance(checkpoint_payload.get("labels"), dict)
        and "family" in checkpoint_payload["labels"]
        else family_names
    )
    for index, sample in enumerate(samples):
        for name in DOCUMENT_TASK_NAMES[1:]:
            sample[f"{name}_label"] = int(label_arrays[name][index])

    def load_symbol_file(path: Path | None) -> set[str] | None:
        if path is None:
            return None
        frame = pl.read_csv(path)
        column = "symbol" if "symbol" in frame.columns else frame.columns[0]
        return set(frame[column].cast(pl.String).str.to_uppercase().str.strip_chars().to_list())

    train_symbols = load_symbol_file(args.train_symbols_file)
    test_symbols = load_symbol_file(args.test_symbols_file)
    if train_symbols is not None and test_symbols is not None and train_symbols & test_symbols:
        raise ValueError("train and test symbol files overlap")
    evaluation_samples = samples
    if test_symbols is not None:
        evaluation_samples = [sample for sample in samples if str(sample["symbol"]).upper() in test_symbols]
        if not evaluation_samples:
            raise ValueError("test symbol file does not match any corpus samples")
    if args.inference_only:
        if args.prediction_start_date and args.prediction_start_date != "1900-01-01":
            requested_date = _as_datetime(args.prediction_start_date)
            evaluation_samples = [sample for sample in evaluation_samples if _as_datetime(sample["date"]) == requested_date]
            if not evaluation_samples:
                raise ValueError(f"No feature samples exist for requested inference date {requested_date.date()}")
            symbols_on_date = {str(sample["symbol"]).upper() for sample in evaluation_samples}
            print(f"[multirate-inference] exact-date evaluation set: {len(symbols_on_date)} symbols on {requested_date.date()}", flush=True)
        else:
            latest_by_symbol: dict[str, dict[str, object]] = {}
            for sample in evaluation_samples:
                symbol = str(sample["symbol"]).upper()
                if symbol not in latest_by_symbol or _as_datetime(sample["date"]) > _as_datetime(latest_by_symbol[symbol]["date"]):
                    latest_by_symbol[symbol] = sample
            evaluation_samples = sorted(latest_by_symbol.values(), key=lambda item: (_as_datetime(item["date"]), str(item["symbol"])))
            print(f"[multirate-inference] latest-per-symbol evaluation set: {len(evaluation_samples)} symbols", flush=True)
    train_samples = samples if train_symbols is None else [
        sample for sample in samples if str(sample["symbol"]).upper() in train_symbols
    ]
    if train_symbols is not None and not train_samples:
        raise ValueError("train symbol file does not match any corpus samples")
    if args.train_end_date:
        train_end = _as_datetime(args.train_end_date)
        train_samples = [sample for sample in samples if _as_datetime(sample["date"]) < train_end]
        if not train_samples:
            raise ValueError(f"no training samples exist before {args.train_end_date}")

    # Keep the latest 20% of training dates as a chronological validation
    # holdout so validation samples never contribute gradients.
    training_dates = sorted({_as_datetime(sample["date"]) for sample in train_samples})
    validation_samples: list[dict[str, object]] = []
    if not 0.0 <= args.validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")
    if args.validation_fraction > 0.0 and len(training_dates) >= 2:
        validation_start = training_dates[max(1, int(math.ceil(len(training_dates) * (1.0 - args.validation_fraction)))) - 1]
        validation_samples = [sample for sample in train_samples if _as_datetime(sample["date"]) >= validation_start]
        train_samples = [sample for sample in train_samples if _as_datetime(sample["date"]) < validation_start]
    if not train_samples:
        raise ValueError("chronological validation split removed all training samples")

    if args.sample_build_only:
        cache_metrics = {
            "size_per_rate": args.context_cache_size,
            "hits": context_cache_hits,
            "misses": context_cache_misses,
            "build_seconds": context_cache_build_seconds,
            "hit_rate": {
                rate: context_cache_hits[rate] / max(1, context_cache_hits[rate] + context_cache_misses[rate])
                for rate in context_cache_hits
            },
        }
        (output_dir / "sample_build_summary.json").write_text(json.dumps({
            "samples": len(samples), "training_samples": len(train_samples), "context_cache": cache_metrics,
        }, indent=2))
        print(json.dumps({"samples": len(samples), "context_cache": cache_metrics}, indent=2), flush=True)
        return

    # Inference must recreate the checkpoint architecture.  Live notebooks
    # commonly use CLI defaults, while smoke/production checkpoints may have
    # different width or depth.  Prefer explicit metadata and fall back to
    # the state-dict shapes for older checkpoints.
    if checkpoint_payload is not None:
        checkpoint_metrics = checkpoint_payload.get("metrics", {}) if isinstance(checkpoint_payload, dict) else {}
        state_dict = checkpoint_payload.get("state_dict", {}) if isinstance(checkpoint_payload, dict) else {}
        checkpoint_width = checkpoint_metrics.get("d_model")
        if checkpoint_width is None and isinstance(state_dict, dict):
            weight = state_dict.get("task_heads.oracle_is_buy.weight")
            if weight is not None and getattr(weight, "ndim", 0) == 2:
                checkpoint_width = int(weight.shape[1])
        if checkpoint_width:
            args.d_model = int(checkpoint_width)
        layer_keys = [key for key in state_dict if key.startswith("encoders.annual.layers.")] if isinstance(state_dict, dict) else []
        if layer_keys:
            args.layers = max(int(key.split(".layers.", 1)[1].split(".", 1)[0]) for key in layer_keys) + 1
        if args.d_model % args.num_heads:
            args.num_heads = max(divisor for divisor in range(1, args.num_heads + 1) if args.d_model % divisor == 0)
        if checkpoint_metrics.get("document_tasks_disabled"):
            enabled_document_tasks = ()

    device = torch.device(args.device)
    config = MultiRateTransformerConfig(
        backbone="encoder_decoder", d_model=args.d_model, num_heads=args.num_heads,
        layers=args.layers, document_pool="mean", max_position=512,
        learned_aggregation_gate=args.learned_aggregation_gate,
        cacheable_rate_states=not args.legacy_rate_fusion,
        attention_backend=args.attention_backend,
    )
    task_bundle = add_subtoken_temporal_tasks(
        train_samples,
        family_names,
        label_names,
        batch_size=args.batch_size,
        # This is part of the Multi-Rate training contract.  Issuer/date
        # contexts must remain together so annual and quarterly encoder states
        # can be reused inside every training batch.
        batch_key=lambda item: item["annual_context_key"],
    )
    model_tasks = tuple(
        task for task in task_bundle.document_tasks + task_bundle.supervised_tasks
        if task.task_name in enabled_document_tasks or task.task_name not in DOCUMENT_TASK_NAMES
    )
    active_tasks = tuple(task for task in task_bundle.tasks if task.name in {spec.task_name for spec in model_tasks} or task.name in {spec.task_name for spec in task_bundle.prediction_tasks})
    if mrl_dimensions:
        active_tasks = (*active_tasks, Task("mrl", spec="matryoshka_document_alignment", loss_weight=args.mrl_weight))
    expected_task_names = tuple(enabled_document_tasks) + SUPERVISED_TARGET_TASK_NAMES + PREDICTION_TASK_NAMES
    model = MultiRateTransformer(
        {"annual": len(annual_value_columns), "quarterly": len(quarterly_value_columns), "daily": len(daily_value_columns), "sparse": len(sparse_value_columns)},
        config=config,
        feature_families={
            "annual": feature_family_dimensions,
            "quarterly": feature_family_dimensions,
            "daily": feature_family_dimensions,
            "sparse": {family: len(raw_sparse_columns) for family in target_families},
        },
        tasks=model_tasks,
        prediction_tasks=task_bundle.prediction_tasks,
    ).to(device)
    if args.checkpoint is not None:
        checkpoint = checkpoint_payload or torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint)
    if args.compile_model:
        if not hasattr(torch, "compile"):
            parser.error("--compile-model requires torch.compile support")
        model = torch.compile(model)
    if args.optimizer == "adamw8bit":
        try:
            import bitsandbytes as bnb
        except ImportError as exc:
            parser.error("--optimizer adamw8bit requires bitsandbytes")
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=2e-4, weight_decay=1e-4)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    trainer = Trainer(
        model,
        [(task_bundle.corpus, active_tasks)],
        optimizer,
        grad_accumulation_steps=args.grad_accumulation_steps,
        autocast_dtype=torch.float16 if args.mixed_precision else None,
        transformer_engine_fp8=args.fp8,
    )
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    validation_losses: list[float] = []
    if args.progress_every_batches < 0:
        parser.error("--progress-every-batches must be non-negative")
    if args.checkpoint_every_batches < 0:
        parser.error("--checkpoint-every-batches must be non-negative")
    def training_step(module: torch.nn.Module, batch: list[dict[str, object]], active_tasks):
        def stack(name: str) -> torch.Tensor:
            return torch.stack([item[name] if isinstance(item[name], torch.Tensor) else torch.as_tensor(item[name]) for item in batch]).to(device)

        def context(name: str, padding_name: str) -> tuple[torch.Tensor, torch.Tensor]:
            values = stack(name); padding = stack(padding_name).bool()
            empty = padding.all(dim=1)
            if empty.any():
                values[empty, -1] = 0.0; padding[empty, -1] = False
            leading = padding[:, 0]
            if leading.any():
                values[leading, 0] = 0.0; padding[leading, 0] = False
            return values, padding

        annual_batch, annual_mask = context("annual", "annual_padding")
        quarterly_batch, quarterly_mask = context("quarterly", "quarterly_padding")
        sparse_batch, sparse_padding_mask = context("sparse", "sparse_padding")
        masked_positions: dict[str, torch.Tensor] = {}
        masked_batches = {"annual": (annual_batch, annual_mask), "quarterly": (quarterly_batch, quarterly_mask), "daily": (None, None), "sparse": (sparse_batch, sparse_padding_mask)}
        mask_rng = torch.Generator(device=device).manual_seed(trainer.current_epoch * 100000 + trainer.current_step)
        for rate in ("annual", "quarterly", "daily", "sparse"):
            if rate == "daily":
                batch_values, padding = context("daily", "daily_padding")
            else:
                batch_values, padding = masked_batches[rate]
            raw = stack(rate)
            if rate == "sparse":
                raw_shape = raw.shape
                target_valid = torch.isfinite(raw).reshape(*raw_shape[:2], len(target_families), len(raw_sparse_columns)).any(dim=-1)
                selected = target_valid & ~padding.unsqueeze(-1)
                selected &= torch.rand(target_valid.shape, device=device, generator=mask_rng).lt(0.15)
                expanded = selected.unsqueeze(-1).expand(*raw_shape[:2], len(target_families), len(raw_sparse_columns)).reshape(raw_shape)
                batch_values = batch_values.clone(); batch_values[expanded] = float("nan")
            else:
                target_valid = torch.isfinite(raw)
                family_valid = []
                offset = 0
                for family in feature_families:
                    width = feature_family_dimensions[family]
                    family_valid.append(target_valid[:, :, offset:offset + width].any(dim=-1))
                    offset += width
                family_valid = torch.stack(family_valid, dim=-1)
                selected = family_valid & ~padding.unsqueeze(-1)
                selected &= torch.rand(selected.shape, device=device, generator=mask_rng).lt(0.15)
                expanded = []
                offset = 0
                for family in feature_families:
                    width = feature_family_dimensions[family]
                    expanded.append(selected[:, :, feature_families.index(family)].unsqueeze(-1).expand(-1, -1, width))
                    offset += width
                expanded = torch.cat(expanded, dim=-1)
                batch_values = batch_values.clone(); batch_values[expanded] = float("nan")
            masked_batches[rate] = (batch_values, padding); masked_positions[rate] = selected
        daily_batch, daily_mask = masked_batches["daily"]
        annual_batch, annual_mask = masked_batches["annual"]
        quarterly_batch, quarterly_mask = masked_batches["quarterly"]
        sparse_batch, sparse_padding_mask = masked_batches["sparse"]
        # IDs are deliberately rate-specific. Annual/quarterly issuer streams
        # can be shared by instruments with the same issuer/as-of date; sparse
        # streams retain symbol identity because option-native events differ.
        annual_ids = {key: index for index, key in enumerate(sorted({item["annual_context_key"] for item in batch}))}
        quarterly_ids = {key: index for index, key in enumerate(sorted({item["quarterly_context_key"] for item in batch}))}
        sparse_ids = {key: index for index, key in enumerate(sorted({(str(item["symbol"]), str(item["date"])) for item in batch}))}
        annual_context_ids = torch.tensor([annual_ids[item["annual_context_key"]] for item in batch], device=device)
        quarterly_context_ids = torch.tensor([quarterly_ids[item["quarterly_context_key"]] for item in batch], device=device)
        sparse_context_ids = torch.tensor([sparse_ids[(str(item["symbol"]), str(item["date"]))] for item in batch], device=device)
        rate_context_ids = {
            "annual": annual_context_ids if torch.unique(annual_context_ids).numel() < len(batch) else None,
            "quarterly": quarterly_context_ids if torch.unique(quarterly_context_ids).numel() < len(batch) else None,
            "sparse": sparse_context_ids if torch.unique(sparse_context_ids).numel() < len(batch) else None,
        }
        output = module(
            daily_batch, annual_batch, quarterly_batch, sparse_batch,
            daily_padding_mask=daily_mask, annual_padding_mask=annual_mask,
            quarterly_padding_mask=quarterly_mask, sparse_padding_mask=sparse_padding_mask,
            daily_dates=torch.arange(DAILY_WINDOW, device=device), annual_dates=torch.arange(ANNUAL_WINDOW, device=device),
            quarterly_dates=torch.arange(QUARTERLY_WINDOW, device=device), sparse_dates=torch.arange(16, device=device),
            rate_context_ids=rate_context_ids,
            compute_document_outputs=not args.disable_document_tasks,
        )
        if tuple(output["document_outputs"]) + tuple(output["token_outputs"]) + tuple(output["prediction_outputs"]) != expected_task_names:
            raise RuntimeError("model task outputs do not match the temporal token+subtoken MTL contract")
        zero_source = next(iter(output["token_outputs"].values()), None)
        if zero_source is None:
            zero_source = next(iter(output["prediction_outputs"].values()))
        zero = zero_source.sum() * 0.0
        active_names = {task.name for task in active_tasks}
        task_losses = {task.name: zero for task in active_tasks}
        if "mrl" in active_names:
            task_losses["mrl"] = matryoshka_alignment_loss(output["document_state"], mrl_dimensions)
        for name in DOCUMENT_TASK_NAMES[1:]:
            if name not in enabled_document_tasks:
                continue
            if name in active_names:
                target = torch.tensor([item[f"{name}_label"] for item in batch], device=device)
                task_losses[name] = nn.functional.cross_entropy(output["document_outputs"][name], target)
        supervised_targets = stack("supervised_targets")
        supervised_valid = stack("supervised_valid").bool()
        for task_index, name in enumerate(SUPERVISED_TARGET_TASK_NAMES):
            if name not in active_names:
                continue
            valid = supervised_valid[:, :, task_index] & ~daily_mask
            if not valid.any():
                continue
            target = supervised_targets[:, :, task_index]
            prediction = output["token_outputs"][name].squeeze(-1)
            if name in ORACLE_SUPERVISED_TASK_NAMES or name in FUND_ACTIVITY_SUPERVISED_TASK_NAMES or name in HOLDER_ACTIVITY_SUPERVISED_TASK_NAMES:
                task_losses[name] = nn.functional.binary_cross_entropy_with_logits(prediction[valid], target[valid])
            else:
                task_losses[name] = nn.functional.smooth_l1_loss(prediction[valid], target[valid])
        family_labels = torch.arange(len(family_names), device=device).view(1, -1).expand(len(batch), -1)
        family_valid = torch.zeros((len(batch), len(family_names)), dtype=torch.bool, device=device)
        for rate in ("annual", "quarterly", "daily", "sparse"):
            raw = stack(rate)
            if rate == "sparse":
                local_count, width, family_offset = len(target_families), len(raw_sparse_columns), len(feature_families)
                observed = torch.isfinite(raw).reshape(raw.shape[0], raw.shape[1], local_count, width).any(dim=-1).any(dim=1)
            else:
                family_offset = 0
                offset = 0
                family_observed = []
                for family in feature_families:
                    width = feature_family_dimensions[family]
                    family_observed.append(torch.isfinite(raw[:, :, offset:offset + width]).any(dim=-1).any(dim=1))
                    offset += width
                observed = torch.stack(family_observed, dim=1)
                local_count = len(feature_families)
            family_valid[:, family_offset:family_offset + local_count] |= observed
        if "family" in active_names and family_valid.any():
            task_losses["family"] = nn.functional.cross_entropy(output["document_outputs"]["family"][family_valid], family_labels[family_valid])
        for rate in ("annual", "quarterly", "daily", "sparse"):
            raw_target = stack(rate)
            if rate == "sparse":
                raw_shape = raw_target.shape
                family_raw = raw_target.reshape(*raw_shape[:2], len(target_families), len(raw_sparse_columns))
                family_valid = torch.isfinite(family_raw)
                target_valid = family_valid.any(dim=-1)
                target = (
                    (torch.nan_to_num(family_raw) * family_valid).sum(dim=-1)
                    / family_valid.sum(dim=-1).clamp_min(1)
                )
            else:
                raw_valid = torch.isfinite(raw_target)
                family_values = []
                family_valid_parts = []
                offset = 0
                for family in feature_families:
                    width = feature_family_dimensions[family]
                    local_valid = raw_valid[:, :, offset:offset + width]
                    family_valid_parts.append(local_valid.any(dim=-1))
                    family_values.append(
                        (torch.nan_to_num(raw_target[:, :, offset:offset + width]) * local_valid).sum(dim=-1)
                        / local_valid.sum(dim=-1).clamp_min(1)
                    )
                    offset += width
                target_valid = torch.stack(family_valid_parts, dim=-1)
                target = torch.stack(family_values, dim=-1)
            token_valid = target_valid.any(dim=-1)
            token_target = (
                (torch.nan_to_num(target) * target_valid).sum(dim=-1)
                / target_valid.sum(dim=-1).clamp_min(1)
            )
            padding = stack(f"{rate}_padding").bool()
            valid_next = ~padding
            valid_next[:, :-1] = valid_next[:, :-1] & valid_next[:, 1:]; valid_next[:, -1] = False
            valid_next = valid_next.unsqueeze(-1).expand_as(target).clone(); valid_next &= target_valid
            valid_next[:, :-1] = valid_next[:, :-1] & target_valid[:, 1:]
            valid_next_token = (~padding).clone()
            valid_next_token[:, :-1] &= token_valid[:, :-1] & token_valid[:, 1:]
            valid_next_token[:, -1] = False
            next_subtoken_name = f"next_{rate}_subtoken"
            next_token_name = f"next_{rate}_token"
            next_subtoken_prediction = output["prediction_outputs"][next_subtoken_name].squeeze(-1)
            if next_subtoken_name in active_names and valid_next.any():
                task_losses[next_subtoken_name] = nn.functional.mse_loss(next_subtoken_prediction[valid_next], target.roll(-1, dims=1)[valid_next])
            next_token_prediction = output["prediction_outputs"][next_token_name].squeeze(-1)
            if next_token_name in active_names and valid_next_token.any():
                task_losses[next_token_name] = nn.functional.mse_loss(next_token_prediction[valid_next_token], token_target.roll(-1, dims=1)[valid_next_token])
            masked_valid = masked_positions[rate]
            masked_subtoken_name = f"masked_{rate}_subtoken"
            masked_token_name = f"masked_{rate}_token"
            masked_subtoken_prediction = output["prediction_outputs"][masked_subtoken_name].squeeze(-1)
            if masked_subtoken_name in active_names and masked_valid.any():
                task_losses[masked_subtoken_name] = nn.functional.mse_loss(masked_subtoken_prediction[masked_valid], target[masked_valid])
            masked_token_valid = masked_valid.any(dim=-1) & token_valid
            masked_token_prediction = output["prediction_outputs"][masked_token_name].squeeze(-1)
            if masked_token_name in active_names and masked_token_valid.any():
                task_losses[masked_token_name] = nn.functional.mse_loss(masked_token_prediction[masked_token_valid], token_target[masked_token_valid])
        return task_losses

    validation_corpus = Corpus(
        validation_samples,
        name="validation",
        batch_size=args.batch_size,
        batch_key=lambda item: item["annual_context_key"],
    )

    def validation_loss(epoch: int) -> float:
        if not validation_samples:
            return float("nan")
        model.eval()
        total = 0.0
        count = 0
        trainer.current_epoch = epoch
        with torch.inference_mode():
            for step_index, batch in enumerate(validation_corpus.batches(seed=trainer.seed, epoch=epoch)):
                trainer.current_step = step_index
                task_losses = training_step(model, batch, active_tasks)
                total += sum(float(task.loss_weight * task_losses[task.name]) for task in active_tasks)
                count += 1
        return total / max(1, count)

    def epoch_end(epoch: int, epoch_loss: float) -> bool:
        nonlocal best_loss, best_state, stale_epochs
        val_loss = validation_loss(epoch)
        validation_losses.append(val_loss)
        stopping_loss = val_loss if validation_samples else epoch_loss
        if stopping_loss < best_loss - args.min_delta:
            best_loss = stopping_loss; best_state = copy.deepcopy(model.state_dict()); stale_epochs = 0
        else:
            stale_epochs += 1
        print(f"epoch {epoch + 1}/{args.epochs} loss={epoch_loss:.6f} validation_loss={val_loss:.6f}", flush=True)
        if stale_epochs >= max(1, args.patience):
            print(f"early stopping after epoch {epoch + 1}; best_loss={best_loss:.6f}", flush=True)
            return True
        return False

    training_started = perf_counter()

    def save_batch_checkpoint(epoch: int, batch_index: int, batch_loss: float) -> None:
        checkpoint_path = output_dir / "multirate_mtl_checkpoint_latest.pt"
        temporary_path = output_dir / "multirate_mtl_checkpoint_latest.pt.tmp"
        payload = {
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": {
                "epoch": epoch,
                "batch": batch_index,
                "batch_loss": batch_loss,
                "feature_families": feature_families,
                "target_families": target_families,
                "tasks": list(expected_task_names),
                "train_samples": len(train_samples),
            },
        }
        torch.save(payload, temporary_path)
        os.replace(temporary_path, checkpoint_path)
        print(
            f"[multirate-checkpoint] epoch={epoch + 1} batch={batch_index} "
            f"path={checkpoint_path}",
            flush=True,
        )

    def training_progress(epoch: int, batch_index: int, total_batches: int, batch_loss: float) -> None:
        interval = args.progress_every_batches
        if interval <= 0 or (batch_index % interval and batch_index != total_batches):
            if args.checkpoint_every_batches <= 0 or (batch_index % args.checkpoint_every_batches and batch_index != total_batches):
                return
        elapsed = perf_counter() - training_started
        rate = batch_index / max(elapsed, 1e-6)
        remaining = (total_batches - batch_index) / max(rate, 1e-6)
        samples_done = min(len(train_samples), batch_index * args.batch_size)
        memory = ""
        if device.type == "cuda":
            memory = f" cuda_gb={torch.cuda.memory_allocated(device) / 1024**3:.2f}"
        if interval > 0 and (batch_index % interval == 0 or batch_index == total_batches):
            print(
                f"[multirate-train] epoch={epoch + 1}/{args.epochs} "
                f"batch={batch_index}/{total_batches} samples={samples_done}/{len(train_samples)} "
                f"loss={batch_loss:.6f} elapsed_s={elapsed:.1f} "
                f"batches_per_s={rate:.2f} eta_s={remaining:.1f}{memory}",
                flush=True,
            )
        if args.checkpoint_every_batches > 0 and (
            batch_index % args.checkpoint_every_batches == 0 or batch_index == total_batches
        ):
            save_batch_checkpoint(epoch, batch_index, batch_loss)

    losses = [] if args.inference_only else trainer.fit(
        args.epochs,
        training_step,
        on_epoch_end=epoch_end,
        on_batch_end=training_progress,
    )

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval(); predictions: dict[str, list[torch.Tensor]] = {name: [] for name in enabled_document_tasks[1:]}; states: list[torch.Tensor] = []; family_states: list[torch.Tensor] = []; family_valid_rows: list[torch.Tensor] = []
    prediction_rows: list[dict[str, object]] = []
    prediction_start = _as_datetime(args.prediction_start_date) if args.prediction_start_date else None
    family_correct = 0
    family_total = 0
    with torch.inference_mode():
        # Evaluation materializes all prototype states and is more memory-sensitive
        # than the training step. Keep the scalable training batch size, but use a
        # bounded evaluation batch to avoid accelerator kernel failures on large
        # corpora.
        eval_batch_size = min(args.batch_size, 64)
        for start in range(0, len(evaluation_samples), eval_batch_size):
            batch = evaluation_samples[start:start + eval_batch_size]
            if args.inference_only:
                batch_symbols = ",".join(dict.fromkeys(str(item.get("symbol", "?")) for item in batch))
                print(
                    f"[multirate-inference] scoring symbols {start + 1}-{start + len(batch)}/{len(evaluation_samples)}: {batch_symbols}",
                    flush=True,
                )
            def stack(name: str) -> torch.Tensor: return torch.stack([item[name] if isinstance(item[name], torch.Tensor) else torch.as_tensor(item[name]) for item in batch]).to(device)
            def context(name: str, padding_name: str) -> tuple[torch.Tensor, torch.Tensor]:
                values = stack(name); padding = stack(padding_name).bool()
                empty = padding.all(dim=1)
                if empty.any():
                    values[empty, -1] = 0.0; padding[empty, -1] = False
                leading = padding[:, 0]
                if leading.any():
                    values[leading, 0] = 0.0; padding[leading, 0] = False
                return values, padding
            daily_batch, daily_mask = context("daily", "daily_padding")
            annual_batch, annual_mask = context("annual", "annual_padding")
            quarterly_batch, quarterly_mask = context("quarterly", "quarterly_padding")
            sparse_batch, sparse_padding_mask = context("sparse", "sparse_padding")
            output = model(daily_batch, annual_batch, quarterly_batch, sparse_batch, daily_padding_mask=daily_mask, annual_padding_mask=annual_mask, quarterly_padding_mask=quarterly_mask, sparse_padding_mask=sparse_padding_mask, daily_dates=torch.arange(DAILY_WINDOW, device=device), annual_dates=torch.arange(ANNUAL_WINDOW, device=device), quarterly_dates=torch.arange(QUARTERLY_WINDOW, device=device), sparse_dates=torch.arange(16, device=device))
            if prediction_start is not None:
                score_names = tuple(SUPERVISED_TARGET_TASK_NAMES)
                score_arrays = {
                    name: torch.sigmoid(output["token_outputs"][name].squeeze(-1)).cpu()
                    for name in score_names
                }
                for row_index, item in enumerate(batch):
                    dates = [_as_datetime(value) for value in item["daily_dates"]]
                    offset = DAILY_WINDOW - len(dates)
                    for date_index, date in enumerate(dates):
                        if date < prediction_start:
                            continue
                        score_row = {name: float(values[row_index, offset + date_index]) for name, values in score_arrays.items()}
                        prediction_rows.append({"symbol": item["symbol"], "date": date.strftime("%Y-%m-%d"), **score_row})
            if not args.skip_embeddings:
                states.append(output["document_prototypes"].cpu())
            family_labels = torch.arange(len(family_names), device=device).view(1, -1).expand(len(batch), -1)
            family_valid = torch.zeros((len(batch), len(family_names)), dtype=torch.bool, device=device)
            for rate in ("annual", "quarterly", "daily", "sparse"):
                raw = stack(rate)
                if rate == "sparse":
                    local_count, width, family_offset = len(target_families), len(raw_sparse_columns), len(feature_families)
                    observed = torch.isfinite(raw).reshape(raw.shape[0], raw.shape[1], local_count, width).any(dim=-1).any(dim=1)
                else:
                    family_offset = 0
                    offset = 0
                    family_observed = []
                    for family in feature_families:
                        width = feature_family_dimensions[family]
                        family_observed.append(torch.isfinite(raw[:, :, offset:offset + width]).any(dim=-1).any(dim=1))
                        offset += width
                    observed = torch.stack(family_observed, dim=1)
                    local_count = len(feature_families)
                family_valid[:, family_offset:family_offset + local_count] |= observed
            if not args.skip_embeddings:
                family_states.append(output["family_document_prototypes"].cpu())
                family_valid_rows.append(family_valid.cpu())
            if "family" in enabled_document_tasks:
                family_predictions = output["document_outputs"]["family"].argmax(dim=-1)
                family_correct += int((family_predictions[family_valid] == family_labels[family_valid]).sum())
                family_total += int(family_valid.sum())
            for name in predictions: predictions[name].append(output["document_outputs"][name].argmax(dim=-1).cpu())
    evaluation_label_arrays = {
        name: torch.tensor([sample[f"{name}_label"] for sample in evaluation_samples], dtype=torch.long)
        for name in predictions
    }
    task_accuracy = {
        name: float(torch.cat(predictions[name]).eq(evaluation_label_arrays[name]).float().mean())
        for name in predictions
    }
    if "family" in enabled_document_tasks:
        task_accuracy["family"] = family_correct / max(1, family_total)
    metrics = {"device": str(device), "samples": len(samples), "training_samples": len(train_samples), "feature_families": feature_families, "target_families": target_families, "family_labels": family_names, "tasks": [*expected_task_names, *(["mrl"] if mrl_dimensions else [])], "losses": losses, "best_loss": best_loss, "epochs_completed": len(losses), "patience": args.patience, "min_delta": args.min_delta, "task_accuracy": task_accuracy, "rates": ["annual", "quarterly", "daily", "sparse"], "backbone": config.backbone, "document_tasks_disabled": args.disable_document_tasks, "learned_aggregation_gate": args.learned_aggregation_gate, "mrl": bool(mrl_dimensions), "mrl_dimensions": list(mrl_dimensions), "mrl_weight": args.mrl_weight, "train_end_date": args.train_end_date, "prediction_start_date": args.prediction_start_date}
    metrics.update({
        "validation_samples": len(validation_samples),
        "validation_losses": validation_losses,
        "best_validation_loss": best_loss,
        "early_stopping_metric": "validation_loss" if validation_samples else "training_loss",
        "validation_fraction": args.validation_fraction,
    })
    metrics.update({
        "evaluation_samples": len(evaluation_samples),
        "train_symbols": sorted(train_symbols) if train_symbols is not None else None,
        "test_symbols": sorted(test_symbols) if test_symbols is not None else None,
        "cacheable_rate_states": config.cacheable_rate_states,
        "group_context_batches": True,
        "mixed_precision": args.mixed_precision,
        "fp8": args.fp8,
        "compile_model": args.compile_model,
        "optimizer": args.optimizer,
        "skip_embeddings": args.skip_embeddings,
        "skip_t_sne": args.skip_t_sne,
        "skip_predictions": args.skip_predictions,
        "stream_samples": args.stream_samples,
        "option_panel": str(args.option_panel) if args.option_panel else None,
        "option_max_contracts": args.option_max_contracts if args.option_panel else None,
        "option_start_date": args.option_start_date if args.option_panel else None,
        "option_end_date": args.option_end_date if args.option_panel else None,
        "option_dte": sorted(option_dtes) if args.option_panel else None,
        "option_issuer_dte_bins": args.option_issuer_dte_bins if args.option_panel else None,
        "option_features": list(OPTION_FEATURES) if option_columns else [],
        "universe_filter": {"country": args.country, "currency": args.currency, "exchanges": sorted(exchanges), "symbols": sorted(universe_symbols), "currency_unresolved_symbols": sorted(unresolved_currency)},
    })
    metrics["context_cache"] = {
        "size_per_rate": args.context_cache_size,
        "hits": context_cache_hits,
        "misses": context_cache_misses,
        "build_seconds": context_cache_build_seconds,
        "hit_rate": {
            rate: context_cache_hits[rate] / max(1, context_cache_hits[rate] + context_cache_misses[rate])
            for rate in context_cache_hits
        },
    }
    (output_dir / "training_summary.json").write_text(json.dumps(metrics, indent=2))
    torch.save({"state_dict": model.state_dict(), "metrics": metrics, "labels": label_names}, output_dir / "multirate_mtl_model.pt")
    if prediction_rows:
        pl.DataFrame(prediction_rows).sort(["date", "symbol"]).write_csv(output_dir / "supervised_predictions.csv")
    if args.learned_aggregation_gate:
        gate = model.auto_feature_engineer.aggregation_gate
        if gate.family_logits is not None:
            weights = torch.softmax(gate.family_logits.detach(), dim=-1).cpu().tolist()
            gate_rows = [
                {"feature_family": family, "aggregation": aggregation, "weight": float(weights[index, aggregation_index])}
                for index, family in enumerate(model.family_names)
                for aggregation_index, aggregation in enumerate(gate.aggregation_functions)
            ]
            pl.DataFrame(gate_rows).write_csv(output_dir / "aggregation_gate_weights.csv")
    if args.skip_embeddings:
        return
    raise RuntimeError("embedding/t-SNE export is disabled in the Polars/Torch-only trainer; pass --skip-embeddings")


if __name__ == "__main__":
    main()
