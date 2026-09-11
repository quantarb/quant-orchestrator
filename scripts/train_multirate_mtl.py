"""Train the four-rate MultiRateTransformer MTL corpus."""

from __future__ import annotations

import argparse
import csv
import resource
from collections import Counter
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
from quant_orchestrator.research_tools.streaming_context import StreamingContext, StreamingFamilyContext, context_ordered_anchors
from quant_orchestrator.research_tools.sequence_training import sequence_anchors, window_supervision
from quant_orchestrator.research_tools.document_sequences import DOCUMENT_CONTRACT, document_anchors, document_window, prediction_positions, validate_document_predictions
from quant_orchestrator.research_tools.epoch_evaluation import wait_for_epoch_backtest
from quant_orchestrator.research_tools.multirate_supervision import StreamingSupervision, instrument_asset_groups, input_event_families
from quant_orchestrator.research_tools.multirate_objectives import (
    RECONSTRUCTION_CONTRACT, reconstruction_mask, reconstruction_targets, family_channels, feature_family_layout,
)
from quant_orchestrator.research_tools.multirate_audit import verify_corpus_files
from quant_orchestrator.research_tools.ntp_evaluation import NTPPersistenceAudit
from quant_orchestrator.research_tools.multirate_batch import BatchTensors, supervision_counts

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
    TRADE_EVENT_SUPERVISED_TASK_NAMES,
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


def _date_expression(table: pl.DataFrame, column: str) -> pl.Expr:
    value = pl.col(column)
    if table.schema[column] == pl.String:
        value = value.str.to_datetime(strict=False)
    return value.cast(pl.Datetime, strict=False)


def _normalization_stats(table, columns, *, cutoff=None, saved=None, symbols=None):
    """Fit on historical rows only, or reuse a checkpoint without scanning data."""
    if saved is not None:
        mean, scale = saved
        if len(mean) != len(columns) or len(scale) != len(columns):
            raise ValueError("Checkpoint normalization dimensions do not match input columns")
        if any(not math.isfinite(float(v)) for v in mean) or any(
            not math.isfinite(float(v)) or float(v) <= 0 for v in scale
        ):
            raise ValueError("Checkpoint normalization must have finite means and positive scales")
        return list(mean), list(scale)
    values = table.lazy() if isinstance(table, pl.DataFrame) else table
    if symbols is not None:
        values = values.filter(pl.col("symbol").is_in(list(symbols)))
    if cutoff is not None:
        values = values.filter(pl.col("date") < _as_datetime(cutoff))
    finite = [pl.when(pl.col(column).is_finite()).then(pl.col(column)) for column in columns]
    stats = values.select(
        *[value.mean().alias(f"mean_{i}") for i, value in enumerate(finite)],
        *[value.std(ddof=0).alias(f"std_{i}") for i, value in enumerate(finite)],
    ).collect(engine="streaming").row(0)
    mean = [float(v) if v is not None and math.isfinite(float(v)) else 0.0 for v in stats[:len(columns)]]
    scale = [float(v) if v is not None and math.isfinite(float(v)) and float(v) > 1e-6 else 1.0 for v in stats[len(columns):]]
    return mean, scale


def _is_missing(value: object) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _validate_inference_checkpoint(checkpoint):
    missing = [name for name in ("normalization", "labels", "configuration")
               if not isinstance(checkpoint.get(name), dict) or not checkpoint[name]]
    normalization = checkpoint.get("normalization", {})
    if isinstance(normalization, dict):
        missing.extend(f"normalization.{rate}" for rate in ("annual", "quarterly", "daily", "sparse")
                       if rate not in normalization)
    if missing:
        raise ValueError(
            "Checkpoint lacks reproducible inference metadata: " + ", ".join(missing)
            + ". Restore the exact training metadata or retrain with the current trainer; "
            "normalization and label mappings cannot be fitted from live data."
        )


def _encode_labels(values: pl.Series, vocabulary=None) -> tuple[torch.Tensor, list[str], dict[str, int]]:
    normalized = values.cast(pl.String).fill_null("Unknown")
    labels = sorted(normalized.unique().to_list()) if vocabulary is None else list(vocabulary)
    mapping = {value: index for index, value in enumerate(labels)}
    return torch.tensor([mapping.get(value, -100) for value in normalized.to_list()], dtype=torch.long), labels, mapping


def _symbol_rows(table: pl.DataFrame, symbol: str) -> pl.DataFrame:
    return table.filter(pl.col("symbol").cast(pl.String).str.to_uppercase() == str(symbol).upper())


def _read_parquet_polars(path: Path, columns: list[str] | None = None) -> pl.DataFrame:
    """Collect a small metadata table (not a bounded-memory corpus reader)."""
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
            # Convert explicitly to nanosecond epoch integers before crossing
            # the Polars/Torch boundary.  Direct ``Datetime.to_torch()`` may
            # rescale the values to microseconds, which would turn real dates
            # into bogus 1970 prediction dates during inference.
            dates = group["date"].cast(pl.Datetime, strict=False).dt.epoch("ns").to_torch().flatten()
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
        "annual_timestamps", "quarterly_timestamps", "daily_timestamps", "sparse_timestamps",
        "issuer_daily", "issuer_daily_padding", "issuer_daily_timestamps",
        "issuer_sparse", "issuer_sparse_padding", "issuer_sparse_timestamps",
    })

    def __init__(self, metadata: dict[str, object], factory):
        super().__init__(metadata)
        self._factory = factory
        self._loaded = False

    def _materialize(self) -> None:
        if not self._loaded:
            super().update(self._factory())
            self._loaded = True

    def release(self):
        for key in self._LAZY_KEYS:
            self.pop(key, None)
        self._loaded = False

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
    if isinstance(table, (_IndexedTable, StreamingContext)):
        values, padding, dates, _ = table.window(symbol, anchor, length)
        return values, padding, dates
    rows = _symbol_rows(table, symbol).filter(pl.col("date") <= _as_datetime(anchor)).tail(length)
    values = rows.select(value_columns).fill_nan(None).fill_null(float("nan")).to_torch().to(torch.float32) if len(rows) else torch.empty((0, len(value_columns)), dtype=torch.float32)
    dates = rows["date"].cast(pl.Datetime, strict=False).dt.epoch("ns").to_torch().flatten() if len(rows) else torch.empty(0, dtype=torch.long)
    padding = torch.ones(length, dtype=torch.bool)
    output = torch.full((length, len(value_columns)), float("nan"), dtype=torch.float32)
    if len(rows):
        output[-len(rows):] = values
        padding[-len(rows):] = False
    return output, padding, dates


def _sparse_window(table: pl.DataFrame, symbol: str, anchor: datetime, value_columns: list[str], length: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if isinstance(table, (_IndexedTable, StreamingContext)):
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
        dates = rows["date"].cast(pl.Datetime, strict=False).dt.epoch("ns").to_torch().flatten()
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
        _date_expression(options, "entry_date").dt.truncate("1d"),
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
    daily = daily.join(state.lazy() if isinstance(daily, pl.LazyFrame) else state, on=["symbol", "date"], how="left", suffix="_option")
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
        _date_expression(panel, "entry_date").dt.truncate("1d"),
        pl.col("dte").cast(pl.Int64, strict=False),
    ).with_columns(
        pl.col("underlying_symbol").replace_strict(issuer_map, default=None).alias("_issuer"),
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
    global ANNUAL_WINDOW, QUARTERLY_WINDOW, DAILY_WINDOW
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, help="Load an existing multirate_mtl_model.pt for inference without optimizer steps.")
    parser.add_argument("--inference-only", action="store_true", help="Skip training and export predictions from --checkpoint.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--annual-window", type=int, default=252)
    parser.add_argument("--quarterly-window", type=int, default=252)
    parser.add_argument("--daily-window", type=int, default=252)
    parser.add_argument("--issuer-context", choices=("full", "none"), default="full", help="Train a matched ablation without annual/quarterly or issuer daily/irregular context")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--self-supervision", choices=("both", "next", "masked", "none"), default="both",
        help="Auxiliary objectives for controlled comparisons; default both")
    parser.add_argument("--reconstruction-weight", type=float, default=0.1,
        help="Weight of each auxiliary reconstruction loss relative to supervised tasks; default 0.1")
    parser.add_argument("--batch-size", type=int, default=4)
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
    parser.add_argument("--mixed-precision", action="store_true", help="Use CUDA autocast with --autocast-dtype.")
    parser.add_argument("--autocast-dtype", choices=("float16", "bfloat16"), default="float16", help="Autocast format; bfloat16 requires --mixed-precision and supported CUDA hardware.")
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
    parser.add_argument("--skip-embeddings", action="store_true", default=True, help="Do not retain or write evaluation embeddings.")
    parser.add_argument("--skip-t-sne", action="store_true", help="Skip prototype t-SNE generation.")
    parser.add_argument("--skip-predictions", action="store_true", help="Do not export daily supervised-head predictions.")
    parser.add_argument("--train-end-date", help="Train only on document anchors before this YYYY-MM-DD date.")
    parser.add_argument("--train-symbols-file", type=Path, help="CSV of symbols permitted for training.")
    parser.add_argument("--test-symbols-file", type=Path, help="CSV of symbols reserved for evaluation.")
    parser.add_argument("--prediction-start-date", help="Export daily supervised-head scores on and after this YYYY-MM-DD date.")
    parser.add_argument("--prediction-end-date", help="Optional inclusive final scoring date.")
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
        "--stream-samples", action="store_true", default=True,
        help="Keep sample metadata in memory and materialize rate arrays only per batch.",
    )
    parser.add_argument('--sequence-mode', choices=('rolling', 'documents'), default=None,
                        help='Fresh runs default to calendar-quarter documents; inference restores the checkpoint contract.')
    parser.add_argument('--training-sequence-stride', type=int, default=0,
                        help='Supervise all event dates in overlapping chunks; 0 keeps one anchor per event. Must be smaller than daily-window.')
    parser.add_argument('--resume-training', action='store_true', help='Restore checkpoint weights, optimizer, and epoch/batch position')
    parser.add_argument('--backtest-price-symbols', type=Path, help='Explicit dated ticker-change mapping for adjusted backtest price reads.')
    parser.add_argument('--epoch-evaluation-dir', type=Path, help='Wait for monitor backtest reports after each epoch before training continues')
    parser.add_argument(
        "--max-samples", type=int, default=0,
        help="Optional deterministic cap on samples for a fast end-to-end smoke run.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.resume_training and (args.checkpoint is None or args.inference_only):
        parser.error('--resume-training requires --checkpoint and cannot be used for inference')
    if args.epoch_evaluation_dir and args.checkpoint_every_batches <= 0:
        parser.error('--epoch-evaluation-dir requires checkpoint-every-batches > 0')
    args.reconstruction_contract = RECONSTRUCTION_CONTRACT
    if args.reconstruction_weight < 0:
        parser.error("--reconstruction-weight must be non-negative")
    torch.manual_seed(args.seed)
    if args.context_memmap_dir is not None:
        parser.error("--context-memmap-dir is obsolete; bounded Polars windows use --context-cache-size")
    if args.inference_only and args.checkpoint is None:
        parser.error("--inference-only requires --checkpoint")
    if args.option_panel is not None or args.option_target_events is not None:
        parser.error("Prepare instrument observations and targets in the corpus before training; in-memory option-panel assembly is disabled")
    if args.validation_fraction:
        parser.error("Use --train-end-date for an explicit chronological holdout; fraction-based preprocessing is not supported")
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
    if args.autocast_dtype != "float16" and not args.mixed_precision:
        parser.error("--autocast-dtype requires --mixed-precision")
    if args.mixed_precision and args.autocast_dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        parser.error("bfloat16 autocast requires supported CUDA hardware")
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
    input_fingerprint = verify_corpus_files(root, manifest)
    feature_families = list(manifest["feature_families"])
    sparse_input_families = list(manifest["target_families"])
    # The four-rate architecture keeps a sparse stream even for a fresh
    # feature-only build with no target-event parquet yet.  A neutral family
    # preserves the tensor contract without creating a supervised task or
    # contributing any labels.
    if not sparse_input_families:
        sparse_input_families = ["__empty_sparse_family__"]
    checkpoint_payload = None
    if (args.inference_only or args.resume_training) and args.checkpoint is not None:
        checkpoint_payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        _validate_inference_checkpoint(checkpoint_payload)
        checkpoint_metrics = checkpoint_payload.get("metrics", {}) if isinstance(checkpoint_payload, dict) else {}
        if checkpoint_metrics.get("feature_families") != feature_families:
            raise ValueError("Corpus feature-family order does not match the checkpoint")
        # The checkpoint defines the sparse target schema used by its task
        # heads. Fresh live data may contain only a subset of those events.
        sparse_input_families = list(checkpoint_metrics.get("sparse_input_families", sparse_input_families))
    if checkpoint_payload:
        if checkpoint_payload.get("configuration", {}).get("reconstruction_contract") != RECONSTRUCTION_CONTRACT:
            raise ValueError("Checkpoint objective/family layout differs; retrain using the current grouped reconstruction contract")
        for key in ('annual_window', 'quarterly_window', 'daily_window', 'issuer_context'):
            if key in checkpoint_payload['configuration']:
                setattr(args, key, checkpoint_payload['configuration'][key])
    saved_sequence = (checkpoint_payload or {}).get('configuration', {}).get('sequence_mode', 'rolling')
    if checkpoint_payload and args.sequence_mode is not None and args.sequence_mode != saved_sequence:
        raise ValueError('Sequence mode differs from checkpoint; document scoring requires a document-trained checkpoint')
    args.sequence_mode = saved_sequence if checkpoint_payload else (args.sequence_mode or 'documents')
    args.document_contract = DOCUMENT_CONTRACT if args.sequence_mode == 'documents' else None
    if checkpoint_payload and args.sequence_mode == 'documents' and checkpoint_payload['configuration'].get('document_contract') != DOCUMENT_CONTRACT:
        raise ValueError('Document layout differs from checkpoint; retrain with the current document contract')
    if args.sequence_mode == 'documents' and args.legacy_rate_fusion:
        parser.error('Document scoring requires date-aware independent rate fusion')
    if min(args.annual_window, args.quarterly_window, args.daily_window) < 2:
        parser.error('Rate windows must contain at least two observations')
    ANNUAL_WINDOW, QUARTERLY_WINDOW, DAILY_WINDOW = args.annual_window, args.quarterly_window, args.daily_window
    if args.training_sequence_stride and (not args.train_end_date or not 0 < args.training_sequence_stride < DAILY_WINDOW):
        parser.error('Sequence training requires train-end-date and 0 < training-sequence-stride < daily-window')
    taxonomy = pl.read_csv(root / "taxonomy.csv").with_columns(
        pl.col("symbol").cast(pl.String).str.to_uppercase().str.strip_chars()
    )
    # Presence columns are useful for corpus diagnostics but are not model
    # inputs.  Avoid materializing them for the large 10B corpus.
    rate_columns = ["symbol", "date", *[f"value__{family}" for family in feature_families]]
    annual = pl.scan_parquet(root / "annual.parquet").select(rate_columns)
    quarterly = pl.scan_parquet(root / "quarterly.parquet").select(rate_columns)
    daily = pl.scan_parquet(root / "daily.parquet").select(rate_columns)
    sparse_path = root / "sparse_events.parquet"
    if sparse_path.exists():
        sparse = pl.scan_parquet(sparse_path)
    else:
        sparse = pl.DataFrame({
            "symbol": pl.Series([], dtype=pl.String),
            "date": pl.Series([], dtype=pl.Datetime),
            "event_date": pl.Series([], dtype=pl.Datetime),
            "target_family": pl.Series([], dtype=pl.String),
            "signal_value": pl.Series([], dtype=pl.Float32),
            **{f"text_{i}": pl.Series([], dtype=pl.Float32) for i in range(7)},
        })
    if isinstance(sparse, pl.DataFrame):
        sparse = sparse.lazy()
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
            option_symbols = set(option_events["symbol"].cast(pl.String).str.to_uppercase().to_list())
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
    symbols_by_asset = instrument_asset_groups(taxonomy)
    option_document_symbols = symbols_by_asset.get("option", set())
    option_document_start_dates: dict[str, datetime] = {}
    source_symbol_by_symbol: dict[str, str] = {}
    if "underlying_symbol" in taxonomy.columns:
        source_symbol_by_symbol = dict(taxonomy.select("symbol", "underlying_symbol").iter_rows())
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
    if "event_date" in sparse.collect_schema().names():
        sparse = sparse.with_columns(pl.col("event_date").cast(pl.Datetime, strict=False).dt.truncate("1d"))

    # Outcome-derived labels remain on their original event dates. Retain the
    # supervised scan before excluding these families entirely from inputs.
    supervised_target_map = StreamingSupervision(
        sparse, cutoff=_as_datetime(args.train_end_date) if args.train_end_date else None,
    )
    if not args.inference_only:
        coverage = supervised_target_map.coverage(
            symbols_by_asset=instrument_asset_groups(taxonomy),
            required_tasks=(*ORACLE_SUPERVISED_TASK_NAMES, *HITS_SUPERVISED_TASK_NAMES),
        )
        (output_dir / "supervision_coverage.json").write_text(json.dumps(coverage, indent=2))
        print(f"[supervision] {coverage}", flush=True)

    sparse, sparse_input_families = input_event_families(sparse, sparse_input_families)

    daily_value_columns = [f"value__{family}" for family in feature_families]
    if option_columns:
        daily_value_columns.extend(option_columns)
    annual_value_columns = daily_value_columns
    quarterly_value_columns = daily_value_columns
    # Preserve endpoint identity when multiple families occur on one date.
    sparse = sparse.with_columns(pl.col("target_family").replace_strict(
        {name: i for i, name in enumerate(sparse_input_families)}, default=None,
    ).cast(pl.Int64).alias("target_id"))
    sparse_value_columns = ["signal_value", *[f"text_{i}" for i in range(7)]]
    sparse = sparse.group_by(["symbol", "date", "target_family", "target_id"]).agg(
        *[pl.col(column).mean().alias(column) for column in sparse_value_columns],
    )

    if args.issuer_context == 'none':
        # Supervision was retained separately; this view contains inputs only.
        sparse = sparse.filter(pl.lit(False))
    # Standardize numeric rate values using the available corpus while
    # retaining NaN for coverage-aware missingness handling.
    rate_columns = {"annual": annual_value_columns, "quarterly": quarterly_value_columns, "daily": daily_value_columns}
    norms: dict[str, tuple[list[float], list[float]]] = {}
    normalized_tables = {}
    saved_norms = checkpoint_payload.get("normalization", {}) if checkpoint_payload else {}
    normalization_symbols = None
    if args.train_symbols_file:
        selected = pl.read_csv(args.train_symbols_file)
        column = "symbol" if "symbol" in selected.columns else selected.columns[0]
        normalization_symbols = set(selected[column].cast(pl.String).str.to_uppercase().str.strip_chars())
    for name, columns in rate_columns.items():
        table = {"annual": annual, "quarterly": quarterly, "daily": daily}[name]
        mean, scale = _normalization_stats(
            table, columns, cutoff=args.train_end_date, saved=saved_norms.get(name),
            symbols=({source_symbol_by_symbol.get(symbol, symbol) for symbol in normalization_symbols} if normalization_symbols is not None and name in {"annual", "quarterly"} else normalization_symbols),
        )
        norms[name] = (mean, scale)
        # Keep normalization in Polars; the model boundary receives only the
        # resulting per-window tensors later.
        normalized_tables[name] = table.with_columns([
            ((pl.col(column).cast(pl.Float32) - float(mean[index])) / float(scale[index])).alias(column)
            for index, column in enumerate(columns)
        ])
    annual, quarterly, daily = (normalized_tables[name] for name in ("annual", "quarterly", "daily"))
    raw_sparse_columns = ["signal_value", *[f"text_{i}" for i in range(7)]]
    sparse_value_columns: list[str] = []
    # Construct the wide family-specific sparse matrix in Polars.  Assigning
    # hundreds of columns one at a time fragments a large tabular frame and
    # caused the 10B/two-year option run to consume tens of GB before training.
    sparse_wide = sparse.select(["target_family", *raw_sparse_columns])
    sparse_wide_columns = []
    for family in sparse_input_families:
        for column in raw_sparse_columns:
            output_column = f"{family}__{column}"
            sparse_wide_columns.append(
                pl.when(pl.col("target_family") == family)
                .then(pl.col(column))
                .otherwise(None)
                .alias(output_column)
            )
            sparse_value_columns.append(output_column)
    sparse = sparse.with_columns(sparse_wide_columns)
    sparse_means, sparse_scales = _normalization_stats(
        sparse, sparse_value_columns, cutoff=args.train_end_date,
        saved=saved_norms.get('sparse'), symbols=normalization_symbols,
    )
    norms['sparse'] = (sparse_means, sparse_scales)
    sparse = sparse.with_columns([
        ((pl.col(column).cast(pl.Float64) - sparse_means[index]) / sparse_scales[index]).cast(pl.Float32).alias(column)
        for index, column in enumerate(sparse_value_columns)
    ])

    # Build immutable columnar indexes once. All subsequent sample windows use
    # these arrays instead of repeatedly filtering Pandas frames.
    # Feature histories stay lazy. Never concatenate the corpus into tensors
    # or trust a potentially stale normalized context cache from another fit.
    # Cache only normalized observations, never trainable encoder outputs.
    # Fits with different data, normalization or feature layouts cannot share files.
    import hashlib
    index_payload = {
        'version': 'bounded-issuer-tensors-v1', 'inputs': input_fingerprint,
        'normalization': norms, 'columns': rate_columns,
        'sparse_columns': sparse_value_columns, 'sparse_families': sparse_input_families,
        'issuer_context': args.issuer_context,
    }
    if args.option_panel is not None or args.option_target_events is not None:
        index_payload['option_configuration'] = {key: str(value) for key, value in vars(args).items() if key.startswith('option_')}
        index_payload['option_input_sha256'] = {}
        for source in (args.option_panel, args.option_target_events):
            if source is not None:
                with source.open('rb') as handle:
                    index_payload['option_input_sha256'][str(source.resolve())] = hashlib.file_digest(handle, 'sha256').hexdigest()
    index_signature = hashlib.sha256(json.dumps(index_payload, sort_keys=True).encode()).hexdigest()
    index_root = root.parent / 'normalized_context_indexes' / index_signature
    annual_index = StreamingContext(annual, annual_value_columns, index_directory=index_root/'annual')
    quarterly_index = StreamingContext(quarterly, quarterly_value_columns, index_directory=index_root/'quarterly')
    daily_index = StreamingContext(daily, daily_value_columns, index_directory=index_root/'daily')
    sparse_index = StreamingFamilyContext(sparse, sparse_value_columns, families=sparse_input_families, index_directory=index_root/'sparse')
    sparse_window_length = sparse_index.window_length

    # A document can be anchored by a regular annual observation or by a
    # sparse event.  Using their union preserves early event history even
    # when annual fundamentals begin later for a symbol.
    anchor_parts = [annual.select(["symbol", "date"]).collect(engine="streaming"), sparse.select(["symbol", "date"]).collect(engine="streaming")]
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
        exact_daily_anchors = daily.filter(pl.col("date") >= requested_anchor_date).select(["symbol", "date"]).collect(engine="streaming")
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
    # Supervised losses use the final token at its own event date. Earlier
    # tokens in an annual document can see issuer memory from the anchor.
    if supervised_target_map and args.sequence_mode != 'documents':
        event_anchors = supervised_target_map.anchors()
        if args.training_sequence_stride and not args.inference_only:
            sequence_rows, sequence_report = sequence_anchors(daily, event_anchors,
                cutoff=args.train_end_date, stride=args.training_sequence_stride, window=DAILY_WINDOW)
            (output_dir/'sequence_coverage.json').write_text(json.dumps(sequence_report,indent=2))
            print(f"[multirate-sequences] {sequence_report}", flush=True)
            # Keep pre-existing regular documents for SSL, but give supervised
            # events exclusively to the sequence documents that own their dates.
            anchors = anchors.with_columns(pl.col('date').cast(pl.Datetime('ns'))).join(
                sequence_rows.select('symbol','date'),on=['symbol','date'],how='anti')
            anchors = pl.concat([anchors, sequence_rows],how='diagonal_relaxed')
        else:
            anchors = pl.concat([anchors, event_anchors], how="diagonal_relaxed").unique(["symbol", "date"])
    if args.inference_only and args.prediction_start_date and args.prediction_start_date != "1900-01-01":
        requested_anchor_date = _as_datetime(args.prediction_start_date)
        exact_daily_anchors = daily.filter(pl.col("date") >= requested_anchor_date).select(["symbol", "date"]).collect(engine="streaming")
        anchors = pl.concat([anchors, exact_daily_anchors], how="diagonal_relaxed").unique(["symbol", "date"])
    if args.inference_only:
        if args.prediction_start_date:
            anchors = anchors.filter(pl.col("date") >= _as_datetime(args.prediction_start_date))
        if args.prediction_end_date:
            anchors = anchors.filter(pl.col("date") <= _as_datetime(args.prediction_end_date))
    if args.sequence_mode == 'documents':
        anchors = document_anchors((daily, annual, quarterly, sparse),
            start=args.prediction_start_date if args.inference_only else None,
            end=args.prediction_end_date if args.inference_only else None,
            cutoff=args.train_end_date if not args.inference_only else None)
        (output_dir/'document_coverage.json').write_text(json.dumps(dict(
            contract=DOCUMENT_CONTRACT, documents=anchors.height,
            symbols=anchors['symbol'].n_unique(), daily_history=DAILY_WINDOW,
            training_cutoff=args.train_end_date, prediction_start=args.prediction_start_date,
            prediction_end=args.prediction_end_date), indent=2))
    expected_document_dates = daily.select('symbol', 'date').filter(pl.col('symbol').is_in(taxonomy['symbol'].to_list()))
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
            if isinstance(table, StreamingContext):
                return table.version(source, anchor)
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
        value = _sparse_window(sparse_index, symbol, anchor, sparse_value_columns, sparse_window_length)
        context_cache_build_seconds[rate] += perf_counter() - started
        if args.context_cache_size:
            cache[key] = value
            while len(cache) > args.context_cache_size:
                cache.popitem(last=False)
        return value

    samples: list[dict[str, object]] = []
    taxonomy_by_symbol = {str(row["symbol"]): row for row in taxonomy.iter_rows(named=True)}
    taxonomy_symbols = set(taxonomy_by_symbol)
    anchors = context_ordered_anchors(anchors, source_symbol_by_symbol)
    preparation_started = perf_counter()
    print(f"[multirate-prepare] anchors={anchors.height} stage=sample_metadata", flush=True)
    for anchor_index, row in enumerate(anchors.iter_rows(named=True)):
        if anchor_index and anchor_index % 25000 == 0:
            print(f"[multirate-prepare] anchors={anchor_index}/{anchors.height} "
                  f"seconds={perf_counter()-preparation_started:.1f}", flush=True)
        symbol = str(row["symbol"]).upper(); anchor = _as_datetime(row["date"])
        if symbol not in taxonomy_symbols:
            continue
        source_symbol = source_symbol_by_symbol.get(symbol, symbol)
        supervision_start = row.get('document_start') if args.sequence_mode == 'documents' else row.get('supervision_start')
        def materialize(current_symbol=symbol, current_anchor=anchor, current_source=source_symbol,
                        current_supervision_start=supervision_start):
            if args.sequence_mode == 'documents':
                streams = {}
                for rate, index, source, history in (
                    ('daily', daily_index, current_symbol, DAILY_WINDOW),
                    ('annual', annual_index, current_source, ANNUAL_WINDOW),
                    ('quarterly', quarterly_index, current_source, QUARTERLY_WINDOW),
                    ('sparse', sparse_index, current_symbol, sparse_window_length),
                    ('issuer_daily', daily_index, current_source, DAILY_WINDOW),
                    ('issuer_sparse', sparse_index, current_source, sparse_window_length)):
                    if source == current_symbol and rate.startswith('issuer_'):
                        streams[rate] = streams[rate.removeprefix('issuer_')]
                    else:
                        streams[rate] = document_window(index, source, current_supervision_start, current_anchor, history)
                daily_values, daily_padding, daily_dates, _, _ = streams['daily']
                annual_values, annual_padding, annual_dates, _, _ = streams['annual']
                quarterly_values, quarterly_padding, quarterly_dates, _, _ = streams['quarterly']
                sparse_values, sparse_padding, sparse_dates, _, sparse_labels = streams['sparse']
                issuer_daily, issuer_daily_padding, issuer_daily_dates, _, _ = streams['issuer_daily']
                issuer_sparse, issuer_sparse_padding, issuer_sparse_dates, _, _ = streams['issuer_sparse']
            else:
                daily_values, daily_padding, daily_dates = cached_window("daily", daily_index, current_symbol, current_anchor, daily_value_columns, DAILY_WINDOW, current_symbol)
                if current_supervision_start is not None:
                    annual_values, annual_padding, annual_dates, _ = annual_index.sequence_window(current_source,current_supervision_start,current_anchor,ANNUAL_WINDOW)
                    quarterly_values, quarterly_padding, quarterly_dates, _ = quarterly_index.sequence_window(current_source,current_supervision_start,current_anchor,QUARTERLY_WINDOW)
                    sparse_values, sparse_padding, sparse_dates, sparse_labels = sparse_index.sequence_window(current_symbol,current_supervision_start,current_anchor,sparse_window_length)
                    issuer_daily, issuer_daily_padding, issuer_daily_dates, _ = daily_index.sequence_window(current_source,current_supervision_start,current_anchor,DAILY_WINDOW)
                    if current_source == current_symbol:
                        issuer_sparse, issuer_sparse_padding, issuer_sparse_dates = sparse_values,sparse_padding,sparse_dates
                    else:
                        issuer_sparse, issuer_sparse_padding, issuer_sparse_dates, _ = sparse_index.sequence_window(current_source,current_supervision_start,current_anchor,sparse_window_length)
                else:
                    annual_values, annual_padding, annual_dates = cached_window("annual", annual_index, current_symbol, current_anchor, annual_value_columns, ANNUAL_WINDOW, current_source)
                    quarterly_values, quarterly_padding, quarterly_dates = cached_window("quarterly", quarterly_index, current_symbol, current_anchor, quarterly_value_columns, QUARTERLY_WINDOW, current_source)
                    sparse_values, sparse_padding, sparse_labels, sparse_dates = cached_sparse_window(current_symbol, current_anchor)
                    issuer_daily, issuer_daily_padding, issuer_daily_dates = cached_window("daily", daily_index, current_symbol, current_anchor, daily_value_columns, DAILY_WINDOW, current_source)
                    issuer_sparse, issuer_sparse_padding, _, issuer_sparse_dates = cached_sparse_window(current_source, current_anchor)
            supervised_targets = torch.zeros((len(daily_values), len(SUPERVISED_TARGET_TASK_NAMES)), dtype=torch.float32)
            supervised_valid = torch.zeros_like(supervised_targets, dtype=torch.bool)
            if args.sequence_mode == 'documents' and not args.inference_only:
                supervised_targets, supervised_valid = window_supervision(
                    supervised_target_map, current_symbol, daily_dates, length=len(daily_values),
                    tasks=SUPERVISED_TARGET_TASK_NAMES, start=current_supervision_start, end=current_anchor,
                    positions=slice(1, len(daily_dates)+1))
            elif args.training_sequence_stride and not args.inference_only:
                if current_supervision_start is not None:
                    supervised_targets, supervised_valid = window_supervision(
                        supervised_target_map, current_symbol, daily_dates, length=DAILY_WINDOW,
                        tasks=SUPERVISED_TARGET_TASK_NAMES, start=current_supervision_start, end=current_anchor)
            elif len(daily_dates) and not args.inference_only:
                offset = DAILY_WINDOW - len(daily_dates)
                for position, date in enumerate(daily_dates):
                    values = supervised_target_map.get((current_symbol, _as_datetime(date)), {}) if _as_datetime(date) == current_anchor else {}
                    for task_index, task_name in enumerate(SUPERVISED_TARGET_TASK_NAMES):
                        if task_name in values:
                            supervised_targets[offset + position, task_index] = values[task_name]
                            supervised_valid[offset + position, task_index] = True
            def timestamps(dates, length):
                result = torch.full((length,), torch.iinfo(torch.long).min, dtype=torch.long)
                if len(dates):
                    result[-len(dates):] = dates
                return result
            if args.issuer_context == 'none':
                annual_values = torch.full_like(torch.as_tensor(annual_values), float('nan'))
                quarterly_values = torch.full_like(torch.as_tensor(quarterly_values), float('nan'))
                annual_padding = torch.ones(len(annual_values), dtype=torch.bool)
                quarterly_padding = torch.ones(len(quarterly_values), dtype=torch.bool)
                annual_dates = quarterly_dates = []
            result = {
                "issuer_daily": issuer_daily, "issuer_daily_padding": issuer_daily_padding,
                "issuer_daily_timestamps": timestamps(issuer_daily_dates, len(issuer_daily)),
                "issuer_sparse": issuer_sparse, "issuer_sparse_padding": issuer_sparse_padding,
                "issuer_sparse_timestamps": timestamps(issuer_sparse_dates, len(issuer_sparse)),
                "annual_timestamps": timestamps(annual_dates, len(annual_values)),
                "quarterly_timestamps": timestamps(quarterly_dates, len(quarterly_values)),
                "daily_timestamps": timestamps(daily_dates, len(daily_values)),
                "sparse_timestamps": timestamps(sparse_dates, len(sparse_values)),
                "annual": annual_values, "annual_padding": annual_padding,
                "quarterly": quarterly_values, "quarterly_padding": quarterly_padding,
                "daily": daily_values, "daily_padding": daily_padding,
                "daily_dates": [_as_datetime(date).strftime("%Y-%m-%d") for date in daily_dates],
                "sparse": sparse_values, "sparse_padding": sparse_padding, "sparse_labels": sparse_labels,
                "supervised_targets": supervised_targets, "supervised_valid": supervised_valid,
            }
            if args.sequence_mode == 'documents':
                for rate, payload in streams.items():
                    result[f'{rate}_timestamps'] = payload[3]
            return result
        metadata = {
            "symbol": symbol, "date": anchor.strftime("%Y-%m-%d"),
            "sequence_mode": args.sequence_mode,
            "document_start": _as_datetime(supervision_start).strftime('%Y-%m-%d') if supervision_start is not None else None,
            "issuer": str(taxonomy_by_symbol[symbol]["issuer"]),
            "asset_class": str(taxonomy_by_symbol[symbol]["asset_class"]),
            "issuer_context_key": (source_symbol, _epoch_ns(anchor), str(supervision_start)),
            "annual_context_key": (source_symbol, rate_version("annual", annual_index, source_symbol, anchor), str(supervision_start)),
            "quarterly_context_key": (source_symbol, rate_version("quarterly", quarterly_index, source_symbol, anchor), str(supervision_start)),
            "sector": str(taxonomy_by_symbol[symbol]["sector"]), "subsector": str(taxonomy_by_symbol[symbol]["subsector"]),
            "industry": str(taxonomy_by_symbol[symbol]["industry"]),
        }
        samples.append(_LazySample(metadata, materialize) if args.stream_samples else {**metadata, **materialize()})
    print(f"[multirate-prepare] samples={len(samples)} stage=metadata_complete "
          f"seconds={perf_counter()-preparation_started:.1f}", flush=True)
    # The indexes own the compact sorted arrays used by lazy samples. Release
    # the source Polars frames before model construction/training; retaining
    # both representations is the main avoidable memory spike on 100B runs.
    del annual, quarterly, daily, sparse
    if args.max_samples:
        if args.max_samples < 1:
            parser.error("--max-samples must be positive when provided")
        ordered_samples = sorted(samples, key=lambda item: (_as_datetime(item["date"]), str(item["symbol"])))
        count = min(args.max_samples, len(ordered_samples))
        indices = [round(i * (len(ordered_samples) - 1) / max(1, count - 1)) for i in range(count)]
        chosen = {(ordered_samples[i]["symbol"], ordered_samples[i]["date"]): ordered_samples[i] for i in indices}
        if not args.inference_only:
            rare = supervised_target_map.scan.filter(pl.any_horizontal(pl.col(name).is_not_null() for name in ORACLE_SUPERVISED_TASK_NAMES)).select("symbol", "date").collect(engine="streaming")
            required = {(row[0], row[1].strftime("%Y-%m-%d")) for row in rare.iter_rows()}
            last_by_symbol = {}
            for item in ordered_samples:
                if not args.train_end_date or item["date"] < args.train_end_date:
                    last_by_symbol[item["symbol"]] = item
            required.update((item["symbol"], item["date"]) for item in last_by_symbol.values())
            for item in ordered_samples:
                key = (item["symbol"], item["date"])
                if key in required:
                    chosen[key] = item
            if len(required) > args.max_samples:
                raise ValueError("--max-samples is too small to retain required supervised and final historical anchors")
            optional = [key for key in chosen if key not in required]
            for key in optional[max(0, args.max_samples - len(required)):]:
                del chosen[key]
        samples = sorted(chosen.values(), key=lambda item: (item["date"], item["symbol"]))
    frame = pl.DataFrame([{key: value for key, value in sample.items() if isinstance(value, (str, int))} for sample in samples])
    label_arrays: dict[str, torch.Tensor] = {}
    label_names: dict[str, list[str]] = {}
    checkpoint_labels = checkpoint_payload.get("labels", {}) if checkpoint_payload else {}
    label_fit_frame = frame
    if args.train_end_date and not args.inference_only:
        label_fit_frame = label_fit_frame.filter(pl.col("date") < args.train_end_date)
    if normalization_symbols is not None:
        label_fit_frame = label_fit_frame.filter(pl.col("symbol").is_in(list(normalization_symbols)))
    for task in DOCUMENT_TASK_NAMES[1:]:
        vocabulary = checkpoint_labels.get(task)
        if vocabulary is None:
            _, vocabulary, _ = _encode_labels(label_fit_frame[task])
        label_arrays[task], label_names[task], _ = _encode_labels(frame[task], vocabulary)
    if option_columns:
        feature_families = [*feature_families, "options"]
    feature_family_dimensions = feature_family_layout([family for family in feature_families if family != "options"])
    if option_columns:
        feature_family_dimensions["options"] = len(option_columns)
    family_names = [*feature_family_dimensions, *sparse_input_families]
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
    if args.train_end_date and not args.inference_only:
        evaluation_samples = [sample for sample in samples if _as_datetime(sample["date"]) >= _as_datetime(args.train_end_date)]
    if test_symbols is not None:
        evaluation_samples = [sample for sample in samples if str(sample["symbol"]).upper() in test_symbols]
        if not evaluation_samples:
            raise ValueError("test symbol file does not match any corpus samples")
    if args.prediction_end_date:
        evaluation_samples = [sample for sample in evaluation_samples if _as_datetime(sample["date"]) <= _as_datetime(args.prediction_end_date)]
    if args.inference_only:
        if args.prediction_start_date and args.prediction_start_date != "1900-01-01":
            requested_date = _as_datetime(args.prediction_start_date)
            evaluation_samples = [sample for sample in evaluation_samples if _as_datetime(sample["date"]) >= requested_date]
            if not evaluation_samples:
                raise ValueError(f"No feature samples exist for requested inference date {requested_date.date()}")
            symbols_on_date = {str(sample["symbol"]).upper() for sample in evaluation_samples}
            print(f"[multirate-inference] daily evaluation set: {len(evaluation_samples)} anchors, {len(symbols_on_date)} symbols from {requested_date.date()}", flush=True)
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
    if args.train_end_date and not args.inference_only:
        train_end = _as_datetime(args.train_end_date)
        train_samples = [sample for sample in train_samples if _as_datetime(sample["date"]) < train_end]
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
        saved_configuration = checkpoint_payload.get("configuration", {})
        for key in ("d_model", "num_heads", "layers", "learned_aggregation_gate", "legacy_rate_fusion", "attention_backend", "disable_document_tasks"):
            if key in saved_configuration:
                setattr(args, key, saved_configuration[key])
        enabled_document_tasks = () if args.disable_document_tasks else DOCUMENT_TASK_NAMES
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
    reconstruction_widths = {
        **{rate: tuple(feature_family_dimensions.values()) for rate in ("annual", "quarterly", "daily")},
        "sparse": (len(raw_sparse_columns),) * len(sparse_input_families),
    }
    task_bundle = add_subtoken_temporal_tasks(
        train_samples,
        family_names,
        label_names,
        feature_dimensions=reconstruction_widths,
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
    supervised_names = {spec.task_name for spec in model_tasks}
    prediction_names = {spec.task_name for spec in task_bundle.prediction_tasks
        if args.self_supervision == "both" or spec.task_name.startswith(args.self_supervision + "_")}
    active_tasks = tuple(Task(task.name, task.spec,
        args.reconstruction_weight if task.name in prediction_names else task.loss_weight)
        for task in task_bundle.tasks if task.name in supervised_names or task.name in prediction_names)
    if mrl_dimensions:
        active_tasks = (*active_tasks, Task("mrl", spec="matryoshka_document_alignment", loss_weight=args.mrl_weight))
    expected_task_names = tuple(enabled_document_tasks) + SUPERVISED_TARGET_TASK_NAMES + PREDICTION_TASK_NAMES
    asset_classes = checkpoint_payload.get("asset_classes") if checkpoint_payload else None
    asset_classes = asset_classes or sorted({sample["asset_class"] for sample in samples})
    asset_class_ids = {name: i for i, name in enumerate(asset_classes)}
    model = MultiRateTransformer(
        {"annual": len(annual_value_columns), "quarterly": len(quarterly_value_columns), "daily": len(daily_value_columns), "sparse": len(sparse_value_columns)},
        config=config,
        feature_families={
            "annual": feature_family_dimensions,
            "quarterly": feature_family_dimensions,
            "daily": feature_family_dimensions,
            "sparse": {family: len(raw_sparse_columns) for family in sparse_input_families},
        },
        modalities=asset_classes,
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
    resume_epoch = resume_batch = 0
    if args.resume_training:
        if args.grad_accumulation_steps != 1:
            parser.error('Training resume currently requires grad-accumulation-steps=1')
        optimizer.load_state_dict(checkpoint_payload['optimizer_state_dict'])
        saved = checkpoint_payload['metrics']
        resume_epoch = int(saved['epoch']) + int(saved.get('epoch_complete',False))
        resume_batch = 0 if saved.get('epoch_complete') else int(saved['batch'])
        print(f'[multirate-resume] epoch={resume_epoch+1} completed_batches={resume_batch} optimizer_restored=true',flush=True)
        if 'torch_rng_state' in checkpoint_payload:
            torch.set_rng_state(checkpoint_payload['torch_rng_state'])
            if device.type == 'cuda' and 'cuda_rng_state_all' in checkpoint_payload:
                torch.cuda.set_rng_state_all(checkpoint_payload['cuda_rng_state_all'])
        else:
            print('[multirate-resume] checkpoint has no RNG state; stochastic continuation is not bit-for-bit identical',flush=True)
    trainer = Trainer(
        model,
        [(task_bundle.corpus, active_tasks)],
        optimizer,
        seed=args.seed,
        grad_accumulation_steps=args.grad_accumulation_steps,
        autocast_dtype=getattr(torch, args.autocast_dtype) if args.mixed_precision else None,
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
    def issuer_inputs(batch, stack):
        if args.issuer_context == "none":
            return {}
        keys = {}
        ids = []
        for item in batch:
            key = item["issuer_context_key"]
            keys.setdefault(key, len(keys))
            ids.append(keys[key])
        payloads = {}
        for rate in ("daily", "sparse"):
            values = stack(f"issuer_{rate}").clone()
            padding = stack(f"issuer_{rate}_padding").bool().clone()
            # Auxiliary prediction heads use only their local rate states,
            # never this supervised fusion path.
            leading = padding[:, 0]
            values[leading, 0] = 0.
            padding[leading, 0] = False
            payloads[rate] = {"values": values, "padding": padding,
                "dates": stack(f"issuer_{rate}_timestamps"),
                "context_ids": torch.tensor(ids, device=device)}
        return payloads

    task_observations = Counter()
    task_family_observations = Counter()
    task_loss_sums = Counter()
    encoder_gradient_sums = Counter()
    def record_gradient(name):
        def record(gradient):
            if not torch.isfinite(gradient).all():
                raise RuntimeError(f"Nonfinite gradient in {name}")
            encoder_gradient_sums[name] += float(gradient.detach().abs().sum())
        return record
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and name.startswith(("annual_encoder.", "quarterly_encoder.", "encoders.daily.", "encoders.sparse.", "instrument_fusion.", "information_age.", "auto_feature_engineer.elapsed_time.")):
            parameter.register_hook(record_gradient(name.split(".layers")[0] if ".layers" in name else name.split(".")[0]))

    epoch_clocks = {}
    def training_step(module: torch.nn.Module, batch: list[dict[str, object]], active_tasks):
        if module.training:
            epoch_clocks.setdefault(trainer.current_epoch,perf_counter())
        stack = BatchTensors(batch, device)

        def context(name: str, padding_name: str) -> tuple[torch.Tensor, torch.Tensor]:
            values = stack(name).clone(); padding = stack(padding_name).bool().clone()
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
        family_masked_positions: dict[str, torch.Tensor] = {}
        token_batches = {}
        masked_batches = {"annual": (annual_batch, annual_mask), "quarterly": (quarterly_batch, quarterly_mask), "daily": (None, None), "sparse": (sparse_batch, sparse_padding_mask)}
        mask_rng = torch.Generator(device=device).manual_seed(trainer.current_epoch * 100000 + trainer.current_step)
        for rate in ("annual", "quarterly", "daily", "sparse"):
            if rate == "daily":
                batch_values, padding = context("daily", "daily_padding")
            else:
                batch_values, padding = masked_batches[rate]
            raw = stack(rate)
            selected, family_selected = reconstruction_mask(
                raw, stack(f"{rate}_padding").bool(), widths=reconstruction_widths[rate], generator=mask_rng,
                family_probability=0.15 if args.self_supervision in ("both", "masked") else 0.,
                feature_probability=0.15 if args.self_supervision in ("both", "masked") else 0.,
            )
            token_values = batch_values.clone()
            token_values[family_selected] = float("nan")
            batch_values = batch_values.clone()
            batch_values[selected] = float("nan")
            if rate in {"annual", "quarterly"}:
                representatives = {}
                first = []
                for index, item in enumerate(batch):
                    key = item[f"{rate}_context_key"]
                    representatives.setdefault(key, index)
                    first.append(representatives[key])
                first = torch.tensor(first, device=device)
                batch_values = batch_values.index_select(0, first)
                selected = selected.index_select(0, first)
                family_selected = family_selected.index_select(0, first)
                token_values = token_values.index_select(0, first)
            if module.training:
                task_observations[f"masked_{rate}_whole_family_values"] += int(family_selected.sum())
                task_observations[f"masked_{rate}_individual_values"] += int(selected.sum())
            masked_batches[rate] = (batch_values, padding); masked_positions[rate] = selected
            family_masked_positions[rate] = family_selected
            token_batches[rate] = token_values
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
            daily_dates=stack("daily_timestamps"), annual_dates=stack("annual_timestamps"),
            quarterly_dates=stack("quarterly_timestamps"), sparse_dates=stack("sparse_timestamps"),
            daily_modality_ids=torch.tensor([asset_class_ids[item["asset_class"]] for item in batch], device=device)[:, None].expand(-1, daily_batch.shape[1]),
            rate_context_ids=rate_context_ids,
            issuer_streams=issuer_inputs(batch, stack),
            **{f"{rate}_family_presence": family_channels(
                torch.isfinite(stack(rate)) & ~stack(f"{rate}_padding").bool().unsqueeze(-1),
                reconstruction_widths[rate],
            ).any(-1) for rate in reconstruction_widths},
            compute_document_outputs=not args.disable_document_tasks,
        )
        token_predictions = {}
        if args.self_supervision in ("both", "masked"):
            # Token MTP gets a separate whole-family view, with no feature-level
            # corruption borrowed from the subtoken MTP pass.
            token_predictions = {name: prediction for name, prediction in module(
                token_batches["daily"], token_batches["annual"], token_batches["quarterly"], token_batches["sparse"],
                daily_padding_mask=daily_mask, annual_padding_mask=annual_mask,
                quarterly_padding_mask=quarterly_mask, sparse_padding_mask=sparse_padding_mask,
                daily_dates=stack("daily_timestamps"), annual_dates=stack("annual_timestamps"),
                quarterly_dates=stack("quarterly_timestamps"), sparse_dates=stack("sparse_timestamps"),
                daily_modality_ids=torch.tensor([asset_class_ids[item["asset_class"]] for item in batch], device=device)[:, None].expand(-1, daily_batch.shape[1]),
                rate_context_ids=rate_context_ids,
                **{f"{rate}_family_presence": family_channels(
                    torch.isfinite(stack(rate)) & ~stack(f"{rate}_padding").bool().unsqueeze(-1),
                    reconstruction_widths[rate],
                ).any(-1) for rate in reconstruction_widths},
                compute_document_outputs=False,
            )["prediction_outputs"].items() if name.startswith("masked_") and name.endswith("_token")}
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
            if args.sequence_mode != 'documents' and not args.training_sequence_stride:
                valid[:, :-1] = False
            if not valid.any():
                continue
            if module.training:
                total, by_asset = supervision_counts(valid, batch)
                task_observations[name] += total
                for asset, count in by_asset.items():
                    task_observations[f"{asset}:{name}"] += count
            target = supervised_targets[:, :, task_index]
            prediction = output["token_outputs"][name].squeeze(-1)
            if name in ORACLE_SUPERVISED_TASK_NAMES or name in FUND_ACTIVITY_SUPERVISED_TASK_NAMES or name in HOLDER_ACTIVITY_SUPERVISED_TASK_NAMES or name in TRADE_EVENT_SUPERVISED_TASK_NAMES:
                task_losses[name] = nn.functional.binary_cross_entropy_with_logits(prediction[valid], target[valid])
            else:
                task_losses[name] = nn.functional.smooth_l1_loss(prediction[valid], target[valid])
        family_labels = torch.arange(len(family_names), device=device).view(1, -1).expand(len(batch), -1)
        family_valid = torch.zeros((len(batch), len(family_names)), dtype=torch.bool, device=device)
        for rate in ("annual", "quarterly", "daily", "sparse"):
            raw = stack(rate)
            if rate == "sparse":
                local_count, width, family_offset = len(sparse_input_families), len(raw_sparse_columns), len(feature_family_dimensions)
                observed = torch.isfinite(raw).reshape(raw.shape[0], raw.shape[1], local_count, width).any(dim=-1).any(dim=1)
            else:
                family_offset = 0
                offset = 0
                family_observed = []
                for family in feature_family_dimensions:
                    width = feature_family_dimensions[family]
                    family_observed.append(torch.isfinite(raw[:, :, offset:offset + width]).any(dim=-1).any(dim=1))
                    offset += width
                observed = torch.stack(family_observed, dim=1)
                local_count = len(feature_family_dimensions)
            family_valid[:, family_offset:family_offset + local_count] |= observed
        if "family" in active_names and family_valid.any():
            task_losses["family"] = nn.functional.cross_entropy(output["document_outputs"]["family"][family_valid], family_labels[family_valid])
        for rate in reconstruction_widths if prediction_names else ():
            targets = reconstruction_targets(
                stack(rate), stack(f"{rate}_padding").bool(), stack(f"{rate}_timestamps"),
                masked_positions[rate], reconstruction_widths[rate], family_selected=family_masked_positions[rate],
            )
            for objective, (target, valid) in targets.items():
                kind, level = objective.split("_")
                name = f"{kind}_{rate}_{level}"
                prediction = token_predictions[name] if objective == "masked_token" and name in token_predictions else output["prediction_outputs"][name]
                if prediction.shape != target.shape:
                    raise ValueError(f"Reconstruction shape mismatch for {name}: {prediction.shape} != {target.shape}")
                if name in active_names and valid.any():
                    task_losses[name] = nn.functional.mse_loss(prediction[valid], target[valid])
                if module.training and name in active_names:
                    task_observations[name] += int(valid.sum())
                    per_family = valid if level == "subtoken" else family_channels(valid, reconstruction_widths[rate])
                    counts = per_family.sum(dim=(0, 1, 3)).detach().cpu().tolist()
                    names = sparse_input_families if rate == "sparse" else list(feature_family_dimensions)
                    for family, count in zip(names, counts):
                        task_family_observations[f"{name}:{family}"] += count
        if module.training:
            for name, loss in task_losses.items():
                task_loss_sums[name] += float(loss.detach())
            for rate in ("annual", "quarterly", "daily", "sparse"):
                task_observations[f"masked_{rate}"] += int(masked_positions[rate].sum())
        for item in batch:
            if isinstance(item, _LazySample):
                item.release()
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
            best_loss = stopping_loss; best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}; stale_epochs = 0
        else:
            stale_epochs += 1
        scope = 'remaining_batches' if epoch == resume_epoch and resume_batch else 'full_epoch'
        print(f"epoch {epoch + 1}/{args.epochs} loss={epoch_loss:.6f} validation_loss={val_loss:.6f} loss_scope={scope}", flush=True)
        if args.epoch_evaluation_dir:
            # Release unused activation reservations before the evaluator starts
            # another CUDA process. Model and optimizer tensors remain intact.
            import gc
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            wait_for_epoch_backtest(args.epoch_evaluation_dir,epoch+1)
        if stale_epochs >= max(1, args.patience):
            print(f"early stopping after epoch {epoch + 1}; best_loss={best_loss:.6f}", flush=True)
            return True
        return False

    training_started = perf_counter()

    def save_batch_checkpoint(epoch: int, batch_index: int, batch_loss: float, *, epoch_complete: bool = False) -> None:
        checkpoint_path = output_dir / "multirate_mtl_checkpoint_latest.pt"
        temporary_path = output_dir / "multirate_mtl_checkpoint_latest.pt.tmp"
        payload = {
            "state_dict": model.state_dict(),
            "asset_classes": asset_classes,
            "labels": label_names,
            "normalization": norms,
            "configuration": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "optimizer_state_dict": optimizer.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all() if device.type == 'cuda' else [],
            "task_family_observations": dict(task_family_observations),
            "metrics": {
                "epoch": epoch,
                "batch": batch_index,
                "epoch_complete": epoch_complete,
                "batch_loss": batch_loss,
                "feature_families": feature_families,
                "sparse_input_families": sparse_input_families,
                "tasks": list(expected_task_names),
                "train_samples": len(train_samples),
            },
        }
        torch.save(payload, temporary_path)
        os.replace(temporary_path, checkpoint_path)
        if epoch_complete:
            snapshots = output_dir / 'epoch_checkpoints'
            snapshots.mkdir(exist_ok=True)
            # Atomic replacement of latest leaves this immutable inode intact.
            os.link(checkpoint_path, snapshots / f'epoch_{epoch+1:04d}.pt')
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
        elapsed = perf_counter() - epoch_clocks.get(epoch,training_started)
        processed = batch_index - (resume_batch if epoch == resume_epoch else 0)
        rate = processed / max(elapsed, 1e-6)
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
            save_batch_checkpoint(epoch, batch_index, batch_loss, epoch_complete=batch_index == total_batches)

    if args.epoch_evaluation_dir and not args.inference_only:
        args.epoch_evaluation_dir.mkdir(parents=True,exist_ok=True)
        (args.epoch_evaluation_dir/'training_gate.json').write_text(json.dumps(
            dict(stage='training',epoch=resume_epoch+1)))
    losses = [] if args.inference_only else trainer.fit(
        args.epochs,
        training_step,
        on_epoch_end=epoch_end,
        on_batch_end=training_progress,
        start_epoch=resume_epoch,
        start_batch=resume_batch,
    )

    (output_dir / "training_diagnostics.json").write_text(json.dumps({
        "task_observations": dict(task_observations), "task_family_observations": dict(task_family_observations), "task_loss_sums": dict(task_loss_sums),
        "encoder_gradient_abs_sum": dict(encoder_gradient_sums),
        "peak_process_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0,
    }, indent=2))
    if not args.inference_only:
        missing = [f"{asset}:{name}" for asset in {sample["asset_class"] for sample in train_samples}
                   for name in (*ORACLE_SUPERVISED_TASK_NAMES, *HITS_SUPERVISED_TASK_NAMES)
                   if not task_observations[f"{asset}:{name}"]]
        if missing:
            raise ValueError(f"Training completed without observed supervision for: {missing}")

    if best_state is not None:
        model.load_state_dict(best_state)

    evaluation_samples = sorted(evaluation_samples, key=lambda item: (item["symbol"], item["date"]))
    model.eval(); predictions: dict[str, list[torch.Tensor]] = {name: [] for name in enabled_document_tasks[1:]}; states: list[torch.Tensor] = []; family_states: list[torch.Tensor] = []; family_valid_rows: list[torch.Tensor] = []
    prediction_rows: list[dict[str, object]] = []
    prediction_count = 0
    prediction_temporary = output_dir / "supervised_predictions.csv.tmp"
    prediction_temporary.write_text("")
    prediction_start = _as_datetime(args.prediction_start_date) if args.prediction_start_date else None
    ntp_path = output_dir / "ntp_evaluation.sqlite"
    if ntp_path.exists():
        ntp_path.unlink()
    ntp_audit = NTPPersistenceAudit(ntp_path,
        start_ns=_epoch_ns(prediction_start) if prediction_start else None,
        end_ns=_epoch_ns(_as_datetime(args.prediction_end_date)) if args.prediction_end_date else None)
    family_correct = 0
    family_total = 0
    with torch.inference_mode():
        # Evaluation materializes all prototype states and is more memory-sensitive
        # than the training step. Keep the scalable training batch size, but use a
        # bounded evaluation batch to avoid accelerator kernel failures on large
        # corpora.
        eval_batch_size = min(args.batch_size, 128 if args.inference_only and args.sequence_mode != 'documents' else 64)
        evaluation_started = perf_counter()
        for start in range(0, len(evaluation_samples), eval_batch_size):
            batch = evaluation_samples[start:start + eval_batch_size]
            if args.inference_only:
                batch_symbols = ",".join(dict.fromkeys(str(item.get("symbol", "?")) for item in batch))
                print(
                    f"[multirate-inference] scoring {'documents' if args.sequence_mode == 'documents' else 'symbols'} {start + 1}-{start + len(batch)}/{len(evaluation_samples)}: {batch_symbols} "
                    f"elapsed_s={perf_counter() - evaluation_started:.1f} "
                    f"observations_per_s={start / max(perf_counter() - evaluation_started, 1e-9):.2f} "
                    f"score_rows={prediction_count} score_rows_per_s={prediction_count / max(perf_counter() - evaluation_started, 1e-9):.2f}",
                    flush=True,
                )
            stack = BatchTensors(batch, device)
            def context(name: str, padding_name: str) -> tuple[torch.Tensor, torch.Tensor]:
                values = stack(name).clone(); padding = stack(padding_name).bool().clone()
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
            inference_context_ids = {}
            for rate in ("annual", "quarterly"):
                keys = [item[f"{rate}_context_key"] for item in batch]
                unique = {key: index for index, key in enumerate(dict.fromkeys(keys))}
                if len(unique) < len(batch):
                    inference_context_ids[rate] = torch.tensor([unique[key] for key in keys], device=device)
            output = model(daily_batch, annual_batch, quarterly_batch, sparse_batch, compute_document_outputs=not args.skip_embeddings or not args.disable_document_tasks, rate_context_ids=inference_context_ids,
                **({f"{rate}_family_presence": family_channels(torch.isfinite(stack(rate)) & ~stack(f"{rate}_padding").bool().unsqueeze(-1), widths).any(-1)
                    for rate, widths in reconstruction_widths.items()} if args.sequence_mode == 'documents' else {}), issuer_streams=issuer_inputs(batch, stack), daily_padding_mask=daily_mask, annual_padding_mask=annual_mask, quarterly_padding_mask=quarterly_mask, sparse_padding_mask=sparse_padding_mask, daily_dates=stack("daily_timestamps"), annual_dates=stack("annual_timestamps"), quarterly_dates=stack("quarterly_timestamps"), sparse_dates=stack("sparse_timestamps"), daily_modality_ids=torch.tensor([asset_class_ids[item["asset_class"]] for item in batch], device=device)[:, None].expand(-1, daily_batch.shape[1]))
            for rate, widths in reconstruction_widths.items():
                ntp_audit.update([str(item["symbol"]) for item in batch], rate,
                    sparse_input_families if rate == "sparse" else list(feature_family_dimensions), widths,
                    stack(rate), stack(f"{rate}_padding").bool(), stack(f"{rate}_timestamps"), output["prediction_outputs"])
            if prediction_start is not None:
                score_names = tuple(SUPERVISED_TARGET_TASK_NAMES)
                score_arrays = {
                    name: (output["token_outputs"][name].squeeze(-1) if name in HITS_SUPERVISED_TASK_NAMES else torch.sigmoid(output["token_outputs"][name].squeeze(-1))).cpu()
                    for name in score_names
                }
                for row_index, item in enumerate(batch):
                    for position, date in prediction_positions(item, start=args.prediction_start_date):
                        score_row = {name: float(values[row_index, position]) for name, values in score_arrays.items()}
                        prediction_rows.append({"symbol": item["symbol"], "date": date, "information_date": date, **score_row})
            if prediction_rows:
                with prediction_temporary.open("a", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0]))
                    if prediction_count == 0:
                        writer.writeheader()
                    writer.writerows(prediction_rows)
                prediction_count += len(prediction_rows)
                prediction_rows.clear()
            if not args.skip_embeddings:
                states.append(output["document_prototypes"].cpu())
            if not args.skip_embeddings or "family" in enabled_document_tasks:
                family_labels = torch.arange(len(family_names), device=device).view(1, -1).expand(len(batch), -1)
                family_valid = torch.zeros((len(batch), len(family_names)), dtype=torch.bool, device=device)
                for rate in ("annual", "quarterly", "daily", "sparse"):
                    raw = stack(rate)
                    if rate == "sparse":
                        local_count, width, family_offset = len(sparse_input_families), len(raw_sparse_columns), len(feature_family_dimensions)
                        observed = torch.isfinite(raw).reshape(raw.shape[0], raw.shape[1], local_count, width).any(dim=-1).any(dim=1)
                    else:
                        family_offset = 0
                        offset = 0
                        family_observed = []
                        for family in feature_family_dimensions:
                            width = feature_family_dimensions[family]
                            family_observed.append(torch.isfinite(raw[:, :, offset:offset + width]).any(dim=-1).any(dim=1))
                            offset += width
                        observed = torch.stack(family_observed, dim=1)
                        local_count = len(feature_family_dimensions)
                    family_valid[:, family_offset:family_offset + local_count] |= observed
            if not args.skip_embeddings:
                family_states.append(output["family_document_prototypes"].cpu())
                family_valid_rows.append(family_valid.cpu())
            if "family" in enabled_document_tasks:
                family_predictions = output["document_outputs"]["family"].argmax(dim=-1)
                family_correct += int((family_predictions[family_valid] == family_labels[family_valid]).sum())
                family_total += int(family_valid.sum())
            for name in predictions: predictions[name].append(output["document_outputs"][name].argmax(dim=-1).cpu())
            for item in batch:
                if isinstance(item, _LazySample):
                    item.release()
    evaluation_label_arrays = {
        name: torch.tensor([sample[f"{name}_label"] for sample in evaluation_samples], dtype=torch.long)
        for name in predictions
    }
    task_accuracy = {}
    for name in predictions:
        targets = evaluation_label_arrays[name]
        known = targets != -100
        task_accuracy[name] = (
            float(torch.cat(predictions[name])[known].eq(targets[known]).float().mean())
            if known.any() and predictions[name] else None
        )
    if "family" in enabled_document_tasks:
        task_accuracy["family"] = family_correct / max(1, family_total)
    ntp_report = ntp_audit.report()
    ntp_report.update(training_cutoff=args.train_end_date, evaluation_samples=len(evaluation_samples),
        trained_self_supervision=(checkpoint_payload or {}).get("configuration", {}).get("self_supervision", args.self_supervision),
        sparse_input_families=sparse_input_families)
    (output_dir / "ntp_evaluation.json").write_text(json.dumps(ntp_report, indent=2))
    ntp_audit.close()
    metrics = {"device": str(device), "samples": len(samples), "training_samples": len(train_samples), "feature_families": feature_families, "sparse_input_families": sparse_input_families, "family_labels": family_names, "tasks": [*expected_task_names, *(["mrl"] if mrl_dimensions else [])], "losses": losses, "best_loss": best_loss, "epochs_completed": len(losses), "patience": args.patience, "min_delta": args.min_delta, "task_accuracy": task_accuracy, "rates": ["annual", "quarterly", "daily", "sparse"], "backbone": config.backbone, "document_tasks_disabled": args.disable_document_tasks, "learned_aggregation_gate": args.learned_aggregation_gate, "mrl": bool(mrl_dimensions), "mrl_dimensions": list(mrl_dimensions), "mrl_weight": args.mrl_weight, "train_end_date": args.train_end_date, "prediction_start_date": args.prediction_start_date}
    metrics.update({
        "validation_samples": len(validation_samples),
        "validation_losses": validation_losses,
        "best_validation_loss": best_loss,
        "early_stopping_metric": "validation_loss" if validation_samples else "training_loss",
        "validation_fraction": args.validation_fraction,
    })
    metrics.update({
        "evaluation_samples": len(evaluation_samples),
        "sequence_mode": args.sequence_mode, "prediction_rows": prediction_count,
        "scoring_seconds": perf_counter() - evaluation_started,
        "train_symbols": sorted(train_symbols) if train_symbols is not None else None,
        "test_symbols": sorted(test_symbols) if test_symbols is not None else None,
        "cacheable_rate_states": config.cacheable_rate_states,
        "group_context_batches": True,
        "mixed_precision": args.mixed_precision,
        "autocast_dtype": args.autocast_dtype if args.mixed_precision else None,
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
    metrics["input_fingerprint"] = input_fingerprint
    metrics["feature_family_dimensions"] = feature_family_dimensions
    metrics["normalized_context_indexes"] = {
        "directory": str(index_root), "fingerprint": index_signature,
        "max_build_tensor_bytes": annual_index.index_build_limit,
        "max_mapped_issuers_per_rate": 8,
        "contents": "normalized observations only; no trainable encoded states",
    }
    metrics["streaming_blocks"] = {rate: {"hits": index.cache_hits, "misses": index.cache_misses, "max_blocks": 32 * (len(sparse_input_families) if rate == "sparse" else 1), "extra_rows_per_block": 1024}
        for rate, index in {"annual": annual_index, "quarterly": quarterly_index, "daily": daily_index, "sparse": sparse_index}.items()}
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
    torch.save({"state_dict": model.state_dict(), "metrics": metrics, "labels": label_names,
                "normalization": norms, "asset_classes": asset_classes,
                "configuration": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}}, output_dir / "multirate_mtl_model.pt")
    if args.inference_only and args.sequence_mode == 'documents' and not args.max_samples and args.prediction_start_date:
        expected = expected_document_dates.filter(pl.col('date') >= _as_datetime(args.prediction_start_date))
        if args.prediction_end_date:
            expected = expected.filter(pl.col('date') <= _as_datetime(args.prediction_end_date))
        if test_symbols is not None:
            expected = expected.filter(pl.col('symbol').is_in(test_symbols))
        if not prediction_count:
            raise ValueError('Document inference produced no daily scores')
        coverage = validate_document_predictions(pl.scan_csv(prediction_temporary, try_parse_dates=True), expected)
        (output_dir/'prediction_coverage.json').write_text(json.dumps(coverage, indent=2))
    if prediction_count:
        os.replace(prediction_temporary, output_dir / "supervised_predictions.csv")
        (output_dir / "prediction_timing.json").write_text(json.dumps({
            "date_semantics": "EOD information date, not execution date",
            "information_date_column": "information_date",
            "context_rule": "Use observations with recorded corpus dates at or before the information date",
            "execution_rule": "Existing replay uses each score on the following observed trading session",
            "supervision_rule": "Train on same-date features and event labels; apply the EOD score on the following trading session",
            "calendar_rule": "Existing scoring anchors follow instrument price dates; weekend-only updates are not separately scored",
        }, indent=2))
    else:
        prediction_temporary.unlink(missing_ok=True)
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
