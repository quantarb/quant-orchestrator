"""Bounded Polars parquet windows: never collect the complete feature corpus."""
from datetime import datetime
from bisect import bisect_right
from collections import OrderedDict

import polars as pl
import torch


def context_ordered_anchors(anchors: pl.DataFrame, source_symbols: dict[str, str]) -> pl.DataFrame:
    """Group the small symbol/date index so bounded issuer caches can be reused."""
    return anchors.with_columns(
        pl.col('symbol').replace_strict(source_symbols, default=pl.col('symbol')).alias('_context_source')
    ).sort('_context_source', 'symbol', 'date').drop('_context_source')


class StreamingContext:
    def __init__(self, scan: pl.LazyFrame, columns: list[str], *, target_column=None):
        # Independent endpoint rows on one date must form one union row.
        # The query optimizer can push symbol/date predicates through these keys.
        self.scan = scan if target_column else scan.group_by("symbol", "date").agg(
            pl.col(column).drop_nulls().last() for column in columns
        )
        self.columns = columns
        self.target_column = target_column
        self._versions = OrderedDict()
        self._blocks = OrderedDict()
        self.cache_hits = 0
        self.cache_misses = 0

    @staticmethod
    def _retain(cache, key, value, limit=32):
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > limit:
            cache.popitem(last=False)

    def version(self, symbol: str, anchor: datetime) -> int:
        if symbol not in self._versions:
            dates = self.scan.filter(pl.col("symbol") == symbol).select("date").unique().top_k(4096, by="date").sort("date").collect(engine="streaming")["date"].to_list()
            self._retain(self._versions, symbol, dates)
        dates = self._versions[symbol]
        self._versions.move_to_end(symbol)
        stop = bisect_right(dates, anchor)
        if stop:
            # Conversion uses the same Polars epoch semantics as window dates.
            return int(pl.Series([dates[stop - 1]]).dt.epoch("ns")[0])
        if len(dates) < 4096:
            return -1
        # An older anchor may precede the bounded metadata cache.
        result = self.scan.filter((pl.col("symbol") == symbol) & (pl.col("date") <= anchor)).select(
            pl.col("date").max().dt.epoch("ns")
        ).collect(engine="streaming").item()
        return -1 if result is None else int(result)

    def window(self, symbol: str, anchor: datetime, length: int):
        # Projection/predicate pushdown and bounded top-k avoid a full history
        # materialization. Only the requested model window crosses to Torch.
        ordering = ["date", *([self.target_column] if self.target_column else [])]
        def read(end, count):
            return self.scan.filter((pl.col("symbol") == symbol) & (pl.col("date") <= end)).select(
                "date", *self.columns, *([self.target_column] if self.target_column else [])
            ).top_k(count, by=ordering).sort(ordering).collect(engine="streaming")
        def tensors(frame):
            return (
                frame.select(self.columns).cast(pl.Float32).fill_nan(None).fill_null(float("nan")).to_torch(),
                frame["date"].dt.epoch("ns").to_torch(),
                frame[self.target_column].to_torch().long() if self.target_column else None,
            )
        key = (symbol, anchor.year, length)
        if key not in self._blocks:
            block = read(datetime(anchor.year, 12, 31, 23, 59, 59, 999999), length + 1024)
            self._retain(self._blocks, key, (block, block["date"].to_list(), tensors(block)))
            self.cache_misses += 1
        else:
            self.cache_hits += 1
            self._blocks.move_to_end(key)
        block, dates, converted = self._blocks[key]
        stop = bisect_right(dates, anchor)
        if stop < length and block.height == length + 1024:
            frame = read(anchor, length)
            selected_values, selected_dates, selected_targets = tensors(frame)
        else:
            start = max(0, stop - length)
            selected_values, selected_dates = converted[0][start:stop], converted[1][start:stop]
            selected_targets = converted[2][start:stop] if self.target_column else None
        values = torch.full((length, len(self.columns)), float("nan"), dtype=torch.float32)
        padding = torch.ones(length, dtype=torch.bool)
        if len(selected_dates):
            values[-len(selected_dates):] = selected_values
            padding[-len(selected_dates):] = False
        return values, padding, selected_dates, selected_targets


class StreamingFamilyContext(StreamingContext):
    """Reserve history for each sparse family, then union equal-date observations."""

    def __init__(self, scan, columns, *, families, family_column="target_family", history_per_family=16):
        super().__init__(scan, columns)
        self.history_per_family = history_per_family
        self.family_contexts = [
            StreamingContext(scan.filter(pl.col(family_column) == family), columns)
            for family in families
        ]
        self.window_length = history_per_family * len(self.family_contexts)

    def window(self, symbol, anchor, length):
        if length != self.window_length:
            raise ValueError("Sparse window must reserve history for every family")
        parts, date_parts = [], []
        for context in self.family_contexts:
            values, padding, dates, _ = context.window(symbol, anchor, self.history_per_family)
            parts.append(values[~padding])
            date_parts.append(dates)
        self.cache_hits = sum(context.cache_hits for context in self.family_contexts)
        self.cache_misses = sum(context.cache_misses for context in self.family_contexts)
        values = torch.full((length, len(self.columns)), float("nan"))
        padding = torch.ones(length, dtype=torch.bool)
        joined_dates = torch.cat(date_parts)
        dates, inverse = joined_dates.unique(sorted=True, return_inverse=True)
        if len(dates):
            joined = torch.cat(parts)
            rows, columns = torch.isfinite(joined).nonzero(as_tuple=True)
            values[inverse[rows] + length - len(dates), columns] = joined[rows, columns]
            padding[-len(dates):] = False
        return values, padding, dates, None
