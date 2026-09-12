"""Bounded Polars parquet windows: never collect the complete feature corpus."""
from datetime import datetime
from pathlib import Path
import hashlib
import os
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
    def __init__(self, scan: pl.LazyFrame, columns: list[str], *, target_column=None, index_directory: Path | None = None):
        # Independent endpoint rows on one date must form one union row.
        # The query optimizer can push symbol/date predicates through these keys.
        self.scan = scan if target_column else scan.group_by("symbol", "date").agg(
            pl.col(column).drop_nulls().last() for column in columns
        )
        self.columns = columns
        self.target_column = target_column
        self.index_directory = index_directory
        self._issuer_indexes = OrderedDict()
        self._oversized_issuers = set()
        # The expanded 10B corpus reaches about 136 MiB per daily issuer.
        # Keep those histories on the read-only disk-index path while retaining
        # a bounded fallback for larger individual sources.
        self.index_build_limit = 256 * 1024**2
        if index_directory is not None:
            index_directory.mkdir(parents=True, exist_ok=True)
        self._versions = OrderedDict()
        self._blocks = OrderedDict()
        self._sequence_blocks = OrderedDict()
        self.sequence_cache_bytes = 0
        self.sequence_cache_limit = 32 * 1024**2
        self.cache_hits = 0
        self.cache_misses = 0

    @staticmethod
    def _retain(cache, key, value, limit=32):
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > limit:
            cache.popitem(last=False)

    def _issuer_index(self, symbol, *, families=None):
        """Build at most one bounded issuer table; reuse read-only mapped tensors.

        Source/normalization/layout fingerprints belong to the caller-supplied
        directory. Never mmap a cache produced from a different fit or corpus.
        Oversized issuers use the bounded calendar-window path instead.
        """
        if self.index_directory is None or symbol in self._oversized_issuers:
            return None
        if symbol in self._issuer_indexes:
            self.cache_hits += 1
            self._issuer_indexes.move_to_end(symbol)
            return self._issuer_indexes[symbol]
        self.cache_misses += 1
        path = self.index_directory / (hashlib.sha256(symbol.encode()).hexdigest() + '.pt')
        if not path.exists():
            family_column = self.family_column if families is not None else None
            extra_column = family_column or self.target_column
            scan = self._family_scan if families is not None else self.scan
            order = ['date', *([extra_column] if extra_column else [])]
            row_limit = self.index_build_limit // max(1, 4 * len(self.columns) + 16)
            frame = scan.filter(pl.col('symbol') == symbol).select(
                'date', *self.columns, *([extra_column] if extra_column else [])
            ).sort(order).head(row_limit + 1).collect(engine='streaming')
            if frame.height > row_limit:
                self._oversized_issuers.add(symbol)
                return None
            values = frame.select(pl.col(c).cast(pl.Float32).fill_nan(None).fill_null(float('nan')) for c in self.columns).to_torch()
            dates = frame['date'].dt.epoch('ns').to_torch()
            extra = (frame[family_column].replace_strict({f:i for i,f in enumerate(families)}).to_torch().long()
                     if family_column else frame[extra_column].to_torch().long() if extra_column else None)
            temporary = path.with_suffix(f'.{os.getpid()}.tmp')
            torch.save({'values': values, 'dates': dates, 'extra': extra}, temporary)
            os.replace(temporary, path)
            del frame, values, dates, extra
        stored = torch.load(path, mmap=True, weights_only=True)
        block = (stored['values'], stored['dates'], stored['extra'])
        self._retain(self._issuer_indexes, symbol, block, limit=8)
        return block

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
        indexed = self._issuer_index(symbol)
        if indexed is not None:
            raw, ns, labels = indexed
            stop = int(torch.searchsorted(ns, pl.Series([anchor]).dt.epoch('ns').to_torch(), right=True)[0])
            first = max(0, stop-length)
            values = torch.full((length, len(self.columns)), float('nan'))
            padding = torch.ones(length, dtype=torch.bool)
            if stop > first:
                values[-(stop-first):] = raw[first:stop]
                padding[-(stop-first):] = False
            return values, padding, ns[first:stop], labels[first:stop] if labels is not None else None
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

    def _sequence_block(self, symbol, start, end, history, *, families=None):
        """Cache a bounded calendar partition, then slice by exact source dates.

        Reading future rows into a cache does not make them model inputs: callers
        always slice at end and retain exactly history observations at start.
        """
        key = (symbol, start.year, end.year, history)
        if key in self._sequence_blocks:
            self.cache_hits += 1
            self._sequence_blocks.move_to_end(key)
            return self._sequence_blocks[key][0]
        self.cache_misses += 1
        lower = datetime(start.year, 1, 1)
        upper = datetime(end.year + 1, 1, 1)
        family_column = self.family_column if families is not None else None
        ordering = ['date', *([family_column] if family_column else
                                [self.target_column] if self.target_column else [])]
        scan = self._family_scan if families is not None else self.scan
        selected = scan.filter(pl.col('symbol') == symbol).select(
            'date', *self.columns, *([family_column] if family_column else
                                    [self.target_column] if self.target_column else []))
        before = selected.filter(pl.col('date') < lower)
        if family_column:
            before = before.sort(ordering).group_by(family_column).tail(history).select(selected.collect_schema().names())
        else:
            before = before.top_k(history, by=ordering)
        updates = selected.filter((pl.col('date') >= lower) & (pl.col('date') < upper))
        # Bound transient tensors as well as retained cache memory. Dense or
        # unusually long partitions fall back to the exact requested window.
        row_limit = max(history, self.sequence_cache_limit // max(1, 4 * len(self.columns) + 16))
        frame = pl.concat([before, updates]).sort(ordering).head(row_limit + 1).collect(engine='streaming', optimizations=pl.QueryOptFlags(slice_pushdown=False))
        if frame.height > row_limit:
            return None
        values = frame.select(pl.col(c).cast(pl.Float32).fill_nan(None).fill_null(float('nan')) for c in self.columns).to_torch()
        dates = frame['date'].dt.epoch('ns').to_torch()
        extra = (frame[family_column].replace_strict({f:i for i,f in enumerate(families)}).to_torch().long()
                 if family_column else frame[self.target_column].to_torch().long() if self.target_column else None)
        block = (values, dates, extra)
        size = sum(t.numel() * t.element_size() for t in block if t is not None)
        while self._sequence_blocks and (self.sequence_cache_bytes + size > self.sequence_cache_limit or len(self._sequence_blocks) >= 32):
            _, (_, removed) = self._sequence_blocks.popitem(last=False)
            self.sequence_cache_bytes -= removed
        if size <= self.sequence_cache_limit:
            self._sequence_blocks[key] = (block, size)
            self.sequence_cache_bytes += size
        return block

    def sequence_window(self, symbol: str, start: datetime, end: datetime, history: int):
        """Retain start history and every subsequent update using indexed slices."""
        if end < start:
            raise ValueError('Sequence end precedes start')
        block = self._issuer_index(symbol)
        if block is None:
            block = self._sequence_block(symbol, start, end, history)
        if block is None:
            return self._uncached_sequence_window(symbol, start, end, history)
        raw, ns, targets = block
        bounds = pl.Series([start, end]).dt.epoch('ns').to_torch()
        left, right = torch.searchsorted(ns, bounds, right=True).tolist()
        first = max(0, left - history)
        selected = raw[first:right]
        length = max(history, len(selected))
        values = torch.full((length, len(self.columns)), float('nan'))
        padding = torch.ones(length, dtype=torch.bool)
        if len(selected):
            values[-len(selected):] = selected
            padding[-len(selected):] = False
        return values, padding, ns[first:right], targets[first:right] if targets is not None else None

    def _uncached_sequence_window(self, symbol: str, start: datetime, end: datetime, history: int):
        """Retain history at the first supervised date plus all subsequent updates."""
        ordering = ['date', *([self.target_column] if self.target_column else [])]
        selected = self.scan.filter(pl.col('symbol') == symbol).select(
            'date', *self.columns, *([self.target_column] if self.target_column else []))
        before = selected.filter(pl.col('date') <= start).top_k(history, by=ordering)
        updates = selected.filter((pl.col('date') > start) & (pl.col('date') <= end))
        frame = pl.concat([before, updates]).sort(ordering).collect(engine='streaming')
        length = max(history, frame.height)
        values = torch.full((length,len(self.columns)),float('nan'))
        padding = torch.ones(length,dtype=torch.bool)
        dates = frame['date'].dt.epoch('ns').to_torch()
        targets = frame[self.target_column].to_torch().long() if self.target_column else None
        if frame.height:
            values[-frame.height:] = frame.select(pl.col(c).cast(pl.Float32).fill_nan(None).fill_null(float('nan')) for c in self.columns).to_torch()
            padding[-frame.height:] = False
        return values,padding,dates,targets


class StreamingFamilyContext(StreamingContext):
    """Reserve history for each sparse family, then union equal-date observations."""

    def __init__(self, scan, columns, *, families, family_column="target_family", history_per_family=16, index_directory: Path | None = None):
        super().__init__(scan, columns, index_directory=index_directory)
        self.history_per_family = history_per_family
        self.family_column = family_column
        self.families = list(families)
        self._family_scan = scan.filter(pl.col(family_column).is_in(self.families)).group_by('symbol', 'date', family_column).agg(
            pl.col(c).drop_nulls().last() for c in columns)
        self.family_contexts = [
            StreamingContext(scan.filter(pl.col(family_column) == family), columns)
            for family in families
        ]
        self.window_length = history_per_family * len(self.family_contexts)

    def window(self, symbol, anchor, length):
        if length != self.window_length:
            raise ValueError("Sparse window must reserve history for every family")
        if self.index_directory is not None:
            return self.sequence_window(symbol, anchor, anchor, length)
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

    def sequence_window(self, symbol, start, end, history):
        if end < start:
            raise ValueError('Sequence end precedes start')
        block = self._issuer_index(symbol, families=self.families)
        if block is None:
            block = self._sequence_block(symbol, start, end, self.history_per_family, families=self.families)
        if block is None:
            return self._uncached_family_sequence_window(symbol, start, end, history)
        raw, ns, family_ids = block
        bounds = pl.Series([start, end]).dt.epoch('ns').to_torch()
        indices = []
        for family in range(len(self.families)):
            positions = torch.where(family_ids == family)[0]
            left, right = torch.searchsorted(ns[positions], bounds, right=True).tolist()
            indices.append(positions[max(0, left-self.history_per_family):right])
        selected = torch.cat(indices)
        dates, inverse = ns[selected].unique(sorted=True, return_inverse=True)
        length = max(self.window_length, len(dates))
        values = torch.full((length, len(self.columns)), float('nan'))
        padding = torch.ones(length, dtype=torch.bool)
        if len(dates):
            joined = raw[selected]
            rows, columns = torch.isfinite(joined).nonzero(as_tuple=True)
            values[inverse[rows] + length-len(dates), columns] = joined[rows, columns]
            padding[-len(dates):] = False
        return values, padding, dates, None

    def _uncached_family_sequence_window(self, symbol, start, end, history):
        parts,date_parts=[],[]
        for context in self.family_contexts:
            values,padding,dates,_=context.sequence_window(symbol,start,end,self.history_per_family)
            parts.append(values[~padding]);date_parts.append(dates)
        dates,inverse=torch.cat(date_parts).unique(sorted=True,return_inverse=True)
        length=max(self.window_length,len(dates))
        values=torch.full((length,len(self.columns)),float('nan'))
        padding=torch.ones(length,dtype=torch.bool)
        if len(dates):
            joined=torch.cat(parts)
            rows,columns=torch.isfinite(joined).nonzero(as_tuple=True)
            values[inverse[rows]+length-len(dates),columns]=joined[rows,columns]
            padding[-len(dates):]=False
        return values,padding,dates,None
