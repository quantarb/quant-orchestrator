"""Bounded per-symbol indexes over immutable, event-only supervised labels."""
from collections import OrderedDict
import polars as pl


class SupervisionIndex:
    def __init__(self, scan, *, max_bytes=128 * 1024**2):
        self.scan = scan
        self.max_bytes = max_bytes
        self.cache = OrderedDict()
        self.cache_bytes = 0

    def get(self, symbol, tasks):
        key = (symbol, tuple(tasks))
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        row_limit = self.max_bytes // (8 + 4 * len(tasks))
        frame = self.scan.filter(pl.col('symbol') == symbol).select(
            pl.col('date').cast(pl.Datetime('ns')), *tasks
        ).sort('date').head(row_limit + 1).collect(engine='streaming')
        if frame.height > row_limit:
            return None  # The caller retains the bounded date-window query.
        dates = frame['date'].dt.epoch('ns').to_torch()
        values = frame.select(pl.col(c).cast(pl.Float32).fill_nan(None).fill_null(float('nan')) for c in tasks).to_torch()
        if len(dates) > 1 and (dates[1:] == dates[:-1]).any():
            raise ValueError('Supervised labels require unique symbol/date keys')
        size = dates.numel() * dates.element_size() + values.numel() * values.element_size()
        while self.cache and self.cache_bytes + size > self.max_bytes:
            _, (old_dates, old_values) = self.cache.popitem(last=False)
            self.cache_bytes -= old_dates.numel() * old_dates.element_size() + old_values.numel() * old_values.element_size()
        self.cache[key] = (dates, values)
        self.cache_bytes += size
        return dates, values
