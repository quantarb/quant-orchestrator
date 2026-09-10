"""Bounded Polars parquet windows: never collect the complete feature corpus."""
from datetime import datetime, timezone

import polars as pl
import torch


class StreamingContext:
    def __init__(self, scan: pl.LazyFrame, columns: list[str], *, target_column=None):
        # Independent endpoint rows on one date must form one union row.
        # The query optimizer can push symbol/date predicates through these keys.
        self.scan = scan if target_column else scan.group_by("symbol", "date").agg(
            pl.col(column).drop_nulls().last() for column in columns
        )
        self.columns = columns
        self.target_column = target_column

    def version(self, symbol: str, anchor: datetime) -> int:
        result = self.scan.filter((pl.col("symbol") == symbol) & (pl.col("date") <= anchor)).select(
            pl.col("date").max().dt.epoch("ns")
        ).collect(engine="streaming").item()
        return -1 if result is None else int(result)

    def window(self, symbol: str, anchor: datetime, length: int):
        # Projection/predicate pushdown and bounded top-k avoid a full history
        # materialization. Only the requested model window crosses to Torch.
        frame = self.scan.filter((pl.col("symbol") == symbol) & (pl.col("date") <= anchor)).select(
            "date", *self.columns, *([self.target_column] if self.target_column else [])
        ).top_k(length, by=["date", *([self.target_column] if self.target_column else [])]).sort(["date", *([self.target_column] if self.target_column else [])]).collect(engine="streaming")
        values = torch.full((length, len(self.columns)), float("nan"), dtype=torch.float32)
        padding = torch.ones(length, dtype=torch.bool)
        if frame.height:
            values[-frame.height:] = frame.select(self.columns).cast(pl.Float32).fill_nan(None).fill_null(float("nan")).to_torch()
            padding[-frame.height:] = False
        dates = frame["date"].dt.epoch("ns").to_torch()
        return values, padding, dates, frame[self.target_column].to_torch().long() if self.target_column else None
