"""Lazy event-only supervision; collect labels only for a requested instrument/date."""
from datetime import datetime

import polars as pl

from quant_orchestrator.platforms.ml_frameworks.torch.models.transformers.multirate.temporal_tasks import (
    HITS_SUPERVISED_TASK_NAMES, ORACLE_SUPERVISED_TASK_NAMES,
    FUND_ACTIVITY_SUPERVISED_TASK_NAMES, HOLDER_ACTIVITY_SUPERVISED_TASK_NAMES,
)


def instrument_asset_groups(taxonomy: pl.DataFrame) -> dict[str, set[str]]:
    """Use explicit security types; issuer linkage does not identify an option."""
    required = {'symbol', 'asset_class'}
    if required - set(taxonomy.columns):
        raise ValueError('Instrument taxonomy requires symbol and asset_class columns')
    groups: dict[str, set[str]] = {}
    seen = set()
    for symbol, asset in taxonomy.select('symbol', 'asset_class').iter_rows():
        if not symbol or not asset or asset != asset.strip().lower():
            raise ValueError('Instrument taxonomy requires nonempty symbols and lowercase asset_class values')
        if symbol in seen:
            raise ValueError(f'Duplicate instrument taxonomy symbol: {symbol}')
        seen.add(symbol)
        groups.setdefault(asset, set()).add(symbol)
    return groups


class StreamingSupervision:
    def __init__(self, events: pl.LazyFrame, *, cutoff: datetime | None = None):
        if cutoff is not None:
            events = events.filter((pl.col('date') < cutoff) & (pl.col('event_date') < cutoff))
        channels = ['signal_value', *[f'text_{i}' for i in range(7)]]
        expressions = []
        self.tasks = [*HITS_SUPERVISED_TASK_NAMES, *ORACLE_SUPERVISED_TASK_NAMES,
                      *FUND_ACTIVITY_SUPERVISED_TASK_NAMES, *HOLDER_ACTIVITY_SUPERVISED_TASK_NAMES]
        for family, tasks in [('equity.strategy.hits_graph', HITS_SUPERVISED_TASK_NAMES),
                              ('equity.strategy.oracle_trades', ORACLE_SUPERVISED_TASK_NAMES)]:
            for task, channel in zip(tasks, channels):
                expressions.append(pl.when((pl.col('target_family') == family) & pl.col(channel).is_finite()).then(pl.col(channel)).alias(task))
        for prefix, tasks in [('fund_activity', FUND_ACTIVITY_SUPERVISED_TASK_NAMES), ('holder_activity', HOLDER_ACTIVITY_SUPERVISED_TASK_NAMES)]:
            for task in tasks:
                family = prefix + '.' + task.removeprefix(prefix + '_')
                expressions.append(pl.when((pl.col('target_family') == family) & pl.col('signal_value').is_finite()).then(pl.col('signal_value')).alias(task))
        self.scan = events.select('symbol', pl.col('event_date').alias('date'), *expressions).drop_nulls('date')
        self.scan = self.scan.filter(pl.any_horizontal(pl.col(task).is_not_null() for task in self.tasks))
        self.scan = self.scan.group_by('symbol', 'date').agg(pl.col(task).max() for task in self.tasks)

    def anchors(self):
        return self.scan.select('symbol', 'date').collect(engine='streaming')

    def get(self, key, default=None):
        symbol, date = key
        rows = self.scan.filter((pl.col('symbol') == symbol) & (pl.col('date') == date)).collect(engine='streaming')
        if rows.is_empty():
            return {} if default is None else default
        return {task: rows[task][0] for task in self.tasks if rows[task][0] is not None}

    def coverage(self, *, symbols_by_asset, required_tasks):
        counts = {}
        for asset, symbols in sorted(symbols_by_asset.items()):
            if not symbols:
                continue
            row = self.scan.filter(pl.col('symbol').is_in(list(symbols))).select(pl.col(task).count() for task in required_tasks).collect(engine='streaming').row(0, named=True)
            missing = [task for task, count in row.items() if not count]
            if missing:
                raise ValueError(f'Missing supervised training labels for {asset}: {", ".join(missing)}')
            counts[asset] = row
        return counts
