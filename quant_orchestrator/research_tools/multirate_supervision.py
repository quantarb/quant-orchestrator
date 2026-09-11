"""Lazy event-only supervision; collect labels only for a requested instrument/date."""
from datetime import datetime

import polars as pl

from quant_orchestrator.platforms.ml_frameworks.torch.models.transformers.multirate.temporal_tasks import (
    HITS_SUPERVISED_TASK_NAMES, ORACLE_SUPERVISED_TASK_NAMES,
    FUND_ACTIVITY_SUPERVISED_TASK_NAMES, HOLDER_ACTIVITY_SUPERVISED_TASK_NAMES, TRADE_EVENT_SUPERVISED_TASK_NAMES,
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
                      *FUND_ACTIVITY_SUPERVISED_TASK_NAMES, *HOLDER_ACTIVITY_SUPERVISED_TASK_NAMES, *TRADE_EVENT_SUPERVISED_TASK_NAMES]
        for family, tasks in [('equity.strategy.hits_graph', HITS_SUPERVISED_TASK_NAMES),
                              ('equity.strategy.oracle_trades', ORACLE_SUPERVISED_TASK_NAMES)]:
            for task, channel in zip(tasks, channels):
                expressions.append(pl.when((pl.col('target_family') == family) & pl.col(channel).is_finite()).then(pl.col(channel)).alias(task))
        for prefix, tasks in [('fund_activity', FUND_ACTIVITY_SUPERVISED_TASK_NAMES), ('holder_activity', HOLDER_ACTIVITY_SUPERVISED_TASK_NAMES)]:
            for task in tasks:
                family = prefix + '.' + task.removeprefix(prefix + '_')
                # Activity magnitudes are observed inputs, not BCE probabilities.
                # Negatives come only from actual mirrored activity events.
                positive = task.removeprefix(prefix + '_')
                increasing = ('institutional_buy', 'add') if prefix == 'fund_activity' else ('buy', 'add')
                decreasing = ('reduce', 'exit')
                opposite = decreasing if positive in increasing else increasing if positive in decreasing else ()
                magnitude_valid = pl.col('signal_value').is_finite() & (pl.col('signal_value') > 0)
                expressions.append(pl.when((pl.col('target_family') == family) & magnitude_valid).then(1.)
                    .when(pl.col('target_family').is_in([prefix + '.' + name for name in opposite]) & magnitude_valid).then(0.)
                    .alias(task))
        # Labels describe the actual trade date; disclosure gates historical inputs separately.
        for family, prefix in [('equity.ownership.government_trades', 'government'),
                               ('equity.ownership.insider_trading', 'insider')]:
            selected = pl.col('target_family') == family
            buy = pl.col('signal_value') == 1
            sell = pl.col('text_0') == 1
            for side, positive, negative in [('buy', buy, sell), ('sell', sell, buy)]:
                expressions.append(pl.when(selected & positive).then(1.)
                    .when(selected & negative).then(0.).alias(prefix + '_is_' + side))
        self.scan = events.select('symbol', pl.col('event_date').alias('date'), *expressions).drop_nulls('date')
        self.scan = self.scan.filter(pl.any_horizontal(pl.col(task).is_not_null() for task in self.tasks))
        self.scan = self.scan.group_by('symbol', 'date').agg(pl.col(task).max() for task in self.tasks)
        activity_tasks=[*FUND_ACTIVITY_SUPERVISED_TASK_NAMES,*HOLDER_ACTIVITY_SUPERVISED_TASK_NAMES]
        ranges=self.scan.select(*[pl.col(t).min().alias(t+'_min') for t in activity_tasks],
                                *[pl.col(t).max().alias(t+'_max') for t in activity_tasks]).collect(engine='streaming').row(0,named=True)
        self.disabled_activity_tasks={t:'no observed positive/negative event pair in training range'
            for t in activity_tasks if ranges[t+'_min']!=0. or ranges[t+'_max']!=1.}
        if self.disabled_activity_tasks:
            self.scan=self.scan.with_columns(pl.lit(None,dtype=pl.Float64).alias(t) for t in self.disabled_activity_tasks)


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


SUPERVISED_ONLY_FAMILIES = frozenset({
    "equity.strategy.hits_graph", "equity.strategy.oracle_trades",
})


def input_event_families(events, families):
    """Keep outcome labels out of every historical input and reconstruction path."""
    names = [name for name in families if name not in SUPERVISED_ONLY_FAMILIES]
    return events.filter(~pl.col("target_family").is_in(list(SUPERVISED_ONLY_FAMILIES))), names or ["__empty_sparse_family__"]
