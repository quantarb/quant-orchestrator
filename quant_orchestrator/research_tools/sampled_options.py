"""Run-local, hindsight-filtered individual option contracts (never baskets)."""
from datetime import datetime, timedelta
import random

import polars as pl

from .frozen_option_adjustments import split_adjusted_members

SELECTION_POLICY = dict(same_expiration_year=True, positive_expiration_moneyness=True,
    positive_ask_to_bid_profit=True, minimum_quote_days=20, minimum_quote_coverage=.80,
    profit_quantile=.50, quantile_scope='year_universe_separately_by_call_put',
    contracts_per_side=5, seed=0, hindsight_selection=True, fixed_universe=True)


def sample_options_per_side(options, count=5, seed=0):
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError('Contracts per side must be a positive integer')
    if options.is_empty():
        return options.clone()
    groups = []
    for (symbol, right), group in options.group_by('underlying_symbol', 'option_type'):
        group = group.sort('contract_symbol')
        chosen = random.Random(f'{seed}:{symbol}:{right}').sample(range(len(group)), min(count, len(group)))
        groups.append(group[chosen])
    return pl.concat(groups, how='diagonal_relaxed').sort('underlying_symbol', 'contract_symbol')


def select_contracts(audit):
    """Apply stacked filters, then universe-wide per-right medians and sampling."""
    if audit.is_empty():
        return audit
    eligible = audit.filter(pl.col('moneyness').is_finite() & (pl.col('moneyness') > 0)
        & pl.col('profit_pct').is_finite() & (pl.col('profit_pct') > 0)
        & (pl.col('valid_quote_days') >= 20) & (pl.col('quote_coverage') >= .80))
    cutoffs = eligible.group_by('option_type').agg(pl.col('profit_pct').quantile(.50, interpolation='linear').alias('profit_cutoff_pct'))
    survivors = eligible.join(cutoffs, on='option_type').filter(pl.col('profit_pct') >= pl.col('profit_cutoff_pct'))
    return sample_options_per_side(survivors)


def contract_candidates(warehouse, symbol, year, end, read, calendar):
    """Read one issuer/year afresh; audit terminal outcomes and retain native quotes.

    Prices are expressed per original option unit across forward splits. Each
    document maps to one entry contract; no strike averaging or reweighting.
    """
    first = calendar.sessions_in_range(f'{year}-01-01', f'{year}-01-10')[0].to_pydatetime().replace(tzinfo=None)
    chain = read(first, first)
    if chain.is_empty():
        return pl.DataFrame(), pl.DataFrame(), 'missing_first_session_chain'
    members = chain.filter((pl.col('snapshot_date') == first) & (pl.col('expiration').dt.year() == year)
        & (pl.col('expiration') > first) & pl.col('option_type').is_in(['call', 'put'])).unique('contract_symbol')
    if members.is_empty():
        return members, pl.DataFrame(), 'no_same_year_contracts'
    members = members.select('underlying_symbol', 'contract_symbol', 'expiration', 'strike', 'option_type',
        pl.col('ask').alias('entry_ask')).with_columns(
        pl.col('contract_symbol').alias('document_symbol'), pl.lit(1.).alias('weight'),
        (pl.col('expiration')-first).dt.total_days().alias('dte'))
    settlements = {e: calendar.date_to_session(e.date().isoformat(), direction='previous').to_pydatetime().replace(tzinfo=None)
        for e in members['expiration'].unique()}
    members = members.with_columns(pl.col('expiration').replace_strict(settlements).alias('settlement'))
    # An incomplete contract cannot pass expiration-outcome filters.
    members = members.filter(pl.col('settlement') <= end)
    if members.is_empty():
        return members, pl.DataFrame(), 'no_completed_same_year_contracts'
    stop = min(end, members['settlement'].max())
    splits = warehouse.read_fundamentals(symbol, section='historical_splits', start=first.date().isoformat(), end=stop.date().isoformat())
    splits = [r for r in splits.sort('date').to_dicts() if first < r['date'] <= stop] if splits.height else []
    unsupported = None
    for r in splits:
        a,b = r.get('numerator'),r.get('denominator')
        ratio = a/b if a and b and b > 0 else float('nan')
        if not 1 <= ratio < float('inf'):
            unsupported = r['date']; break
    if unsupported:
        members = members.filter(pl.col('settlement') < unsupported)
        if members.is_empty():
            return members, pl.DataFrame(), 'unsupported_split'
        stop = members['settlement'].max()
    pieces, spots = [], []
    day = first
    while day <= stop:
        right = min(day+timedelta(days=90), stop)
        quotes = read(day, right)
        if quotes.height:
            spots.append(quotes.filter(pl.col('underlying_price').is_finite() & (pl.col('underlying_price')>0))
                .group_by('snapshot_date').agg(pl.col('underlying_price').median().alias('spot')))
            boundaries = [day, *[r['date'] for r in splits if day < r['date'] <= right], right+timedelta(days=1)]
            for left, next_day in zip(boundaries, boundaries[1:]):
                segment = quotes.filter(pl.col('snapshot_date').is_between(left, next_day, closed='left'))
                if segment.is_empty():
                    continue
                factor = 1.
                for split in splits:
                    if split['date'] <= left:
                        factor *= split['numerator']/split['denominator']
                mapped = split_adjusted_members(members, segment, factor)
                joined = segment.join(mapped.select('contract_symbol','document_symbol','weight'), on='contract_symbol')
                pieces.append(joined.select(pl.col('document_symbol').alias('symbol'), pl.col('snapshot_date').alias('date'),
                    (pl.col('bid')*pl.col('weight')).alias('low'), (pl.col('ask')*pl.col('weight')).alias('high'), 'volume'))
        day = right+timedelta(days=1)
    if not pieces:
        return members.head(0), pl.DataFrame(), 'missing_contract_histories'
    raw = pl.concat(pieces, how='diagonal_relaxed').unique(['symbol','date']).join(
        members.select(pl.col('contract_symbol').alias('symbol'),'settlement'), on='symbol').filter(pl.col('date') <= pl.col('settlement'))
    sessions = calendar.sessions_in_range(first, stop).tz_localize(None).to_pydatetime().tolist()
    valid = raw.filter(pl.col('date').is_in(sessions) & pl.col('low').is_finite() & pl.col('high').is_finite()
        & (pl.col('low')>0) & (pl.col('high')>0) & (pl.col('high')>=pl.col('low')))
    counts = valid.group_by('symbol').agg(pl.col('date').n_unique().alias('valid_quote_days'))
    last = raw.filter(pl.col('date')>first).sort('date').group_by('symbol').agg(
        pl.col('low').last().alias('exit_bid'), pl.col('date').last().alias('exit_date'))
    spot = pl.concat(spots).unique('snapshot_date')
    audit = members.join(last, left_on='contract_symbol', right_on='symbol', how='left').join(
        counts, left_on='contract_symbol', right_on='symbol', how='left').join(
        spot, left_on='settlement', right_on='snapshot_date', how='left')
    factors = {}
    expected = {}
    for d in audit['settlement'].unique():
        factor = 1.
        for r in splits:
            if r['date'] <= d:
                factor *= r['numerator']/r['denominator']
        factors[d] = factor
        expected[d] = len(calendar.sessions_in_range(first, d))
    audit = audit.with_columns(pl.col('settlement').replace_strict(factors,return_dtype=pl.Float64).alias('split_factor'),
        pl.col('settlement').replace_strict(expected,return_dtype=pl.Int64).alias('expected_quote_days'),
        pl.col('valid_quote_days').fill_null(0))
    audit = audit.with_columns(
        pl.when(pl.col('option_type')=='call').then(pl.col('spot')*pl.col('split_factor')-pl.col('strike'))
          .otherwise(pl.col('strike')-pl.col('spot')*pl.col('split_factor')).alias('moneyness'),
        pl.when(pl.col('entry_ask').is_finite() & (pl.col('entry_ask')>0) & pl.col('exit_bid').is_finite() & (pl.col('exit_bid')>=0))
          .then(100*(pl.col('exit_bid')/pl.col('entry_ask')-1)).otherwise(None).alias('profit_pct'),
        (pl.col('valid_quote_days')/pl.col('expected_quote_days')).alias('quote_coverage'))
    # Native bid/ask observations plus explicit intrinsic settlement, per contract.
    paths = valid.select('symbol','date','low','high','volume').with_columns(((pl.col('low')+pl.col('high'))/2).alias('close'))
    terminal = audit.filter(pl.col('moneyness').is_finite()).select(pl.col('contract_symbol').alias('symbol'),
        pl.col('settlement').alias('date'), pl.col('moneyness').clip(lower_bound=0).alias('close'), pl.lit(0.).alias('volume'))
    terminal = terminal.with_columns(pl.col('close').alias('low'),pl.col('close').alias('high'))
    paths = pl.concat([paths, terminal],how='diagonal_relaxed').unique(['symbol','date'],keep='last').with_columns(
        pl.col('close').alias('open')).sort('symbol','date')
    return audit, paths, 'audited'
