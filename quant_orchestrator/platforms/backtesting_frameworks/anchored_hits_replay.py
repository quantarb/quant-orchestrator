"""Streaming Polars replay of the anchored HITS percentile shared-book policy."""
from pathlib import Path
import json
import math
import polars as pl


def ranked_day(frame: pl.DataFrame, side: str) -> pl.DataFrame:
    """Average-tie percentile ranks, independently for hub and authority."""
    return frame.select('symbol', *[
        pl.when(pl.col(f'hits_{side}_return_{role}').is_finite())
        .then(pl.col(f'hits_{side}_return_{role}')).alias(role)
        for role in ('hub', 'authority')
    ]).with_columns(*[
        (pl.col(role).rank(method='average') / pl.col(role).count()).alias(role)
        for role in ('hub', 'authority')
    ])


def update_book(held, ranks, *, threshold, top_k):
    held = set(held)
    events = []
    for symbol in sorted(held):
        if symbol in ranks and ranks[symbol]['authority'] is not None and ranks[symbol]['authority'] > threshold:
            held.remove(symbol)
            events.append((symbol, 'exit', ranks[symbol]['authority']))
    candidates = [(r['hub'], s) for s, r in ranks.items()
                  if s not in held and r['hub'] is not None and r['hub'] > threshold]
    for score, symbol in sorted(candidates, key=lambda x: (-x[0], x[1]))[:max(0, top_k-len(held))]:
        held.add(symbol)
        events.append((symbol, 'enter', score))
    return held, events


def replay_anchored_hits(predictions, prices, output, *, start, end, side,
                         threshold=.8, top_k=20, cost_bps=5.5, initial_cash=100000.):
    """EOD decisions execute the following close; held weights earn subsequent returns.

    Preserve the former shared-book fixed signed 1/min(top_k, universe size) weights and its cost
    convention (cost on changes in target weights). No forced terminal exit.
    Inputs are lazy frames of equity scores and split/dividend-adjusted closes.
    """
    if side not in ('long', 'short') or top_k < 1 or not 0 <= threshold <= 1:
        raise ValueError('Invalid side, capacity, or percentile threshold')
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    date_filter = pl.col('date').cast(pl.Date).is_between(pl.lit(start).str.to_date(), pl.lit(end).str.to_date())
    prices = prices.filter(date_filter).with_columns(pl.col('date').cast(pl.Date))
    scores = predictions.filter(date_filter).with_columns(pl.col('date').cast(pl.Date))
    dates = prices.select('date').unique().sort('date').collect(engine='streaming')['date'].to_list()
    symbols = scores.select('symbol').unique().collect(engine='streaming')['symbol'].to_list()
    if not symbols:
        raise ValueError('Empty scoring universe')
    allocation_slots = min(top_k, len(symbols))
    held, previous_ranks, last_quotes = set(), {}, {}
    nav = initial_cash; peak = initial_cash; curve = []; actions = []; weights = []
    sign = 1 if side == 'long' else -1
    for date in dates:
        quotes = {r['symbol']: r['close'] for r in prices.filter(pl.col('date') == date).collect(engine='streaming').iter_rows(named=True)}
        gross = 0.
        for symbol in held:
            if symbol not in quotes or symbol not in last_quotes or not math.isfinite(quotes[symbol]) or quotes[symbol] <= 0:
                raise ValueError(f'Missing/invalid held equity quote: {symbol} {date}')
            gross += sign / allocation_slots * (quotes[symbol] / last_quotes[symbol] - 1)
        updated, events = update_book(held, previous_ranks, threshold=threshold, top_k=top_k)
        if not updated.issubset(quotes):
            raise ValueError(f'Missing entry quote on {date}')
        turnover = len(held.symmetric_difference(updated)) / allocation_slots
        net = gross - turnover * cost_bps / 10000
        nav *= 1 + net; peak = max(peak, nav)
        for symbol, action, rank in events:
            actions.append(dict(date=date, symbol=symbol, action=f'{action}_{side}', rank=rank))
        held = updated
        for symbol in sorted(held):
            weights.append(dict(date=date, symbol=symbol, weight=sign/allocation_slots))
        curve.append(dict(date=date, equity=nav, return_=net, drawdown=nav/peak-1,
                          positions=len(held), gross_exposure=len(held)/allocation_slots, turnover=turnover))
        day = scores.filter(pl.col('date') == date).collect(engine='streaming')
        if day['symbol'].n_unique() != day.height:
            raise ValueError('Duplicate symbol/date predictions')
        previous_ranks = {r['symbol']: r for r in ranked_day(day, side).iter_rows(named=True)}
        last_quotes = quotes
    if not curve:
        raise ValueError('Empty replay calendar')
    equity = pl.DataFrame(curve); equity.write_parquet(output/'equity_curve.parquet')
    pl.DataFrame(actions, schema={'date':pl.Date,'symbol':pl.String,'action':pl.String,'rank':pl.Float64}).write_parquet(output/'action_tape.parquet')
    pl.DataFrame(weights, schema={'date':pl.Date,'symbol':pl.String,'weight':pl.Float64}).write_parquet(output/'weights.parquet')
    result = dict(side=side, start=start, end=end, initial_cash=initial_cash, final_equity=nav,
        total_return=nav/initial_cash-1, max_drawdown=equity['drawdown'].min(),
        entries=sum(a['action'].startswith('enter') for a in actions),
        exits=sum(a['action'].startswith('exit') for a in actions),
        open_positions=len(held), mean_gross_exposure=equity['gross_exposure'].mean(),
        threshold=threshold,top_k=top_k,allocation_slots=allocation_slots,universe_symbols=len(symbols),cost_bps=cost_bps,
        timing='EOD ranks execute next session close; subsequent close-to-close returns',
        prices='splits_and_dividends', model='fixed pre-2024 v6 checkpoint',
        limitations=['Equity-only; no option baskets', 'No borrow fees or locate constraints for shorts',
                     'Fixed target-weight return accounting matches older shared-book convention',
                     'Costs apply to target-weight changes, not drift rebalancing',
                     'No forced terminal liquidation; open positions marked to final close',
                     'Retrospective universe; no annual model retraining'])
    (output/'summary.json').write_text(json.dumps(result,indent=2))
    return result
