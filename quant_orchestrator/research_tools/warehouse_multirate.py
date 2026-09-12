"""Annual documents assembled when requested, directly from warehouse sources.

The on-disk schema contains field names only. No corpus, fitted normalization,
old roster, or feature/event export is an input to this loader.
"""
from collections import OrderedDict, deque
from datetime import datetime, timedelta
import json
from pathlib import Path
import random

import polars as pl
import torch

from quant_warehouse import Warehouse
from quant_warehouse.platforms.data_providers.fmp.feature_engineering.broad_observations import (
    build_issuer_families, calendar_features, numeric, select_fields, group_contexts,
)
from quant_warehouse.platforms.data_providers.fmp.feature_engineering.event_observations import issuer_event_observations
from quant_warehouse.platforms.data_providers.thetadata.options import (
    read_option_chain_arctic, read_thetadata_eod_option_chain,
    OPTIONS_THETADATA_EOD_LIBRARY, OPTIONS_THETADATA_PROVIDER,
)
from quant_warehouse.warehouse.storage import provider_library
from quant_warehouse.warehouse.sections import DEFAULT_ECONOMIC_SERIES
from .frozen_option_adjustments import first_session_baskets, basket_quotes, split_adjusted_members
from .multirate_corpus import statement_fields
from .multirate_targets import materialize_instrument_targets, VALUE_COLUMNS
from .multirate_supervision import StreamingSupervision
from .multirate_objectives import feature_family_layout
from quant_orchestrator.platforms.ml_frameworks.torch.models.transformers.multirate.temporal_tasks import SUPERVISED_TARGET_TASK_NAMES

SPARSE_FAMILIES = (
    'equity.calendar.dividend', 'equity.calendar.splits', 'equity.estimates.price_target',
    'equity.ownership.government_trades', 'equity.ownership.insider_trading',
    'fund_activity.add', 'fund_activity.institutional_buy', 'fund_activity.reduce', 'fund_activity.exit',
)


def annual_tensor(frame, columns, start):
    """Reserve one memory token and preserve every native observation."""
    frame = frame.sort('date')
    size = max(2, frame.height + 1)
    if size > 512:
        raise ValueError(f'Annual stream exceeds 512 positions: {size}; refusing to truncate')
    values = torch.full((size, len(columns)), float('nan'))
    padding = torch.ones(size, dtype=torch.bool)
    dates = torch.full((size,), torch.iinfo(torch.long).max, dtype=torch.long)
    dates[0] = int((start - datetime(1970, 1, 1)).total_seconds()) * 10**9 - 1
    if frame.height:
        raw = frame.select(columns).cast(pl.Float32).fill_null(float('nan')).to_torch()
        raw = torch.where(torch.isfinite(raw), raw, float('nan'))
        # Fixed, checkpointed transformation: no global data-fitting pass and
        # no changing running statistics between training and inference.
        values[1:frame.height+1] = raw.sign() * raw.abs().log1p() / 10
        padding[1:frame.height+1] = False
        dates[1:frame.height+1] = frame['date'].cast(pl.Datetime('ns')).dt.epoch('ns').to_torch()
    return values, padding, dates


def merge_observations(frames, columns):
    if not frames:
        return pl.DataFrame(schema={'date': pl.Datetime('ns'), **dict.fromkeys(columns, pl.Float32)})
    frame = pl.concat([f.with_columns(pl.col('date').cast(pl.Datetime('ns'))) for f in frames], how='diagonal_relaxed')
    frame = frame.group_by('date').agg(pl.all().drop_nulls().last()).sort('date')
    return frame.with_columns(*[pl.lit(None, dtype=pl.Float32).alias(c) for c in columns if c not in frame.columns]).select('date', *columns)


class WarehouseAnnualStream:
    def __init__(self, *, min_market_cap, start, end, cutoff, output, warehouse=None):
        self.warehouse = warehouse or Warehouse()
        self.start, self.end, self.cutoff = map(datetime.fromisoformat, (start, end, cutoff))
        self.output = Path(output)
        schema = json.loads(Path(__file__).with_name('multirate_feature_schema.json').read_text())
        self.features = schema['features']
        self.layout = feature_family_layout(self.features)
        self.columns = ['value__' + f for f in self.features]
        self.feature_set = set(self.features)
        self.sources = OrderedDict()
        self.option_cache = OrderedDict()
        self.coverage, self.observed = {}, {}
        profiles = self.warehouse.catalog.query_symbol_profiles(provider='fmp', min_market_cap=min_market_cap,
            country='US', exchanges=['NASDAQ', 'NYSE'], exclude_etf=True, exclude_fund=True)
        self.prices, self.profiles, excluded = {}, {}, []
        for profile in profiles:
            frame = self.warehouse.read_prices(profile.symbol, provider='fmp', start=start, end=end)
            if frame.is_empty() or frame.filter(pl.col('date') < self.cutoff).height < 2:
                excluded.append(dict(symbol=profile.symbol, reason='fewer than two pre-cutoff equity observations'))
                continue
            self.prices[profile.symbol] = frame.with_columns(pl.col('date').cast(pl.Datetime('ns')))
            self.profiles[profile.symbol] = profile
        if not self.prices:
            raise ValueError('The requested market-cap universe has no training price history')
        stored = set(self.warehouse.backend.list_symbols(provider_library(OPTIONS_THETADATA_EOD_LIBRARY, OPTIONS_THETADATA_PROVIDER)))
        self.option_years = {}
        for symbol in sorted(self.prices):
            dates = read_option_chain_arctic(symbol, start_date=start, end_date=end,
                columns=['snapshot_date'], backend=self.warehouse.backend) if symbol in stored else pl.DataFrame()
            counts = (dates.group_by(pl.col('snapshot_date').dt.year().alias('year'))
                .agg(pl.col('snapshot_date').n_unique().alias('dates')).sort('year').to_dicts()) if dates.height else []
            self.option_years[symbol] = [r['year'] for r in counts]
            self.coverage[symbol] = dict(stored_option_years=counts, cohorts=[])
        self.expected_option_symbols = {s for s, years in self.option_years.items() if any(y < self.cutoff.year for y in years)}
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output/'universe.json').write_text(json.dumps(dict(min_market_cap=min_market_cap,
            equity_symbols=sorted(self.prices), option_symbols=sorted(self.expected_option_symbols), excluded=excluded), indent=2))
        self.macro = None
        self.contexts = None
        self.write_coverage()

    def write_coverage(self):
        (self.output/'option_coverage.json').write_text(json.dumps(dict(source='warehouse',
            construction='on_demand', coverage=self.coverage, trained=self.observed), indent=2, default=str))

    def issuer(self, symbol):
        return {'GOOG': 'GOOGL', 'BRK-A': 'BRK-B'}.get(symbol, symbol)

    def source(self, symbol):
        issuer = self.issuer(symbol)
        if issuer in self.sources:
            self.sources.move_to_end(issuer)
            return self.sources[issuer]
        w = self.warehouse
        start, end = self.start.date().isoformat(), self.end.date().isoformat()
        rates = {r: [] for r in ('annual', 'quarterly', 'daily')}
        for family, rate, frame, _ in build_issuer_families(w, issuer, start=start, end=end):
            fields = numeric(frame)
            if fields:
                unknown = {family+'.'+c for c in fields} - self.feature_set
                if unknown:
                    raise ValueError(f'Warehouse fields outside the model schema: {sorted(unknown)}')
                rates[rate].append(select_fields(frame, fields).rename({c: 'value__'+family+'.'+c for c in fields}))
        for period, rate in [('annual','annual'), ('quarter','quarterly')]:
            for section in ('income','balance','cash'):
                frame = w.read_fundamentals(issuer, provider='fmp', section=section, period=period, start=start, end=end)
                if frame.is_empty():
                    continue
                dc = 'period_ending' if 'period_ending' in frame.columns else 'date'
                fields = statement_fields(frame)
                unknown = {'fmp.'+section+'.'+c for c in fields} - self.feature_set
                if unknown:
                    raise ValueError(f'Warehouse statement fields outside the model schema: {sorted(unknown)}')
                rates[rate].append(frame.select(pl.col(dc).alias('date'), *[pl.col(c).cast(pl.Float32).alias('value__fmp.'+section+'.'+c) for c in fields]))
        events = [frame for _, frame in issuer_event_observations(w, issuer, start=start, end=end) if frame.height]
        from quant_warehouse.research_tools.fund_activity import build_institutional_activity_events
        raw = w.read_fundamentals(issuer, section='ownership_institutional', start=start, end=end)
        if raw.height:
            event = build_institutional_activity_events(raw.with_columns(pl.lit(issuer).alias('symbol')))
            if event.height:
                events.append(event.select('symbol','date','event_date','target_family',pl.col('signal_value').cast(pl.Float32),
                    *[pl.lit(None,dtype=pl.Float32).alias(c) for c in VALUE_COLUMNS[1:]]))
        sparse = pl.concat(events, how='diagonal_relaxed') if events else pl.DataFrame(schema={
            'symbol':pl.String,'date':pl.Datetime('ns'),'event_date':pl.Datetime('ns'),'target_family':pl.String,
            **dict.fromkeys(VALUE_COLUMNS,pl.Float32)})
        result = rates, sparse.with_columns(pl.col('date','event_date').cast(pl.Datetime('ns')))
        self.sources[issuer] = result
        # Bounded source cache, independent of the number of annual documents.
        while len(self.sources) > 16:
            self.sources.popitem(last=False)
        return result

    def common(self):
        if self.macro is None:
            self.macro = []
            for family, codes in [('economic_indicators', DEFAULT_ECONOMIC_SERIES),
                                  ('treasury_rates', self.warehouse.macro.list_treasury_series_codes())]:
                frame = self.warehouse.read_macro_panel(codes, start=self.start.date().isoformat(), end=self.end.date().isoformat())
                fields = numeric(frame)
                if fields:
                    self.macro.append(select_fields(frame, fields).rename({c:'value__'+family+'.'+c for c in fields}))
        return self.macro

    def peer_context(self, symbol):
        if self.contexts is None:
            symbols=sorted({self.issuer(s) for s in self.prices})
            panel=pl.concat([self.prices[s].select('date','close').with_columns(pl.lit(s).alias('symbol')) for s in symbols])
            pe=[]
            names=['income__net_income','income__ebit','income__ebitda','income__revenue',
                   'cash__free_cash_flow','balance__total_equity_non_controlling_interests']
            prefix='value__fmp_daily_mcap_multiple.'
            for s in symbols:
                rates,_=self.source(s)
                for f in rates['daily']:
                    fields=[c for c in names if prefix+c in f.columns]
                    if fields:
                        pe.append(f.select('date',pl.lit(s).alias('symbol'),*[pl.col(prefix+c).alias('pe_'+c) for c in fields]))
            if not pe:raise ValueError('Missing source observations for peer valuation context')
            panel=panel.lazy().sort('symbol','date').join_asof(pl.concat(pe,how='diagonal_relaxed').lazy().sort('symbol','date'),
                on='date',by='symbol',strategy='backward')
            groups=pl.DataFrame([dict(symbol=s,sector=self.profiles[s].sector or 'Unknown',industry=self.profiles[s].industry or 'Unknown') for s in symbols])
            self.contexts=list(group_contexts(panel,groups))
        profile=self.profiles[symbol]
        result=[]
        for family,key,frame in self.contexts:
            local=frame.filter(pl.col(key)==(getattr(profile,key,None) or 'Unknown')).drop(key)
            result.append(local.rename({c:'value__'+family+'.'+c for c in numeric(local)}))
        return result

    def cohorts(self, symbol, year):
        key = symbol, year
        if key in self.option_cache:
            self.option_cache.move_to_end(key)
            return self.option_cache[key]
        import exchange_calendars as xcals
        calendar = xcals.get_calendar('XNYS', start='1990-01-01', end='2030-12-31')
        first = calendar.sessions_in_range(f'{year}-01-01', f'{year}-01-10')[0].to_pydatetime().replace(tzinfo=None)
        read = lambda a,b: read_thetadata_eod_option_chain(symbol, start_date=a, end_date=b, backend=self.warehouse.backend)
        chain = read(first,first)
        record = dict(year=year, first_session=str(first), status='loading')
        self.coverage[symbol]['cohorts'].append(record)
        if chain.is_empty():
            record['status'] = 'missing_first_session_chain'
            self.write_coverage()
            return None
        members = first_session_baskets(chain, first)
        audit=self.output/'cohort_members'
        audit.mkdir(exist_ok=True)
        members.write_parquet(audit/f'{symbol}_{year}.parquet')
        splits=self.warehouse.read_fundamentals(symbol,section='historical_splits',
            start=first.date().isoformat(),end=min(datetime(year,12,31),self.end).date().isoformat())
        split_rows=splits.sort('date').to_dicts() if splits.height else []
        if any(r['numerator']/r['denominator']<1 for r in split_rows):
            raise ValueError(f'{symbol}/{year}: reverse-split deliverables require an explicit option adjustment mapping')
        record['splits']=[dict(date=str(r['date']),ratio=r['numerator']/r['denominator']) for r in split_rows]
        parts, spots = [], []
        day, end = first, min(datetime(year,12,31),self.end)
        while day <= end:
            stop = min(day+timedelta(days=90),end)
            quotes = read(day,stop)
            if quotes.height:
                boundaries=[day,*[r['date'] for r in split_rows if day<r['date']<=stop],stop+timedelta(days=1)]
                for left,right in zip(boundaries,boundaries[1:]):
                    segment=quotes.filter(pl.col('snapshot_date').is_between(left,right,closed='left'))
                    factor=1.
                    for split in split_rows:
                        if split['date']<=left:factor*=split['numerator']/split['denominator']
                    if segment.height:
                        adjusted=split_adjusted_members(members,segment,factor)
                        parts.append(basket_quotes(segment,adjusted))
                spots.append(quotes.filter(pl.col('underlying_price').is_finite() & (pl.col('underlying_price')>0))
                    .group_by('snapshot_date').agg(pl.col('underlying_price').median().alias('spot')))
            day = stop+timedelta(days=1)
        paths = pl.concat(parts,how='diagonal_relaxed') if parts else pl.DataFrame()
        spot=pl.concat(spots).unique('snapshot_date') if spots else pl.DataFrame()
        terminal=[]
        for (identity,), group in members.group_by('document_symbol'):
            expiration=group['expiration'][0]
            settlement=calendar.date_to_session(expiration.date().isoformat(),direction='previous').to_pydatetime().replace(tzinfo=None)
            if settlement>end:continue
            price=spot.filter(pl.col('snapshot_date')==settlement)
            if price.is_empty():
                # Old source history may have only monthly quotes. Preserve
                # the missing settlement rather than inventing a payoff.
                record.setdefault('missing_settlement',[]).append(identity)
                continue
            underlying=float(price['spot'][0])
            factor=1.
            for split in split_rows:
                if split['date']<=settlement:factor*=split['numerator']/split['denominator']
            # Express payoff per original basket constituent, conserving
            # economic exposure through splits and avoiding adjusted equities
            # mixed with unadjusted option strikes.
            intrinsic=((pl.lit(underlying*factor)-pl.col('strike')) if group['option_type'][0]=='call' else
                       (pl.col('strike')-pl.lit(underlying*factor))).clip(lower_bound=0)
            value=float(group.select((intrinsic*pl.col('weight')).sum()).item())
            terminal.append(dict(symbol=identity,date=settlement,open=value,high=value,low=value,close=value,volume=0.,
                observed=group.height,expected=group.height))
        if terminal:
            terminal=pl.DataFrame(terminal).with_columns(pl.col('date').cast(paths.schema['date']))
            paths=pl.concat([paths,terminal],how='diagonal_relaxed').unique(['symbol','date'],keep='last').sort('symbol','date')
        actual = set(paths['symbol']) if paths.height else set()
        expected = set(members['document_symbol'])
        if actual != expected:
            raise ValueError(f'{symbol}/{year}: frozen baskets without any complete quote: {sorted(expected-actual)}')
        record.update(status='loaded', baskets=len(expected), rows=paths.height,
            temporal_baskets=paths.group_by('symbol').len().filter(pl.col('len')>1).height)
        result = members, paths
        self.option_cache[key] = result
        while len(self.option_cache) > 3:
            self.option_cache.popitem(last=False)
        self.write_coverage()
        return result

    def sample(self, symbol, year, *, option_symbol=None, members=None, prices=None, training=True, start=None, end=None):
        first = max(datetime(year,1,1),start or self.start)
        last = min(datetime(year,12,31),end or self.end)
        identity = option_symbol or symbol
        asset = 'option' if option_symbol else 'equity'
        prices = prices if prices is not None else self.prices[symbol]
        prices = prices.filter(pl.col('date').is_between(first,last)).sort('date')
        rates, events = self.source(symbol)
        streams = {r:[f.filter(pl.col('date').is_between(first,last)) for f in fs] for r,fs in rates.items()}
        for f in [*self.common(),*self.peer_context(symbol)]:
            streams['daily'].append(f.filter(pl.col('date').is_between(first,last)))
        streams['daily'].append(calendar_features(prices.select('date')).rename({c:'value__time_calendar.'+c for c in ('weekday','month','day_of_year')}))
        equity_daily = streams['daily'] + [self.prices[symbol].filter(pl.col('date').is_between(first,last)).select('date',
            *[pl.col(c).alias('value__price.'+c) for c in ('open','high','low','close','volume')])]
        streams['daily'] = [prices.select('date',*[pl.col(c).alias('value__price.'+c) for c in ('open','high','low','close','volume')])] if option_symbol else equity_daily
        if option_symbol:
            subset = members.filter(pl.col('document_symbol')==option_symbol)
            terms = {'instrument.strike':float((subset['strike']*subset['weight']).sum()),
                'instrument.entry_dte':float(subset['dte'][0]),'instrument.contract_size':100.,
                'instrument.option_type':float(subset['option_type'][0]=='call'),
                'instrument.expiration':float((subset['expiration'][0]-datetime(1970,1,1)).days)}
            streams['daily'].append(prices.select('date',pl.col('low').alias('value__options.bid'),
                pl.col('high').alias('value__options.ask'),pl.col('close').alias('value__options.midpoint'),pl.col('volume').alias('value__options.volume'),
                *[pl.lit(v).alias('value__'+k) for k,v in terms.items()]))
        filtered_events = events.filter(pl.col('date').is_between(first,last))
        sparse_columns = [f'{f}:{c}' for f in SPARSE_FAMILIES for c in VALUE_COLUMNS]
        sparse_parts=[]
        for f in SPARSE_FAMILIES:
            part=filtered_events.filter(pl.col('target_family')==f)
            if part.height:
                sparse_parts.append(part.select('date',*[pl.col(c).alias(f'{f}:{c}') for c in VALUE_COLUMNS]))
        sparse_frame=merge_observations(sparse_parts,sparse_columns)
        sample=dict(symbol=identity, issuer=self.issuer(symbol), underlying_symbol=symbol, asset_class=asset,
            date=last.date().isoformat(),document_start=first.date().isoformat(),sequence_mode='annual_memory',
            issuer_context_key=(symbol,year),annual_context_key=(symbol,year),quarterly_context_key=(symbol,year))
        for rate in ('annual','quarterly','daily'):
            frame=merge_observations(streams[rate],self.columns)
            values,padding,dates=annual_tensor(frame,self.columns,first)
            sample.update({rate:values,rate+'_padding':padding,rate+'_timestamps':dates})
            if rate=='daily':
                daily_frame=frame
        for rate,frame,columns in [('sparse',sparse_frame,sparse_columns),
                ('issuer_sparse',sparse_frame,sparse_columns),('issuer_daily',merge_observations(equity_daily,self.columns),self.columns)]:
            values,padding,dates=annual_tensor(frame,columns,first)
            sample.update({rate:values,rate+'_padding':padding,rate+'_timestamps':dates})
        length=len(sample['daily'])
        sample['supervised_targets']=torch.zeros(length,len(SUPERVISED_TARGET_TASK_NAMES))
        sample['supervised_valid']=torch.zeros(length,len(SUPERVISED_TARGET_TASK_NAMES),dtype=torch.bool)
        if training:
            labels=materialize_instrument_targets(identity,prices)
            if not option_symbol:
                labels=pl.concat([labels,events],how='diagonal_relaxed')
            supervision=StreamingSupervision(labels.lazy(),cutoff=self.cutoff).scan.collect(engine='streaming')
            target=daily_frame.select('date').join(supervision.filter(pl.col('symbol')==identity).drop('symbol'),on='date',how='left')
            values=target.select(SUPERVISED_TARGET_TASK_NAMES).fill_null(float('nan')).cast(pl.Float32).to_torch()
            sample['supervised_targets'][1:len(values)+1]=torch.nan_to_num(values)
            sample['supervised_valid'][1:len(values)+1]=torch.isfinite(values)
        sample['daily_dates']=daily_frame['date'].to_list()
        sample['daily_score_valid']=daily_frame['value__price.close'].is_finite().fill_null(False).to_list()
        sample['prices']=prices
        return sample

    def documents(self, *, training=True, seed=0, start=None, end=None):
        groups=[]
        for symbol in sorted(self.prices):
            years=sorted(self.prices[symbol]['date'].dt.year().unique().to_list())
            years=[y for y in years if (y < self.cutoff.year if training else (start.year <= y <= end.year))]
            def equity_stream(s=symbol,ys=years):
                for y in ys:
                    yield self.sample(s,y,training=training,start=start,end=end)
            def option_stream(s=symbol):
                for y in self.option_years[s]:
                    if not (y < self.cutoff.year if training else start.year <= y <= end.year):continue
                    cohort=self.cohorts(s,y)
                    if cohort is None:continue
                    members,paths=cohort
                    for identity in sorted(set(members['document_symbol'])):
                        yield self.sample(s,y,option_symbol=identity,members=members,
                            prices=paths.filter(pl.col('symbol')==identity),training=training,start=start,end=end)
            groups.extend([equity_stream(),option_stream()])
        random.Random(seed).shuffle(groups)
        pending=deque(groups)
        # A few independent streams keep source memory bounded and start the
        # optimizer without constructing documents for the remaining universe.
        active=[]
        while pending or active:
            while pending and len(active)<16:active.append(pending.popleft())
            for stream in list(active):
                try:yield next(stream)
                except StopIteration:active.remove(stream)
