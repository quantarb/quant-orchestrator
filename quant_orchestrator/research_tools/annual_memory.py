"""Chronological annual documents and detached, instrument-local recurrent state."""
from collections import defaultdict, deque
from datetime import datetime, timedelta
import random
import torch

ANNUAL_CONTRACT = 'calendar_year_recurrent_v1'


def annual_window(index, symbol, start, end):
    # sequence_window owns (start, end]; include observations on January 1.
    values, padding, dates, labels = index.sequence_window(symbol, start-timedelta(microseconds=1), end, 0)
    start_ns = int((start-datetime(1970,1,1)).total_seconds()) * 1_000_000_000
    keep = dates >= start_ns
    real = values[~padding][keep]
    dates = dates[keep]
    length = max(2, len(real)+1)
    if length > 512:
        raise ValueError(f'Annual document exceeds capacity without dropping observations: {symbol} {start}: {length}')
    result = torch.full((length, values.shape[-1]), float('nan'))
    mask = torch.ones(length, dtype=torch.bool)
    timestamps = torch.full((length,), torch.iinfo(torch.long).max, dtype=torch.long)
    timestamps[0] = start_ns-1
    if len(real):
        result[1:len(real)+1] = real
        mask[1:len(real)+1] = False
        timestamps[1:len(real)+1] = dates
    return result, mask, dates, timestamps, labels


class AnnualCorpus:
    """Batch independent instruments while preserving each instrument's years."""
    def __init__(self, rows, batch_size=16, name='annual'):
        self.rows, self.batch_size, self.name = rows, batch_size, name
        if batch_size < 1:
            raise ValueError('batch_size must be positive')

    def __len__(self):
        return len(self.rows)

    def batches(self, *, seed=0, epoch=0):
        groups = defaultdict(list)
        for item in self.rows:
            groups[item['symbol']].append(item)
        symbols = sorted(groups)
        random.Random(seed+epoch).shuffle(symbols)
        pending = deque(deque(sorted(groups[s], key=lambda r:r['date'])) for s in symbols)
        active = []
        while pending or active:
            while pending and len(active) < self.batch_size:
                active.append(pending.popleft())
            yield [stream.popleft() for stream in active]
            active = [stream for stream in active if stream]

    def batch_count(self, *, seed=0, epoch=0):
        return sum(1 for _ in self.batches(seed=seed, epoch=epoch))


class AnnualMemory:
    """Detached state is checkpointable and cannot cross symbols or flow backward."""
    def __init__(self):
        self.epoch = None
        self.states = {}
        self.processed = 0

    def begin_epoch(self, epoch):
        if self.epoch != epoch:
            self.states = {}
            self.processed = 0
            self.epoch = epoch

    def inputs(self, batch, model):
        if len({r['symbol'] for r in batch}) != len(batch):
            raise ValueError('Recurrent batch must contain at most one year per instrument')
        result = {}
        for rate in (*model.config.rates, 'issuer_daily', 'issuer_sparse'):
            family_rate = rate.removeprefix('issuer_')
            families = len(model.coverage_inputs[family_rate].family_names)
            rows = []
            for item in batch:
                previous = self.states.get(item['symbol'])
                if previous and previous['date'] >= item['document_start']:
                    raise ValueError('Annual memory must precede the current document')
                payload = previous['rates'].get(rate) if previous else None
                if payload is None:
                    payload = {'states':torch.zeros(model.config.d_model),
                               'subtokens':torch.zeros(families, model.config.d_model),
                               'presence':torch.zeros(families, dtype=torch.bool)}
                rows.append(payload)
            device = next(model.parameters()).device
            result[rate] = {key:torch.stack([row[key] for row in rows]).to(device) for key in rows[0]}
        return result

    def update(self, batch, output):
        self.processed += len(batch)
        for i, item in enumerate(batch):
            self.states[item['symbol']] = {'date':item['date'], 'rates':{
                rate:{key:value[i].detach().cpu().clone() for key,value in payload.items()}
                for rate,payload in output['annual_memory'].items()}}

    def state_dict(self):
        return {'epoch':self.epoch, 'states':self.states, 'processed':self.processed}

    def load_state_dict(self, state):
        self.epoch, self.states = state['epoch'], state['states']
        self.processed = state['processed']
