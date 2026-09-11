"""Disk-deduplicated, per-family NTP comparison with last-known-value persistence."""
import sqlite3
from collections import OrderedDict
import torch
import numpy as np
from .multirate_objectives import family_channels, reconstruction_targets, next_observation_targets


class NTPPersistenceAudit:
    def __init__(self, path, *, start_ns=None, end_ns=None):
        self.connection = sqlite3.connect(path)
        self.connection.execute('PRAGMA cache_size=-8192')
        self.connection.execute('CREATE TABLE IF NOT EXISTS pairs (symbol TEXT, rate TEXT, level TEXT, family TEXT, source INTEGER, target INTEGER, n INTEGER, model_error REAL, baseline_error REAL, PRIMARY KEY(symbol,rate,level,family,source,target))')
        self.start_ns, self.end_ns = start_ns, end_ns
        self.groups = set()
        self._recent_pairs = OrderedDict()
        self.duplicate_pairs_skipped = 0

    @torch.no_grad()
    def update(self, symbols, rate, names, widths, values, padding, dates, predictions):
        self.groups.update((rate, level, name) for level in ("subtoken", "token") for name in names)
        valid = torch.isfinite(values) & ~padding.unsqueeze(-1)
        rows = torch.arange(values.shape[1], device=values.device).view(1, -1, 1)
        last = torch.where(valid, rows, -1).cummax(1).values
        baseline = values.gather(1, last.clamp_min(0))
        baseline_valid = last >= 0
        targets = reconstruction_targets(values, padding, dates, torch.zeros_like(valid), widths)
        presence = family_channels(valid, widths).any(-1)
        future_family, _ = next_observation_targets(dates.unsqueeze(-1).expand_as(presence), presence, dates)
        future_token, _ = next_observation_targets(dates.unsqueeze(-1), valid.any(-1, keepdim=True), dates)
        inserts = []
        for level in ('subtoken', 'token'):
            target, target_valid = targets['next_' + level]
            prediction = predictions[f'next_{rate}_{level}']
            if level == 'token':
                target, target_valid, prediction = [family_channels(t, widths) for t in (target, target_valid, prediction)]
            keep = target_valid & family_channels(baseline_valid, widths)
            future = future_family if level == 'subtoken' else future_token.expand_as(future_family)
            if self.start_ns is not None:
                keep &= (future >= self.start_ns).unsqueeze(-1)
            if self.end_ns is not None:
                keep &= (future <= self.end_ns).unsqueeze(-1)
            counts = keep.sum(-1)
            model_error = torch.where(keep, (prediction.double() - target.double()).square(), 0).sum(-1)
            baseline_error = torch.where(keep, (family_channels(baseline, widths).double() - target.double()).square(), 0).sum(-1)
            if not torch.isfinite(model_error).all() or not torch.isfinite(baseline_error).all():
                raise ValueError('Nonfinite NTP evaluation errors')
            # Gather columns on-device, then transfer each column once. Python
            # scalar indexing previously repeated millions of tiny tensor operations.
            b, t, f = (counts > 0).nonzero(as_tuple=True)
            columns = [b, f, dates[b,t], future[b,t,f], counts[b,t,f],
                       model_error[b,t,f], baseline_error[b,t,f]]
            host_columns = [column.cpu().numpy() for column in columns]
            # Rolling windows repeat most source/target pairs within a batch.
            # Deduplicate in native array code before converting rows to Python;
            # return_index preserves the original first-observation semantics.
            symbol_ids = {symbol: index for index, symbol in enumerate(dict.fromkeys(symbols))}
            row_symbols = np.asarray([symbol_ids[symbol] for symbol in symbols], dtype=np.int64)
            keys = np.rec.fromarrays([row_symbols[host_columns[0]], *host_columns[1:4]])
            _, first = np.unique(keys, return_index=True)
            first.sort()
            self.duplicate_pairs_skipped += len(keys) - len(first)
            host_columns = [column[first].tolist() for column in host_columns]
            for bi, fi, source, target, count, model_sum, baseline_sum in zip(*host_columns):
                group = (symbols[bi], rate, level)
                if group not in self._recent_pairs:
                    self._recent_pairs[group] = set()
                    if len(self._recent_pairs) > 16:
                        self._recent_pairs.popitem(last=False)
                self._recent_pairs.move_to_end(group)
                seen = self._recent_pairs[group]
                key = (names[fi], source, target)
                if key in seen:
                    self.duplicate_pairs_skipped += 1
                    continue
                # SQLite remains authoritative after eviction. Forgetting a key
                # only costs another INSERT OR IGNORE; it never changes a result.
                if len(seen) >= 32768:
                    seen.clear()
                seen.add(key)
                inserts.append((symbols[bi], rate, level, names[fi], source, target, count, model_sum, baseline_sum))
        self.connection.executemany('INSERT OR IGNORE INTO pairs VALUES (?,?,?,?,?,?,?,?,?)', inserts)
        self.connection.commit()

    def report(self):
        result = []
        for rate, level, family, pairs, count, model, baseline, minimum_days, maximum_days in self.connection.execute(
            'SELECT rate,level,family,COUNT(*),SUM(n),SUM(model_error),SUM(baseline_error),MIN((target-source)/86400000000000.0),MAX((target-source)/86400000000000.0) FROM pairs GROUP BY rate,level,family'):
            result.append(dict(rate=rate, level=level, family=family, unique_pairs=pairs, values=count, minimum_horizon_days=minimum_days, maximum_horizon_days=maximum_days,
                model_mse=model/count, persistence_mse=baseline/count,
                skill=1-model/baseline if baseline > 0 else None,
                beats_persistence=model < baseline))
        measured = {(row['rate'], row['level'], row['family']) for row in result}
        for rate, level, family in sorted(self.groups - measured):
            result.append(dict(rate=rate, level=level, family=family, unique_pairs=0, values=0,
                minimum_horizon_days=None, maximum_horizon_days=None, model_mse=None,
                persistence_mse=None, skill=None, beats_persistence=None))
        result.sort(key=lambda row: (row['rate'], row['level'], row['family']))
        return dict(units='training-normalized values', weighting='individual observed values; identical model/baseline targets',
            deduplication='instrument/rate/level/family/source date/target date',
            missing_policy='exclude targets with no previously observed value for persistence',
            start_ns=self.start_ns, end_ns=self.end_ns, metrics=result)

    def close(self):
        self.connection.close()
