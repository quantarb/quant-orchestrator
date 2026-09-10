"""Fixed-set NTP trends and immutable snapshots for a running training process."""
import json
import re
from pathlib import Path


def option(command, flag):
    return command[command.index(flag) + 1] if flag in command else None


def evaluation_command(command, checkpoint, output, start, end, max_samples):
    command = list(command)
    for flag in ('--skip-predictions', '--inference-only'):
        if flag in command:
            command.remove(flag)
    for flag, value in {'--output-dir': str(output), '--checkpoint': str(checkpoint),
                        '--prediction-start-date': start, '--prediction-end-date': end,
                        '--max-samples': str(max_samples)}.items():
        if flag in command:
            command[command.index(flag) + 1] = value
        else:
            command += [flag, value]
    return command + ['--inference-only']


def last_epoch_batches(log):
    with Path(log).open('rb') as handle:
        handle.seek(0, 2)
        handle.seek(max(0, handle.tell() - 65536))
        tail = handle.read().decode(errors='replace')
    return {int(epoch)-1: int(total) for epoch, total in re.findall(
        r'\[multirate-train\] epoch=(\d+)/\d+ batch=\d+/(\d+)', tail)}


def trend_report(epoch, report, previous=None):
    previous_rows = {(r['rate'], r['level'], r['family']): r for r in (previous or {}).get('metrics', [])}
    rows = []
    for row in report['metrics']:
        before = previous_rows.get((row['rate'], row['level'], row['family']))
        if before is not None and any(before[key] != row[key] for key in ('values', 'unique_pairs', 'persistence_mse')):
            raise ValueError('Validation targets or persistence baseline changed between epochs')
        skill, old_skill = row['skill'], before['skill'] if before else None
        rows.append({**row, 'delta_skill': skill-old_skill if skill is not None and old_skill is not None else None})
    measured = [r for r in rows if r['values']]
    return {**report, 'epoch': epoch, 'previous_epoch': (previous or {}).get('epoch'), 'metrics': rows,
            'groups_beating_persistence': sum(r['beats_persistence'] is True for r in measured),
            'measured_groups': len(measured), 'zero_error_baseline_groups': sum(r['persistence_mse'] == 0 for r in measured)}


def format_epoch_report(report):
    # TOON tabular arrays; JSON string quoting also escapes commas/newlines.
    fields = ('rate', 'level', 'family', 'model_mse', 'persistence_mse', 'skill', 'delta_skill', 'values')
    lines = [f"epoch: {report['epoch']}", f"groups_beating_persistence: {report['groups_beating_persistence']}",
             f"measured_groups: {report['measured_groups']}", 'skill_direction: larger is better; positive beats persistence',
             f"ntp[{len(report['metrics'])}]{{{','.join(fields)}}}:"]
    for row in report['metrics']:
        lines.append('  ' + ','.join(json.dumps(round(row[f], 6) if isinstance(row[f], float) else row[f]) for f in fields))
    return '\n'.join(lines)


def anchored_epoch_backtest(command, directory, start, end, previous=None):
    """Use full-calendar epoch scores and a frozen adjusted-price snapshot."""
    import polars as pl
    from quant_warehouse import Warehouse
    from quant_orchestrator.platforms.backtesting_frameworks.anchored_hits_replay import replay_anchored_hits
    corpus = Path(option(command, '--corpus'))
    symbols = pl.read_csv(corpus/'taxonomy.csv').filter(pl.col('asset_class') == 'equity')['symbol'].to_list()
    cache = directory.parent/'backtest_prices'
    cache.mkdir(exist_ok=True)
    warehouse = Warehouse()
    paths = []
    for symbol in symbols:
        path = cache/f'{symbol}.parquet'
        if not path.exists():
            frame = warehouse.read_prices(symbol,provider='fmp',start=start,end=end,adjustment='splits_and_dividends')
            if frame.is_empty():
                raise ValueError(f'Missing adjusted backtest prices for {symbol}')
            temporary = path.with_suffix('.tmp')
            frame.select(pl.lit(symbol).alias('symbol'),'date','close').write_parquet(temporary)
            temporary.replace(path)
        paths.append(path)
    scores = pl.scan_csv(directory/'supervised_predictions.csv',try_parse_dates=True).filter(pl.col('symbol').is_in(symbols))
    reports = []
    for side in ('long','short'):
        report = replay_anchored_hits(scores,pl.scan_parquet(paths),directory/f'backtest_{side}',start=start,end=end,side=side)
        before = next((r for r in (previous or []) if r['side'] == side),None)
        report['return_change_vs_previous_epoch'] = report['total_return']-before['total_return'] if before else None
        reports.append(report)
    (directory/'backtest_metrics.json').write_text(json.dumps(reports,indent=2))
    return reports


def format_backtest_report(reports):
    fields = ('side','total_return','max_drawdown','entries','mean_gross_exposure','return_change_vs_previous_epoch')
    lines = [f"backtest[{len(reports)}]{{{','.join(fields)}}}:"]
    for row in reports:
        lines.append('  '+','.join(json.dumps(round(row[f],6) if isinstance(row[f],float) else row[f]) for f in fields))
    return '\n'.join(lines)
