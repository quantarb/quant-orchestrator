"""Print fixed-validation NTP metrics after each completed training epoch."""
import argparse
from datetime import date
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quant_orchestrator.research_tools.epoch_evaluation import (
    option, evaluation_command, last_epoch_batches, trend_report, format_epoch_report,
    anchored_epoch_backtest, yearly_epoch_backtests, format_backtest_report,
)


class Parser(argparse.ArgumentParser):
    def error(self, message):
        print('error: ' + json.dumps(message))
        print('help: Run monitor_multirate_epochs.py --help for valid arguments')
        raise SystemExit(2)


def main():
    parser = Parser(description=__doc__, epilog='Example: monitor_multirate_epochs.py --command-file run.json --training-log train.log --output-dir epoch_validation --validation-start 2024-01-02 --validation-end 2024-12-31 --training-pid 1234')
    parser.add_argument('--command-file', type=Path, required=True)
    parser.add_argument('--training-log', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--validation-start', required=True)
    parser.add_argument('--validation-end', required=True)
    parser.add_argument('--inference-corpus', type=Path, help='Separate frozen corpus for evaluation; training corpus is unchanged')
    parser.add_argument('--backtest-by-year', action='store_true', help='Report independent calendar-year books, including a partial final year')
    parser.add_argument('--max-samples', type=int, default=256)
    parser.add_argument('--poll-seconds', type=float, default=10)
    parser.add_argument('--training-pid', type=int, help='Detect training exit without stopping or modifying it')
    parser.add_argument('--backtest-anchored-hits', action='store_true', help='Score the full validation calendar and run the original transformer HITS policy/shared-book engine after every epoch.')
    parser.add_argument('--once', action='store_true', help='Evaluate the latest complete epoch once; fail if none exists')
    args = parser.parse_args()
    command = json.loads(args.command_file.read_text())
    cutoff = option(command, '--train-end-date')
    if not cutoff or not date.fromisoformat(cutoff) <= date.fromisoformat(args.validation_start) <= date.fromisoformat(args.validation_end):
        parser.error('Validation must start on or after the recorded training cutoff and end on or after its start')
    if args.max_samples < 1 or not 0 < args.poll_seconds <= 60:
        parser.error('max-samples must be positive and poll-seconds must be in (0, 60]')
    if args.training_pid is not None and args.training_pid <= 0:
        parser.error('training-pid must be positive')
    training = Path(option(command, '--output-dir'))
    if args.inference_corpus:
        command[command.index('--corpus') + 1] = str(args.inference_corpus.resolve())
    if args.backtest_by_year and not args.backtest_anchored_hits:
        parser.error('--backtest-by-year requires --backtest-anchored-hits')
    latest = training / 'multirate_mtl_checkpoint_latest.pt'
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configuration = dict(command=command, validation_start=args.validation_start,
                         validation_end=args.validation_end, max_samples=0 if args.backtest_anchored_hits else args.max_samples,
                         backtest_anchored_hits=args.backtest_anchored_hits,
                         backtest_by_year=args.backtest_by_year)
    config_path = args.output_dir / 'configuration.json'
    if config_path.exists() and json.loads(config_path.read_text()) != configuration:
        raise ValueError('Existing monitor output has a different validation configuration; select a new output directory')
    config_path.write_text(json.dumps(configuration, indent=2))
    previous_reports = sorted(args.output_dir.glob('epoch_*/epoch_metrics.json'))
    previous = json.loads(previous_reports[-1].read_text()) if previous_reports else None
    completed = previous['epoch'] if previous else 0
    previous_backtest = None
    if previous_reports and args.backtest_anchored_hits:
        previous_backtest = json.loads((previous_reports[-1].parent/"backtest_metrics.json").read_text())
    signature = None
    def status(stage, **details):
        temporary = args.output_dir / 'status.json.tmp'
        temporary.write_text(json.dumps(dict(stage=stage, completed_epoch=completed, **details), indent=2))
        temporary.replace(args.output_dir / 'status.json')
    status('waiting_for_epoch')
    import torch
    while True:
        queued = training / 'epoch_checkpoints' / f'epoch_{completed+1:04d}.pt'
        source = queued if queued.exists() else latest
        if source.exists():
            current_signature = (str(source), source.stat().st_mtime_ns)
            if current_signature != signature:
                # Preserve the inode contents even if training atomically replaces latest.
                snapshot = args.output_dir / 'candidate.pt'
                shutil.copyfile(source, snapshot)
                payload = torch.load(snapshot, map_location='cpu', weights_only=True)
                metrics = payload['metrics']
                epoch, batch = int(metrics['epoch']), int(metrics['batch'])
                del payload
                totals = last_epoch_batches(args.training_log)
                if epoch + 1 > completed and (metrics.get('epoch_complete') or batch == totals.get(epoch)):
                    if epoch != completed:
                        raise ValueError(f'Missed an epoch checkpoint: expected {completed+1}, found {epoch+1}')
                    directory = args.output_dir / f'epoch_{epoch+1:04d}'
                    directory.mkdir(exist_ok=True)
                    checkpoint = directory / 'model.pt'
                    snapshot.replace(checkpoint)
                    status('evaluating', epoch=epoch+1)
                    inference_batch = min(int(option(command, '--batch-size') or 32), 64)
                    device_name = option(command, '--device') or 'cpu'
                    if device_name.startswith('cuda') and torch.cuda.is_available():
                        free_bytes, _ = torch.cuda.mem_get_info(torch.device(device_name))
                        # Larger evaluation batches amortize Python/kernel overhead.
                        # Keep substantial headroom for wide family reconstruction.
                        if free_bytes >= 48 * 1024**3:
                            inference_batch = 64 if option(command, '--sequence-mode') == 'documents' else 128
                        elif free_bytes >= 24 * 1024**3:
                            inference_batch = 32 if option(command, '--sequence-mode') == 'documents' else 64
                    invocation = evaluation_command(command, checkpoint.resolve(), directory.resolve(),
                        args.validation_start, args.validation_end, 0 if args.backtest_anchored_hits else args.max_samples,
                        batch_size=inference_batch)
                    (directory / 'command.json').write_text(json.dumps(invocation, indent=2))
                    inference_started = time.monotonic()
                    with (directory / 'inference.log').open('w') as log:
                        result = subprocess.run(invocation, stdout=log, stderr=subprocess.STDOUT)
                    if result.returncode:
                        raise RuntimeError(f'Epoch {epoch+1} evaluation failed; inspect {directory / "inference.log"}')
                    report = trend_report(epoch+1, json.loads((directory/'ntp_evaluation.json').read_text()), previous)
                    report['inference_seconds'] = time.monotonic() - inference_started
                    if args.backtest_anchored_hits:
                        backtest_started = time.monotonic()
                        status('backtesting', epoch=epoch+1, inference_seconds=report['inference_seconds'])
                        backtest = yearly_epoch_backtests if args.backtest_by_year else anchored_epoch_backtest
                        previous_backtest = backtest(command, directory, args.validation_start, args.validation_end, previous_backtest)
                        report['backtest_seconds'] = time.monotonic() - backtest_started
                        print(f"backtest_epoch: {epoch+1}\n" + format_backtest_report(previous_backtest), flush=True)
                    (directory/'epoch_metrics.json').write_text(json.dumps(report, indent=2))
                    print(format_epoch_report(report), flush=True)
                    previous, completed = report, epoch+1
                    status('waiting_for_epoch')
                    if args.once:
                        status('complete')
                        return
                elif args.once and epoch + 1 > completed:
                    raise ValueError('Latest checkpoint is not an end-of-epoch checkpoint')
                snapshot.unlink(missing_ok=True)
                signature = current_signature
        if args.once:
            if completed:
                status('complete')
                print(f'completed_epoch: {completed}\nstatus: already evaluated')
                return
            raise ValueError('No completed epoch checkpoint is available')
        alive = True
        if args.training_pid:
            try:
                os.kill(args.training_pid, 0)
            except ProcessLookupError:
                alive = False
        if not alive:
            if (training / 'epoch_checkpoints' / f'epoch_{completed+1:04d}.pt').exists():
                continue
            if not (training / 'multirate_mtl_model.pt').exists():
                raise RuntimeError('Training exited before writing its final model')
            status('complete')
            print(f'status: complete\ncompleted_epochs: {completed}', flush=True)
            return
        time.sleep(args.poll_seconds)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print('error: ' + json.dumps(str(exc)), flush=True)
        print('help: Inspect the epoch inference log or correct the monitor arguments', flush=True)
        raise SystemExit(1)
