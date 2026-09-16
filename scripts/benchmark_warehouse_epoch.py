"""Time a fresh full warehouse epoch using an existing run's configuration.

Evaluation is deliberately excluded: stop at the normal evaluation boundary,
after the complete epoch and its checkpoint. Never reuse a corpus or checkpoint.
"""
import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from time import perf_counter
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--configuration', type=Path, required=True, help='Existing run configuration.json; settings only')
    parser.add_argument('--output-dir', type=Path, required=True, help='New run directory (must not exist)')
    cli = parser.parse_args()
    configuration = json.loads(cli.configuration.read_text())
    if configuration['options_per_side'] != 0:
        parser.error('This benchmark requires an equities-only configuration')
    if cli.output_dir.exists():
        parser.error('--output-dir must name a new directory')
    configuration.update(output_dir=cli.output_dir, epochs=1, checkpoint=None, resume_training=False, inference_only=False)
    configuration.pop('_warehouse_run_started', None)
    from quant_orchestrator.research_tools import warehouse_multirate_training as training
    import polars as pl
    events = []
    destination = sys.stdout
    class Capture:
        def __init__(self):
            self.pending = ''
        def write(self, value):
            destination.write(value)
            self.pending += value
            while '\n' in self.pending:
                line, self.pending = self.pending.split('\n', 1)
                if line.startswith('[warehouse-training] '):
                    events.append(json.loads(line.split(' ', 1)[1]))
            return len(value)
        def flush(self):
            destination.flush()
    class EpochMeasured(Exception):
        pass
    started_at = datetime.now(timezone.utc).isoformat()
    started = perf_counter()
    def finish(model, stream, args, epoch):
        event = events[-1]
        expected = sum(frame.filter(pl.col('date') < stream.cutoff)['date'].dt.year().n_unique()
                       for frame in stream.prices.values())
        if not event['epoch_training_complete'] or event['documents'] != {'equity': expected}:
            raise RuntimeError('Benchmark did not train every expected equity document')
        report = dict(started_at=started_at, finished_at=datetime.now(timezone.utc).isoformat(),
            wall_seconds_including_setup=perf_counter()-started, epoch_training=event,
            equities=len(stream.prices), expected_documents=expected, evaluation_excluded=True,
            checkpoint=str(args.output_dir/'checkpoint_latest.pt'), baseline_configuration=str(cli.configuration))
        (args.output_dir/'epoch_benchmark.json').write_text(json.dumps(report, indent=2)+'\n')
        (args.output_dir/'status.json').write_text(json.dumps(
            {**event, 'stage':'training_benchmark_complete', 'evaluation_excluded':True}, indent=2)+'\n')
        print('[epoch-benchmark] '+json.dumps(report), flush=True)
        raise EpochMeasured
    evaluate = training.evaluate_epoch
    training.evaluate_epoch = finish
    try:
        with redirect_stdout(Capture()):
            training.run_warehouse_training(SimpleNamespace(**configuration))
    except EpochMeasured:
        pass
    finally:
        training.evaluate_epoch = evaluate


if __name__ == '__main__':
    main()
