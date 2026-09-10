"""Train chronological issuer-context comparisons, score, audit and replay."""

import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.build_multirate_mtl_corpus import Parser


def main():
    parser = Parser(
        description=__doc__,
        epilog="Example: python scripts/validate_multirate_mtl.py --corpus artifacts/corpus --output-dir artifacts/validation --years 2024,2025",
    )
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--years", default="2024,2025", help="Chronological held-out calendar years"
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output_dir.exists() or not args.corpus.is_dir() or args.epochs < 1:
        parser.error("Require an existing corpus, a new output directory and positive epochs")
    try:
        years = sorted({int(y) for y in args.years.split(",")})
    except ValueError:
        parser.error("--years requires comma-separated calendar years")
    from quant_orchestrator.research_tools.multirate_audit import audit_corpus, evaluate_predictions
    from quant_orchestrator.research_tools.multirate_benchmark import benchmark_reuse
    from quant_orchestrator.platforms.backtesting_frameworks.multirate_replay import (
        replay_multirate,
    )
    import polars as pl

    args.output_dir.mkdir(parents=True)
    script = Path(__file__).with_name("train_multirate_mtl.py")
    common = [
        sys.executable,
        str(script),
        "--corpus",
        str(args.corpus.resolve()),
        "--batch-size",
        "8",
        "--d-model",
        "16",
        "--num-heads",
        "2",
        "--layers",
        "1",
        "--annual-window",
        "8",
        "--quarterly-window",
        "16",
        "--daily-window",
        "64",
        "--mrl-dimensions",
        "",
        "--skip-embeddings",
        "--skip-t-sne",
        "--disable-document-tasks",
        "--country",
        "",
        "--currency",
        "",
        "--exchanges",
        "",
        "--device",
        args.device,
        "--checkpoint-every-batches",
        "0",
    ]
    reports = []
    for year in years:
        start, end = f"{year}-01-01", f"{year}-12-31"
        audit_corpus(args.corpus, start)
        for mode in ["full", "none"]:
            train = args.output_dir / f"{year}_{mode}"
            scores = args.output_dir / f"{year}_{mode}_scores"
            for name, command in [
                (
                    "train",
                    [
                        *common,
                        "--output-dir",
                        str(train),
                        "--epochs",
                        str(args.epochs),
                        "--issuer-context",
                        mode,
                        "--train-end-date",
                        start,
                        "--prediction-start-date",
                        start,
                    ],
                ),
                (
                    "scores",
                    [
                        *common,
                        "--output-dir",
                        str(scores),
                        "--checkpoint",
                        str(train / "multirate_mtl_model.pt"),
                        "--inference-only",
                        "--prediction-start-date",
                        start,
                        "--prediction-end-date",
                        end,
                    ],
                ),
            ]:
                print(f"{year}/{mode}/{name}", file=sys.stderr, flush=True)
                with (args.output_dir / f"{year}_{mode}_{name}.log").open("w") as log:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            predictions = scores / "supervised_predictions.csv"
            metrics = evaluate_predictions(args.corpus, predictions, start=start, end=end)
            metrics["replay"] = replay_multirate(
                args.corpus,
                predictions,
                args.output_dir / f"{year}_{mode}_replay",
                start=start,
                end=end,
            )
            metrics["issuer_context"] = mode
            (args.output_dir / f"{year}_{mode}_evaluation.json").write_text(
                json.dumps(metrics, indent=2)
            )
            reports.append(metrics)
        baseline = args.output_dir / f"{year}_baseline_scores.csv"
        pl.scan_parquet(args.corpus / "daily.parquet").filter(
            pl.col("date").dt.year() == year
        ).select("symbol", "date").with_columns(
            pl.lit(1.0).alias("oracle_is_buy"),
            pl.lit(0.0).alias("oracle_is_short"),
            pl.lit(0.0).alias("oracle_is_sell"),
            pl.lit(0.0).alias("hits_long_return_hub"),
        ).sink_csv(baseline)
        replay_multirate(
            args.corpus,
            baseline,
            args.output_dir / f"{year}_equity_hold",
            start=start,
            end=end,
            baseline=True,
        )
    benchmark_reuse(args.output_dir / "cache_benchmark.json", device=args.device)
    (args.output_dir / "evaluation.json").write_text(json.dumps(reports, indent=2))
    print("status: complete")
    print("report: " + json.dumps(str(args.output_dir / "evaluation.json")))


if __name__ == "__main__":
    main()
