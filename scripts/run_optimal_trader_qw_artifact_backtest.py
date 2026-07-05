from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quant_orchestrator.platforms.backtesting_frameworks.optimal_trader.artifact_replay import (
    OptimalTraderArtifactReplayConfig,
    run_optimal_trader_artifact_replay,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", default=str(PROJECT_ROOT / "optimal_trader" / "artifacts" / "raw_stack"))
    parser.add_argument("--feature-start", default="2020-01-01")
    parser.add_argument("--backtest-start", default="2022-01-01")
    parser.add_argument("--end-date", default="2026-06-23")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--component-threshold", type=float, default=0.50)
    parser.add_argument("--price-provider", default="fmp")
    parser.add_argument("--max-symbols", type=int, default=0)
    parser.add_argument("--output-dir", default=str(Path("artifacts") / "optimal_trader_qw_artifact_backtest_after_2021"))
    parser.add_argument("--reuse-feature-panel", action="store_true")
    parser.add_argument("--reuse-scored-panel", action="store_true")
    parser.add_argument("--run-fmp-synthetic-options", action="store_true")
    parser.add_argument("--option-workers", type=int, default=1)
    args = parser.parse_args()

    result = run_optimal_trader_artifact_replay(
        OptimalTraderArtifactReplayConfig(
            artifact_dir=Path(args.artifact_dir),
            output_dir=Path(args.output_dir),
            feature_start=str(args.feature_start),
            backtest_start=str(args.backtest_start),
            end_date=str(args.end_date),
            top_k=int(args.top_k),
            component_threshold=float(args.component_threshold),
            price_provider=str(args.price_provider),
            max_symbols=int(args.max_symbols),
            reuse_feature_panel=bool(args.reuse_feature_panel),
            reuse_scored_panel=bool(args.reuse_scored_panel),
            run_fmp_synthetic_options=bool(args.run_fmp_synthetic_options),
            option_workers=max(1, int(args.option_workers)),
        )
    )
    print(json.dumps(result.summary, indent=2, default=str))


if __name__ == "__main__":
    main()
