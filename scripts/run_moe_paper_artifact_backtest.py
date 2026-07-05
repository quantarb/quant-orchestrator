from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quant_orchestrator.platforms.backtesting_frameworks.optimal_trader.moe_paper import (  # noqa: E402
    MoePaperReplayConfig,
    run_moe_paper_artifact_replay,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay optimal_trader MoE paper-trading artifacts historically.")
    parser.add_argument(
        "--score-artifact",
        default=str(PROJECT_ROOT / "optimal_trader" / "artifacts" / "moe_paper_trading" / "latest_scored.pkl"),
    )
    parser.add_argument(
        "--model-artifact-dir",
        default=str(PROJECT_ROOT / "optimal_trader" / "artifacts" / "synthetic_options_classifier_families"),
    )
    parser.add_argument("--feature-panel", default="", help="Optional parquet feature panel with date/symbol columns or index.")
    parser.add_argument("--scored-panel", default="", help="Optional parquet scored panel; skips model scoring when supplied.")
    parser.add_argument("--backtest-start", default="2021-01-01")
    parser.add_argument("--end-date", default="2026-06-24")
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--initial-balance", type=float, default=100_000.0)
    parser.add_argument("--fee-bps", type=float, default=5.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--output-dir", default=str(Path("artifacts") / "moe_paper_artifact_backtest_after_2021"))
    parser.add_argument("--run-fmp-synthetic-options", action="store_true")
    parser.add_argument("--option-workers", type=int, default=1)
    args = parser.parse_args()

    result = run_moe_paper_artifact_replay(
        MoePaperReplayConfig(
            score_artifact=Path(args.score_artifact),
            model_artifact_dir=Path(args.model_artifact_dir) if str(args.model_artifact_dir).strip() else None,
            output_dir=Path(args.output_dir),
            feature_panel_path=Path(args.feature_panel) if str(args.feature_panel).strip() else None,
            scored_panel_path=Path(args.scored_panel) if str(args.scored_panel).strip() else None,
            backtest_start=str(args.backtest_start),
            end_date=str(args.end_date),
            top_k=int(args.top_k),
            threshold=float(args.threshold),
            initial_balance=float(args.initial_balance),
            fee_bps=float(args.fee_bps),
            slippage_bps=float(args.slippage_bps),
            run_fmp_synthetic_options=bool(args.run_fmp_synthetic_options),
            option_workers=max(1, int(args.option_workers)),
        )
    )
    print(json.dumps(result.summary, indent=2, default=str))


if __name__ == "__main__":
    main()
