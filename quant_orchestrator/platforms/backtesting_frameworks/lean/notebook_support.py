from __future__ import annotations

from pathlib import Path

from quant_orchestrator.platforms.backtesting_frameworks.lean.runner import (
    LeanRunConfig,
    lean_available,
    run_lean_backtest,
    run_lean_report,
)


def build_lean_config(
    workspace_dir: str | Path,
    project_name: str,
    *,
    lean_executable: str = "lean",
) -> LeanRunConfig:
    return LeanRunConfig(
        workspace_dir=Path(workspace_dir).expanduser(),
        project_name=project_name,
        lean_executable=lean_executable,
    )


def run_lean_backtest_and_report(
    workspace_dir: str | Path,
    project_name: str,
    *,
    lean_executable: str = "lean",
) -> dict[str, object]:
    config = build_lean_config(workspace_dir, project_name, lean_executable=lean_executable)
    backtest_result = run_lean_backtest(config)
    report_result = run_lean_report(config)
    return {
        "config": config,
        "backtest": backtest_result,
        "report": report_result,
        "report_path": config.report_path,
        "lean_available": lean_available(lean_executable),
    }
