from __future__ import annotations

from dataclasses import dataclass
import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class LeanRunConfig:
    workspace_dir: Path
    project_name: str
    lean_executable: str = "lean"
    capture_output: bool = True

    @property
    def project_dir(self) -> Path:
        return self.workspace_dir / self.project_name

    @property
    def report_path(self) -> Path:
        return self.project_dir / "report.html"


@dataclass(frozen=True)
class LeanRunResult:
    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


def default_config(
    *,
    workspace_dir: str | os.PathLike[str] | None = None,
    project_name: str | None = None,
    lean_executable: str = "lean",
) -> LeanRunConfig:
    return LeanRunConfig(
        workspace_dir=Path(workspace_dir or os.environ.get("LEAN_WORKSPACE_DIR", ".")).expanduser(),
        project_name=str(project_name or os.environ.get("LEAN_PROJECT_NAME", "")).strip(),
        lean_executable=lean_executable,
    )


def lean_available(lean_executable: str = "lean") -> bool:
    return shutil.which(lean_executable) is not None


def run_lean_backtest(
    config: LeanRunConfig,
    *,
    backtest_args: Sequence[str] | None = None,
) -> LeanRunResult:
    command = [config.lean_executable, "backtest", config.project_name]
    if backtest_args:
        command.extend(str(arg) for arg in backtest_args)
    return _run_command(command, cwd=config.workspace_dir, capture_output=config.capture_output)


def run_lean_report(
    config: LeanRunConfig,
    *,
    report_args: Sequence[str] | None = None,
) -> LeanRunResult:
    command = [config.lean_executable, "report"]
    if report_args:
        command.extend(str(arg) for arg in report_args)
    return _run_command(command, cwd=config.workspace_dir, capture_output=config.capture_output)


def _run_command(
    command: Sequence[str],
    *,
    cwd: Path,
    capture_output: bool,
) -> LeanRunResult:
    proc = subprocess.run(
        list(command),
        cwd=str(cwd),
        text=True,
        capture_output=capture_output,
        check=False,
    )
    stdout = proc.stdout if capture_output and proc.stdout is not None else ""
    stderr = proc.stderr if capture_output and proc.stderr is not None else ""
    if proc.returncode != 0:
        cmd_str = " ".join(shlex.quote(part) for part in command)
        raise RuntimeError(
            f"LEAN command failed with exit code {proc.returncode}: {cmd_str}\n"
            f"stdout:\n{stdout}\n"
            f"stderr:\n{stderr}",
        )
    return LeanRunResult(
        command=tuple(str(part) for part in command),
        returncode=proc.returncode,
        stdout=stdout,
        stderr=stderr,
    )
