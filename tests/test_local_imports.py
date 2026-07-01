from __future__ import annotations

from pathlib import Path


def test_quant_orchestrator_imports_from_checkout() -> None:
    import quant_orchestrator

    repo_root = Path(__file__).resolve().parents[1]
    package_path = Path(quant_orchestrator.__file__).resolve()
    assert package_path.is_relative_to(repo_root)
