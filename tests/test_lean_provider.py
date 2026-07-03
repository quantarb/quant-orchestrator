from __future__ import annotations

from quant_orchestrator.platforms.builtins import register_builtin_providers
from quant_orchestrator.platforms.registry import registry


def test_lean_provider_is_registered() -> None:
    register_builtin_providers()
    manifest = registry.get("backtesting_framework", "lean")
    assert manifest.display_name == "QuantConnect LEAN"
    assert manifest.capabilities == ("run",)


def test_lean_provider_requires_runner_callable() -> None:
    register_builtin_providers()
    engine_cls = registry.adapter("backtesting_framework", "lean")
    engine = engine_cls()

    def runner(*, strategy, data, **kwargs):
        return {"strategy": strategy, "data": data, "kwargs": kwargs}

    result = engine.run("strategy", {"prices": [1, 2, 3]}, runner=runner, foo="bar")
    assert result["strategy"] == "strategy"
    assert result["data"] == {"prices": [1, 2, 3]}
    assert result["kwargs"]["foo"] == "bar"
