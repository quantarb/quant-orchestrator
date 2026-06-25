from __future__ import annotations

from importlib.metadata import entry_points
from typing import Any

from quant_orchestrator.platform.contracts import ProviderManifest

ENTRY_POINT_GROUPS = {
    "ml_framework": "quant_orchestrator.ml_framework",
    "backtest_engine": "quant_orchestrator.backtest_engine",
    "broker": "quant_orchestrator.broker",
    "experiment_tracker": "quant_orchestrator.experiment_tracker",
}


class ProviderRegistry:
    """Runtime registry for built-in and installed orchestrator providers."""

    def __init__(self) -> None:
        self._providers: dict[str, dict[str, ProviderManifest]] = {
            category: {} for category in ENTRY_POINT_GROUPS
        }
        self._loaded_entry_points = False

    def register(self, manifest: ProviderManifest) -> ProviderManifest:
        category = _normalize_category(manifest.category)
        self._providers.setdefault(category, {})[manifest.name] = manifest
        return manifest

    def get(self, category: str, name: str) -> ProviderManifest:
        self.load_entry_points()
        normalized_category = _normalize_category(category)
        try:
            return self._providers[normalized_category][name]
        except KeyError as exc:
            available = ", ".join(sorted(self._providers.get(normalized_category, {})))
            raise KeyError(
                f"Unknown {normalized_category} provider '{name}'. Available: {available}",
            ) from exc

    def adapter(self, category: str, provider: str, adapter: str = "default") -> type[Any]:
        manifest = self.get(category, provider)
        try:
            return manifest.adapters[adapter]
        except KeyError as exc:
            available = ", ".join(sorted(manifest.adapters))
            raise KeyError(
                f"Unknown adapter '{adapter}' for {category} provider '{provider}'. "
                f"Available: {available}",
            ) from exc

    def list(self, category: str | None = None) -> dict[str, list[ProviderManifest]]:
        self.load_entry_points()
        if category is not None:
            normalized_category = _normalize_category(category)
            return {normalized_category: list(self._providers.get(normalized_category, {}).values())}
        return {
            provider_category: list(providers.values())
            for provider_category, providers in self._providers.items()
        }

    def load_entry_points(self) -> None:
        if self._loaded_entry_points:
            return
        for category, group in ENTRY_POINT_GROUPS.items():
            for entry_point in entry_points(group=group):
                loaded = entry_point.load()
                manifest = loaded() if callable(loaded) and not isinstance(loaded, ProviderManifest) else loaded
                if not isinstance(manifest, ProviderManifest):
                    raise TypeError(
                        f"Entry point {entry_point.name!r} in {group!r} did not return ProviderManifest",
                    )
                if _normalize_category(manifest.category) != category:
                    raise ValueError(
                        f"Entry point {entry_point.name!r} category mismatch: {manifest.category!r}",
                    )
                self.register(manifest)
        self._loaded_entry_points = True


registry = ProviderRegistry()


def _normalize_category(category: str) -> str:
    normalized = str(category).strip().lower().replace("-", "_")
    aliases = {
        "ml": "ml_framework",
        "ml_frameworks": "ml_framework",
        "backtest": "backtest_engine",
        "backtests": "backtest_engine",
        "backtest_engines": "backtest_engine",
        "brokers": "broker",
        "tracker": "experiment_tracker",
        "trackers": "experiment_tracker",
        "tracking": "experiment_tracker",
        "experiment_trackers": "experiment_tracker",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in ENTRY_POINT_GROUPS:
        raise ValueError(f"Unknown provider category: {category}")
    return normalized
