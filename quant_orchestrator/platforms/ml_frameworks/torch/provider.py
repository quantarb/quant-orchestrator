from __future__ import annotations

from typing import Any

from quant_orchestrator.platform.contracts import ProviderManifest


class TorchFramework:
    name = "torch"

    def fit(self, dataset: Any, **kwargs: Any) -> Any:
        trainer = kwargs.get("trainer")
        if trainer is None:
            raise ValueError("TorchFramework.fit requires trainer=<callable>")
        return trainer(dataset, **{k: v for k, v in kwargs.items() if k != "trainer"})

    def predict(self, model: Any, dataset: Any, **kwargs: Any) -> Any:
        predictor = kwargs.get("predictor")
        if predictor is not None:
            return predictor(model, dataset, **kwargs)
        return model(dataset)


torch_provider = ProviderManifest(
    name="torch",
    category="ml_framework",
    display_name="PyTorch",
    description="Adapter shell for PyTorch model training and inference.",
    website="https://pytorch.org",
    capabilities=("fit", "predict"),
    adapters={"default": TorchFramework},
)
