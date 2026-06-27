from __future__ import annotations

from typing import Any

from quant_orchestrator.platform.contracts import ProviderManifest
from quant_orchestrator.runtime import move_to_device, resolve_torch_device


class TorchFramework:
    name = "torch"

    def fit(self, dataset: Any, **kwargs: Any) -> Any:
        trainer = kwargs.get("trainer")
        if trainer is None:
            raise ValueError("TorchFramework.fit requires trainer=<callable>")
        device = resolve_torch_device(kwargs.get("device"))
        prepared = move_to_device(dataset, device)
        trainer_kwargs = {k: v for k, v in kwargs.items() if k not in {"trainer", "device"}}
        return trainer(prepared, device=device, **trainer_kwargs)

    def predict(self, model: Any, dataset: Any, **kwargs: Any) -> Any:
        device = resolve_torch_device(kwargs.get("device"))
        prepared_model = move_to_device(model, device)
        prepared_dataset = move_to_device(dataset, device)
        predictor = kwargs.get("predictor")
        if predictor is not None:
            predictor_kwargs = {k: v for k, v in kwargs.items() if k not in {"predictor", "device"}}
            return predictor(prepared_model, prepared_dataset, device=device, **predictor_kwargs)
        return prepared_model(prepared_dataset)


torch_provider = ProviderManifest(
    name="torch",
    category="ml_framework",
    display_name="PyTorch",
    description="CUDA-first adapter for PyTorch model training and inference.",
    website="https://pytorch.org",
    capabilities=("fit", "predict", "cuda"),
    adapters={"default": TorchFramework},
)
