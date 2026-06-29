from __future__ import annotations

from typing import Any

from quant_orchestrator.platforms.contracts import ProviderManifest


class TransformersFramework:
    name = "transformers"

    def fit(self, dataset: Any, **kwargs: Any) -> Any:
        trainer = kwargs.get("trainer")
        if trainer is None:
            raise ValueError("TransformersFramework.fit requires trainer=<Trainer>")
        return trainer.train(**kwargs.get("train_kwargs", {}))

    def predict(self, model: Any, dataset: Any, **kwargs: Any) -> Any:
        pipeline = kwargs.get("pipeline")
        if pipeline is not None:
            return pipeline(dataset)
        return model(dataset)


transformers_provider = ProviderManifest(
    name="transformers",
    category="ml_framework",
    display_name="Transformers",
    description="Adapter shell for Hugging Face Transformers training and inference.",
    website="https://huggingface.co/docs/transformers",
    capabilities=("fit", "predict"),
    adapters={"default": TransformersFramework},
)
