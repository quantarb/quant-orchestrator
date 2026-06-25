from __future__ import annotations

from typing import Any

from quant_orchestrator.platform.contracts import ProviderManifest


class SklearnFramework:
    name = "sklearn"

    def fit(self, dataset: Any, **kwargs: Any) -> Any:
        model = kwargs.get("model")
        if model is None:
            raise ValueError("SklearnFramework.fit requires model=<estimator>")
        features = kwargs.get("features", getattr(dataset, "features", None))
        target = kwargs.get("target", getattr(dataset, "target", None))
        if features is None or target is None:
            raise ValueError("SklearnFramework.fit requires features and target")
        return model.fit(features, target)

    def predict(self, model: Any, dataset: Any, **kwargs: Any) -> Any:
        features = kwargs.get("features", getattr(dataset, "features", dataset))
        if hasattr(model, "predict_proba") and kwargs.get("probabilities", False):
            return model.predict_proba(features)
        return model.predict(features)


sklearn_provider = ProviderManifest(
    name="sklearn",
    category="ml_framework",
    display_name="scikit-learn",
    description="Adapter for scikit-learn estimators and pipelines.",
    website="https://scikit-learn.org",
    capabilities=("fit", "predict", "predict_proba"),
    adapters={"default": SklearnFramework},
)
