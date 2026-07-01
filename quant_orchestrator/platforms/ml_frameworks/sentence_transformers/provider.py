from __future__ import annotations

from typing import Any

from quant_orchestrator.platforms.contracts import ProviderManifest


class SentenceTransformersFramework:
    name = "sentence_transformers"

    def fit(self, dataset: Any, **kwargs: Any) -> Any:
        model = kwargs.get("model")
        estimator = kwargs.get("estimator")
        if model is None or estimator is None:
            raise ValueError("SentenceTransformersFramework.fit requires model=<SentenceTransformer> and estimator=<estimator>")
        embeddings = model.encode(
            dataset.train_texts,
            batch_size=kwargs.get("batch_size", 128),
            convert_to_numpy=True,
            show_progress_bar=kwargs.get("show_progress_bar", False),
            normalize_embeddings=kwargs.get("normalize_embeddings", True),
        )
        return estimator.fit(embeddings, dataset.train_labels)

    def predict(self, model: Any, dataset: Any, **kwargs: Any) -> Any:
        estimator = kwargs.get("estimator")
        if estimator is None:
            raise ValueError("SentenceTransformersFramework.predict requires estimator=<fitted estimator>")
        embeddings = model.encode(
            dataset.test_texts,
            batch_size=kwargs.get("batch_size", 128),
            convert_to_numpy=True,
            show_progress_bar=kwargs.get("show_progress_bar", False),
            normalize_embeddings=kwargs.get("normalize_embeddings", True),
        )
        return estimator.predict(embeddings)


sentence_transformers_provider = ProviderManifest(
    name="sentence_transformers",
    category="ml_framework",
    display_name="Sentence Transformers",
    description="Adapter shell for sentence-transformers embedding models.",
    website="https://www.sbert.net",
    capabilities=("fit", "predict", "embed"),
    adapters={"default": SentenceTransformersFramework},
)
