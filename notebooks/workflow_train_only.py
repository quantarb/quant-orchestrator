from __future__ import annotations

from dataclasses import asdict

import pandas as pd

from quant_orchestrator.artifacts import get_artifact_store


def main() -> None:
    store = get_artifact_store()
    run = store.create_run(
        run_type="ml_training",
        name="train_only_example",
        params={"framework": "sklearn", "dataset": "warehouse:features:v1"},
    )

    # Replace this with a real training call when wiring a specific ML framework.
    model = {"framework": "sklearn", "estimator": "placeholder"}
    predictions = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "symbol": ["AAPL", "MSFT"],
            "prob_buy": [0.71, 0.53],
        }
    )

    model_artifact = store.save_json(
        run_id=run.id,
        kind="ml_model",
        name="train_only_model",
        payload=model,
    )
    prediction_artifact = store.save_dataframe(
        run_id=run.id,
        kind="ml_predictions",
        name="train_only_predictions",
        frame=predictions,
    )
    store.complete_run(run.id, metrics={"rows": float(len(predictions))})

    print(asdict(model_artifact))
    print(asdict(prediction_artifact))


if __name__ == "__main__":
    main()
