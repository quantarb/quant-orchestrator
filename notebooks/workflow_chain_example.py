from __future__ import annotations

import pandas as pd

from quant_orchestrator.artifacts import get_artifact_store
from quant_orchestrator.monte_carlo import simulate_return_paths


def main() -> None:
    store = get_artifact_store()

    train_run = store.create_run(
        run_type="ml_training",
        name="chain_train",
        params={"window": "2020-2022"},
    )
    prediction_artifact = store.save_dataframe(
        run_id=train_run.id,
        kind="ml_predictions",
        name="chain_predictions",
        frame=pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
                "symbol": ["AAPL", "AAPL", "MSFT"],
                "score": [0.82, 0.77, 0.69],
            }
        ),
    )
    store.complete_run(train_run.id)

    strategy_run = store.create_run(
        run_type="strategy_run",
        name="chain_strategy",
        params={"input_predictions": prediction_artifact.uri},
    )
    returns = pd.Series([0.01, -0.005, 0.007, 0.004, -0.002], name="returns")
    mc = simulate_return_paths(returns, iterations=250, horizon=5, block_size=2)
    paths_artifact = store.save_dataframe(
        run_id=strategy_run.id,
        kind="monte_carlo_paths",
        name="chain_monte_carlo_paths",
        frame=mc.paths,
    )
    summary_artifact = store.save_dataframe(
        run_id=strategy_run.id,
        kind="monte_carlo_summary",
        name="chain_monte_carlo_summary",
        frame=mc.summary,
    )
    store.complete_run(
        strategy_run.id,
        metrics={"terminal_return_mean": float(mc.summary.loc[0, "terminal_return_mean"])},
    )

    print(prediction_artifact.uri)
    print(paths_artifact.uri)
    print(summary_artifact.uri)


if __name__ == "__main__":
    main()
