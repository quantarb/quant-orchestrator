from __future__ import annotations

from dataclasses import asdict

import pandas as pd

from quant_orchestrator.artifacts import get_artifact_store


def main() -> None:
    store = get_artifact_store()
    run = store.create_run(
        run_type="backtest",
        name="backtest_only_example",
        params={"framework": "vectorbt", "strategy": "native_strategy"},
    )

    # Replace this with a real backtest engine adapter.
    equity = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=5, freq="B"),
            "equity": [100000, 100800, 101100, 100900, 101700],
        }
    )
    report = {
        "framework": "vectorbt",
        "strategy": "native_strategy",
        "total_return": 0.017,
    }

    equity_artifact = store.save_dataframe(
        run_id=run.id,
        kind="backtest_equity",
        name="backtest_only_equity",
        frame=equity,
    )
    report_artifact = store.save_json(
        run_id=run.id,
        kind="backtest_report",
        name="backtest_only_report",
        payload=report,
    )
    store.complete_run(run.id, metrics={"total_return": float(report["total_return"])})

    print(asdict(equity_artifact))
    print(asdict(report_artifact))


if __name__ == "__main__":
    main()
