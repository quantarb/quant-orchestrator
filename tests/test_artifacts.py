from __future__ import annotations

import pandas as pd

from quant_orchestrator.artifacts import ArtifactStore


def test_artifact_store_saves_native_outputs(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")
    run = store.create_run(
        run_type="ml_training",
        name="framework-native-run",
        params={"framework": "sklearn"},
        tags={"dataset": "warehouse:features"},
    )

    frame = pd.DataFrame({"symbol": ["AAPL", "MSFT"], "score": [0.8, 0.2]})
    predictions = store.save_dataframe(
        run_id=run.id,
        kind="ml_predictions",
        name="scores",
        frame=frame,
    )
    report = store.save_text(
        run_id=run.id,
        kind="ml_report",
        name="native-report",
        text="framework-specific report",
    )
    model = store.save_pickle(
        run_id=run.id,
        kind="ml_model",
        name="model",
        value={"weights": [1, 2, 3]},
    )
    completed = store.complete_run(run.id, metrics={"accuracy": 0.75})
    updated = store.update_run(run.id, tags={"promoted": True})
    updated_report = store.update_artifact_metadata(report.id, {"viewer": "native"})

    assert completed.status == "completed"
    assert completed.metrics == {"accuracy": 0.75}
    assert updated.tags["promoted"] is True
    assert updated_report.metadata["viewer"] == "native"
    assert store.get_artifact(predictions.uri).path.exists()
    assert store.load_dataframe(predictions.id).equals(frame)
    assert store.load_text(report.uri) == "framework-specific report"
    assert store.load_pickle(model.uri) == {"weights": [1, 2, 3]}
    assert store.latest_artifact(kind="ml_predictions", name="scores").id == predictions.id


def test_artifact_store_registers_framework_output_directory(tmp_path):
    source = tmp_path / "flair-output"
    source.mkdir()
    (source / "training.log").write_text("native flair log", encoding="utf-8")

    store = ArtifactStore(tmp_path / "artifacts")
    run = store.create_run(run_type="ml_training", name="flair-run")
    artifact = store.register_file(
        run_id=run.id,
        kind="ml_report",
        name="flair-output",
        path=source,
    )

    assert artifact.path.is_dir()
    assert artifact.format == "directory"
    assert (artifact.path / "training.log").read_text(encoding="utf-8") == "native flair log"

    store.delete_artifact(artifact.id, delete_file=True)

    assert not artifact.path.exists()


def test_artifact_store_deletes_run_and_files(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")
    run = store.create_run(run_type="backtest", name="zipline-run")
    artifact = store.save_text(
        run_id=run.id,
        kind="backtest_report",
        name="zipline-report",
        text="zipline-native-output",
    )

    store.delete_run(run.id, delete_files=True)

    assert not artifact.path.exists()
    assert store.list_runs() == []
    assert store.list_artifacts() == []
