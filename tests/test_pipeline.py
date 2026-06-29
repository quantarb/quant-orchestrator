from __future__ import annotations

import pandas as pd
import pytest

from quant_orchestrator.pipeline import FunctionStage, MissingArtifactError, Pipeline, PipelineContext


def test_pipeline_passes_native_artifacts_between_stages() -> None:
    frame = pd.DataFrame({"close": [100.0, 101.0, 102.0]})
    pipeline = Pipeline(
        [
            FunctionStage(
                name="load_data",
                function=lambda context: {"prices": frame},
                produced_outputs=("prices",),
            ),
            FunctionStage(
                name="summarize",
                required_inputs=("prices",),
                produced_outputs=("summary",),
                function=lambda context: {
                    "summary": {
                        "rows": len(context.require("prices")),
                        "last_close": float(context.require("prices")["close"].iloc[-1]),
                    }
                },
            ),
        ],
        name="unit_test_pipeline",
    )

    result = pipeline.run()

    assert result.context.require("prices").equals(frame)
    assert result.context.require("summary") == {"rows": 3, "last_close": 102.0}
    assert [run.stage for run in result.stage_runs] == ["load_data", "summarize"]
    assert result.context.metadata["pipeline_name"] == "unit_test_pipeline"


def test_pipeline_validates_required_inputs() -> None:
    pipeline = Pipeline(
        [
            FunctionStage(
                name="needs_predictions",
                required_inputs=("predictions",),
                function=lambda context: None,
            )
        ]
    )

    with pytest.raises(MissingArtifactError, match="needs_predictions"):
        pipeline.run(PipelineContext())


def test_pipeline_validates_declared_outputs() -> None:
    pipeline = Pipeline(
        [
            FunctionStage(
                name="silent_stage",
                function=lambda context: None,
                produced_outputs=("metrics",),
            )
        ]
    )

    with pytest.raises(MissingArtifactError, match="declared outputs"):
        pipeline.run()

