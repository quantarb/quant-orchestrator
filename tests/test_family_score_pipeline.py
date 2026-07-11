from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest
from quant_warehouse.lineage import build_dataset_lineage_manifest, write_dataset_lineage_manifest

from quant_orchestrator.research_tools.family_score_pipeline import (
    SCORE_COLUMNS,
    FamilyClassifierConfig,
    FamilyScoreStore,
    ScoreMaterializationConfig,
    build_score_ensemble,
    iter_feature_family_batches,
    train_and_materialize_family_scores,
)


@dataclass
class _Encoder:
    classes_: np.ndarray


@dataclass
class _FakeClassifier:
    encoder: _Encoder
    long_probability: float

    def predict_proba_frame(self, frame: pd.DataFrame, features) -> pd.DataFrame:
        del features
        long = np.full(len(frame), self.long_probability)
        short = 1.0 - long
        return pd.DataFrame(
            {
                "prob__oracle_long": long,
                "prob__oracle_short": short,
            },
            index=frame.index,
        )


def _feature_panel() -> pd.DataFrame:
    dates = pd.date_range("2019-12-29", periods=6, freq="D")
    return pd.DataFrame(
        {
            "symbol": ["AAPL"] * 6,
            "date": dates,
            "quality_a": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0],
            "quality_b": [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            "value_a": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
        }
    )


def _metadata() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"source": "fmp", "family": "quality", "feature": "quality_a"},
            {"source": "fmp", "family": "quality", "feature": "quality_b"},
            {"source": "fmp", "family": "value", "feature": "value_a"},
        ]
    )


def _labels() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"symbol": "AAPL", "date": "2019-12-29", "collapsed_label": "oracle_long"},
            {"symbol": "AAPL", "date": "2019-12-30", "collapsed_label": "oracle_short"},
            {"symbol": "AAPL", "date": "2019-12-31", "collapsed_label": "oracle_long"},
        ]
    )


def _lineage_paths(tmp_path):
    feature_manifest = build_dataset_lineage_manifest(
        _feature_panel(),
        dataset_id="features",
        dataset_kind="feature_panel",
        provider="fmp",
        available_at_cutoff="2020-01-04",
        recipe_id="features-v1",
        recipe={"families": ["quality", "value"]},
    )
    label_manifest = build_dataset_lineage_manifest(
        _labels(),
        dataset_id="labels",
        dataset_kind="target_panel",
        provider="fmp",
        available_at_cutoff="2020-01-04",
        recipe_id="labels-v1",
        recipe={"target": "oracle_side"},
    )
    return (
        write_dataset_lineage_manifest(feature_manifest, tmp_path / "feature_lineage.json"),
        write_dataset_lineage_manifest(label_manifest, tmp_path / "label_lineage.json"),
    )


def test_iter_feature_family_batches_yields_only_family_columns():
    batches = list(iter_feature_family_batches(_feature_panel(), _metadata()))

    assert [batch.model_id for batch in batches] == ["fmp.quality", "fmp.value"]
    assert list(batches[0].frame.columns) == ["symbol", "date", "quality_a", "quality_b"]
    assert batches[1].feature_columns == ("value_a",)


def test_streaming_pipeline_materializes_reusable_scores_and_releases_each_model(tmp_path):
    fitted = []
    cleanups = []

    def factory(frame, features, config):
        fitted.append(
            {
                "features": tuple(features),
                "rows": len(frame),
                "missing": int(frame[features].isna().sum().sum()),
                "config": config,
            }
        )
        return _FakeClassifier(
            encoder=_Encoder(np.asarray(["oracle_long", "oracle_short"])),
            long_probability=0.75 if "quality_a" in features else 0.60,
        )

    output = tmp_path / "score-run"
    result = train_and_materialize_family_scores(
        iter_feature_family_batches(_feature_panel(), _metadata()),
        _labels(),
        classifier=FamilyClassifierConfig(
            train_end="2019-12-31",
            score_start="2020-01-01",
            min_train_rows=2,
        ),
        materialization=ScoreMaterializationConfig(
            output_dir=output,
            run_id="run-1",
            target_id="oracle_side",
            input_lineage_paths=_lineage_paths(tmp_path),
            rows_per_chunk=2,
        ),
        classifier_factory=factory,
        cleanup_callback=lambda: cleanups.append("released"),
    )

    scores = FamilyScoreStore(output / "scores").read_scores()
    manifest = json.loads(result.manifest_path.read_text())
    assert result.score_rows == 6
    assert set(scores["model_id"]) == {"fmp.quality", "fmp.value"}
    assert list(scores.columns) == list(SCORE_COLUMNS)
    assert scores["run_id"].eq("run-1").all()
    assert scores["target_id"].eq("oracle_side").all()
    assert scores["is_out_of_sample"].all()
    assert scores["lineage_fingerprint"].nunique() == 1
    assert len(fitted) == 2
    assert all(row["missing"] == 0 for row in fitted)
    assert cleanups == ["released", "released"]
    assert manifest["models_trained"] == 2
    assert manifest["score_rows"] == 6
    assert len(manifest["input_lineage"]) == 2
    assert manifest["lineage_fingerprint"] == scores["lineage_fingerprint"].iloc[0]
    assert len(list((output / "models").glob("*.pkl"))) == 2


def test_score_store_supports_strategy_specific_model_and_date_reads(tmp_path):
    store = FamilyScoreStore(tmp_path)
    base = pd.DataFrame(
        [
            {
                "run_id": "r",
                "model_id": model,
                "symbol": "AAPL",
                "date": date,
                "long_score": score,
                "short_score": 1 - score,
                "net_score": 2 * score - 1,
            }
            for model, date, score in (
                ("fmp.quality", "2020-01-01", 0.7),
                ("fmp.quality", "2021-01-01", 0.8),
                ("fmp.value", "2021-01-01", 0.6),
            )
        ]
    )
    store.write_chunk(base, model_id="mixed", chunk_number=0)

    selected = store.read_scores(model_ids=["fmp.quality"], start="2021-01-01")

    assert len(selected) == 1
    assert selected.iloc[0]["model_id"] == "fmp.quality"
    assert selected.iloc[0]["long_score"] == pytest.approx(0.8)

    with pytest.raises(ValueError, match="score lineage mismatch"):
        store.read_scores(expected_lineage_fingerprint="different-lineage")


def test_score_ensemble_reuses_selected_materialized_models():
    scores = pd.DataFrame(
        [
            {"run_id": "r", "target_id": "oracle", "lineage_fingerprint": "lineage-1", "model_id": "quality", "symbol": "AAPL", "date": "2021-01-01", "long_score": 0.8, "short_score": 0.2},
            {"run_id": "r", "target_id": "oracle", "lineage_fingerprint": "lineage-1", "model_id": "value", "symbol": "AAPL", "date": "2021-01-01", "long_score": 0.6, "short_score": 0.4},
            {"run_id": "r", "target_id": "oracle", "lineage_fingerprint": "lineage-1", "model_id": "other", "symbol": "AAPL", "date": "2021-01-01", "long_score": 0.1, "short_score": 0.9},
        ]
    )

    ensemble = build_score_ensemble(scores, model_ids=["quality", "value"])

    assert len(ensemble) == 1
    assert ensemble.iloc[0]["strategy_source"] == "ensemble_mean"
    assert ensemble.iloc[0]["long_score"] == pytest.approx(0.7)
    assert ensemble.iloc[0]["short_score"] == pytest.approx(0.3)


def test_streaming_pipeline_refuses_to_mix_with_existing_output(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    (output / "old.txt").write_text("old")

    with pytest.raises(FileExistsError, match="not empty"):
        train_and_materialize_family_scores(
            [],
            _labels(),
            classifier=FamilyClassifierConfig(train_end="2019-12-31", score_start="2020-01-01"),
            materialization=ScoreMaterializationConfig(
                output_dir=output,
                run_id="run-1",
                target_id="oracle_side",
                input_lineage_paths=(),
            ),
        )


def test_streaming_pipeline_requires_valid_input_lineage(tmp_path):
    with pytest.raises(ValueError, match="input lineage manifest"):
        train_and_materialize_family_scores(
            [],
            _labels(),
            classifier=FamilyClassifierConfig(train_end="2019-12-31", score_start="2020-01-01"),
            materialization=ScoreMaterializationConfig(
                output_dir=tmp_path / "new",
                run_id="run-1",
                target_id="oracle_side",
                input_lineage_paths=(),
            ),
        )
