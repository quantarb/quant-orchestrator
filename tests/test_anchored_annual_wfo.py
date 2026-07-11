from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from quant_warehouse.lineage import build_dataset_lineage_manifest, write_dataset_lineage_manifest

from quant_orchestrator.research_tools.anchored_annual_wfo import (
    AnchoredAnnualWFOConfig,
    run_anchored_annual_wfo,
)
from quant_orchestrator.research_tools.family_score_pipeline import FeatureFamilyBatch, FamilyScoreStore


@dataclass
class _Encoder:
    classes_: np.ndarray


@dataclass
class _Classifier:
    encoder: _Encoder

    def predict_proba_frame(self, frame, features):
        del features
        return pd.DataFrame(
            {"prob__oracle_long": np.full(len(frame), 0.7), "prob__oracle_short": np.full(len(frame), 0.3)},
            index=frame.index,
        )


def test_annual_wfo_uses_prior_rows_bounds_oos_and_restarts(tmp_path):
    dates = pd.to_datetime(["2018-01-02", "2019-01-02", "2020-01-02", "2021-01-02", "2022-01-02"])
    panel = pd.DataFrame({"symbol": "AAPL", "date": dates, "feature": np.arange(5, dtype=float)})
    labels = pd.DataFrame(
        {"symbol": "AAPL", "date": dates[:3], "collapsed_label": ["oracle_long", "oracle_short", "oracle_long"]}
    )
    lineage = build_dataset_lineage_manifest(
        panel, dataset_id="features", dataset_kind="feature_panel", provider="test",
        available_at_cutoff="2022-01-03", recipe_id="features-v1", recipe={"family": "unit"},
    )
    lineage_path = write_dataset_lineage_manifest(lineage, tmp_path / "lineage.json")
    fit_dates = []
    factory_calls = []
    memory_samples = iter([100.0, 120.0, 125.5])

    def batches():
        factory_calls.append("called")
        return [FeatureFamilyBatch("test", "unit", panel, ("feature",))]

    def classifier_factory(frame, features, config):
        del features
        fit_dates.append((frame["date"].max(), config.train_end))
        return _Classifier(_Encoder(np.asarray(["oracle_long", "oracle_short"])))

    config = AnchoredAnnualWFOConfig(
        output_dir=tmp_path / "wfo", run_id="unit", target_id="oracle-side",
        input_lineage_paths=(lineage_path,), test_years=(2021, 2020), min_train_rows=2,
        persist_models=False, run_diagnostics=False,
    )
    first = run_anchored_annual_wfo(
        batches,
        labels,
        config=config,
        classifier_factory=classifier_factory,
        memory_probe=lambda: next(memory_samples),
    )

    assert list(first.folds["test_year"]) == [2020, 2021]
    assert factory_calls == ["called", "called"]
    assert fit_dates == [(pd.Timestamp("2019-01-02"), "2019-12-31"), (pd.Timestamp("2020-01-02"), "2020-12-31")]
    scores_2020 = FamilyScoreStore(tmp_path / "wfo/test_year=2020/scores").read_scores()
    scores_2021 = FamilyScoreStore(tmp_path / "wfo/test_year=2021/scores").read_scores()
    assert set(scores_2020["date"]) == {pd.Timestamp("2020-01-02")}
    assert set(scores_2021["date"]) == {pd.Timestamp("2021-01-02")}
    assert first.peak_unified_memory_rss_mb == 125.5
    assert first.folds["peak_unified_memory_rss_mb"].tolist() == [120.0, 125.5]

    second = run_anchored_annual_wfo(
        batches,
        labels,
        config=config,
        classifier_factory=classifier_factory,
        memory_probe=lambda: 130.0,
    )

    assert second.folds["resumed"].all()
    assert factory_calls == ["called", "called"]
