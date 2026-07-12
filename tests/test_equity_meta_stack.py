from __future__ import annotations

import pandas as pd

from quant_orchestrator.research_tools.equity_meta_stack import (
    EquityMetaStackConfig,
    build_equity_meta_feature_frame,
    train_equity_meta_stack,
)


def _family_scores() -> pd.DataFrame:
    rows = []
    for date_index, date in enumerate(pd.date_range("2024-01-01", periods=4, freq="D")):
        for model_index, model_id in enumerate(("fmp.technical_math", "fmp.technical_momentum")):
            long_score = 0.8 if date_index % 2 == 0 else 0.2
            rows.append(
                {
                    "run_id": "run",
                    "model_id": model_id,
                    "target_id": "target",
                    "symbol": "AAPL",
                    "date": date,
                    "long_score": long_score - model_index * 0.05,
                    "short_score": 1.0 - long_score + model_index * 0.05,
                    "lineage_fingerprint": "sha256:test",
                }
            )
    return pd.DataFrame(rows)


class FakeClassifier:
    encoder = object()

    def predict_proba_frame(self, frame, features):
        long_score = frame[features].mean(axis=1).clip(0.0, 1.0)
        return pd.DataFrame(
            {
                "prob__oracle_long": long_score,
                "prob__oracle_short": 1.0 - long_score,
            },
            index=frame.index,
        )


def test_meta_features_pivot_family_probabilities():
    wide, model_ids, lineage = build_equity_meta_feature_frame(_family_scores())

    assert model_ids == ["fmp.technical_math", "fmp.technical_momentum"]
    assert lineage == "sha256:test"
    assert len(wide) == 4
    assert {
        "long_score__fmp.technical_math",
        "long_score__fmp.technical_momentum",
    }.issubset(wide.columns)
    assert not any(
        column.startswith(("short_score__", "net_score__", "score_rank__"))
        for column in wide.columns
    )


def test_train_meta_stack_uses_same_oracle_rows_and_persists_scores(tmp_path):
    labels = pd.DataFrame(
        {
            "symbol": ["AAPL"] * 4,
            "date": pd.date_range("2024-01-01", periods=4, freq="D"),
            "collapsed_label": ["oracle_long", "oracle_short", "oracle_long", "oracle_short"],
        }
    )
    captured = {}

    def factory(frame, features, config):
        captured["rows"] = len(frame)
        captured["labels"] = frame[config.target_col].tolist()
        return FakeClassifier()

    result = train_equity_meta_stack(
        _family_scores(),
        labels,
        EquityMetaStackConfig(output_dir=tmp_path, min_train_rows=4),
        classifier_factory=factory,
    )

    assert captured == {
        "rows": 4,
        "labels": ["oracle_long", "oracle_short", "oracle_long", "oracle_short"],
    }
    assert result.training_rows == 4
    assert result.family_models == 2
    assert len(result.features) == 2
    assert all(feature.startswith("long_score__") for feature in result.features)
    assert result.model_path.exists()
    assert result.summary_path.exists()
    assert result.scores_path.exists()
    assert result.scores["model_id"].unique().tolist() == ["equity_meta_stack"]
    assert not result.scores["is_out_of_sample"].any()
