import pickle

import pandas as pd

from quant_orchestrator.research_tools.option_meta_ranker import (
    OptionMetaRankerConfig,
    score_option_meta_ranker,
    train_option_meta_ranker,
)


def test_trains_one_meta_ranker_from_reusable_equity_scores(monkeypatch, tmp_path):
    options = pd.DataFrame(
        [
            {
                "trade_id": f"t{trade}",
                "symbol": "AAPL",
                "entry_date": pd.Timestamp("2026-01-02") + pd.Timedelta(days=trade),
                "option_return": float(candidate),
                    "rank_y": candidate / 4.0,
                    "label_basis": "realized_exit_return",
                    "label_policy": "oracle_exit_survivors_expiration_early_fallback_v1",
                "side": "buy",
                "option_type": "call",
                "dte": 30 + candidate,
            }
            for trade in range(8)
            for candidate in range(1, 5)
        ]
    )
    scores = pd.DataFrame(
        [
            {
                "model_id": "fmp.alpha",
                "symbol": "AAPL",
                "date": pd.Timestamp("2026-01-02") + pd.Timedelta(days=trade),
                "long_score": 0.6,
                "short_score": 0.4,
                "net_score": 0.2,
            }
            for trade in range(8)
        ]
    )
    monkeypatch.setattr(
        "quant_orchestrator.research_tools.option_meta_ranker._load_option_panel",
        lambda *args, **kwargs: options,
    )
    monkeypatch.setattr(
        "quant_orchestrator.research_tools.option_meta_ranker.FamilyScoreStore.read_scores",
        lambda self, **kwargs: scores,
    )

    result = train_option_meta_ranker(
        OptionMetaRankerConfig(
            option_panel=tmp_path / "options.parquet",
            equity_score_store=tmp_path / "scores",
            output_dir=tmp_path / "model",
            n_estimators=50,
            model_backend="sklearn_random_forest",
        )
    )

    assert result.training_rows == 32
    assert result.training_trades == 8
    with result.model_path.open("rb") as handle:
        bundle = pickle.load(handle)
    assert "dte" in bundle["features"]
    assert "long_score__fmp_alpha" in bundle["features"]
    assert bundle["schema_version"] == 4
    assert bundle["equity_score_contract"] == "family_long_probability_only"
    assert bundle["option_target_contract"] == "oracle_horizon_behavior_rank_v2"
    assert bundle["model_backend"] == "sklearn_random_forest"
    assert not any(
        feature.startswith(("short_score__", "net_score__")) for feature in bundle["features"]
    )
    assert {
        "vanna",
        "charm",
        "vomma",
        "speed",
        "zomma",
        "color",
        "ultima",
    }.isdisjoint(bundle["features"])

    live_options = options.head(4).copy()
    live_scores = scores.head(1).copy()
    scored = score_option_meta_ranker(result.model_path, live_options, live_scores)
    assert scored["pred_meta_stack_rank"].notna().all()
    assert scored["selected_by_option_ensemble"].sum() == 1
