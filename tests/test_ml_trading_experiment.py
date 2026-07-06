from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd

from quant_orchestrator.dagster_defs import defs
from quant_orchestrator.backtests.ml_score import (
    build_ml_score_signal_frame,
    run_ml_score_signal_backtest,
)
from quant_orchestrator.platforms.ml_frameworks.torch_autoencoder.latent_index import (
    LatentAutoencoderConfig,
    LatentAutoencoderIndex,
    _candidate_architectures,
)
from quant_orchestrator.research_tools import MLTradingExperimentConfig
from quant_orchestrator.research_tools.ml_trading import (
    classification_probability_diagnostics,
    expected_calibration_error,
    metric_correlation_summary,
    model_vs_trading_summary,
    write_ml_trading_artifact_files,
)
from quant_orchestrator.research_tools.ml_trading_experiment import (
    _apply_feature_representation,
    _filter_selected_strategy_sources,
    _normalized_feature_representations,
    _refresh_phase_summary,
    _score_family_models,
    _run_symbol_rule_diagnostics,
    _single_symbol_positions,
    _train_family_models,
    _write_trained_model_artifacts,
)


class _ConstantClassifier:
    encoder = type("_Encoder", (), {"classes_": np.asarray(["oracle_long", "oracle_short"], dtype=object)})()

    def predict_proba_frame(self, frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "prob__oracle_long": np.full(len(frame), 0.75),
                "prob__oracle_short": np.full(len(frame), 0.25),
            },
            index=frame.index,
        )


def test_ml_trading_experiment_config_defaults_to_fast_1t_smoke() -> None:
    config = MLTradingExperimentConfig()

    assert config.experiment_name == "gpu_rf_shared_book_1t"
    assert config.mode == "classifier"
    assert config.min_market_cap == 1_000_000_000_000
    assert config.log_mlflow is True
    assert config.mlflow_experiment == "ml_trading"
    assert config.target_label_mode == "oracle_only"
    assert config.fit_all_available_data is False
    assert config.refresh_missing_fmp_data is False


def test_train_family_models_can_fit_all_available_rows(monkeypatch) -> None:
    dates = pd.date_range("2020-01-01", periods=6)
    feature_panel = pd.DataFrame(
        {
            "symbol": ["AAPL"] * 6,
            "date": dates,
            "feature_a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    feature_metadata = pd.DataFrame(
        [{"source": "fmp", "family": "unit", "feature": "feature_a"}]
    )
    labels = pd.DataFrame(
        {
            "symbol": ["AAPL"] * 6,
            "date": dates,
            "collapsed_label": ["oracle_long", "oracle_short"] * 3,
            "label_source": ["unit"] * 6,
        }
    )
    fit_rows = []

    def _fake_fit(cls, frame, *, features, target_col, random_state, params):
        fit_rows.append(len(frame))
        return _ConstantClassifier()

    monkeypatch.setattr(
        "quant_orchestrator.research_tools.ml_trading_experiment.RapidsRandomForestClassifier.fit",
        classmethod(_fake_fit),
    )

    results, _models = _train_family_models(
        MLTradingExperimentConfig(
            log_mlflow=False,
            min_train_rows_per_family=1,
            fit_all_available_data=True,
        ),
        feature_panel,
        feature_metadata,
        labels,
        train_end=pd.Timestamp("2020-01-03"),
        oos_start=pd.Timestamp("2020-01-04"),
        fit_all_available_data=True,
    )

    row = results.iloc[0]
    assert fit_rows == [6]
    assert row["status"] == "ok"
    assert row["training_window"] == "all_available"
    assert row["train_rows"] == 6
    assert row["oos_rows"] == 0


def test_refresh_phase_summary_flattens_quant_warehouse_backfill_counts() -> None:
    summary = {
        "equity_prices": {"updated": 2, "empty": 1, "skipped_fresh": 3, "error": 0, "total": 6},
        "macro": {"status": "skipped_complete"},
        "include_prices": True,
    }

    flattened = _refresh_phase_summary(summary)

    assert flattened["equity_prices_updated"] == 2
    assert flattened["equity_prices_total"] == 6
    assert flattened["macro_status"] == "skipped_complete"
    assert flattened["include_prices"] is True


def test_filter_selected_strategy_sources_keeps_only_requested_feature_families() -> None:
    metadata = pd.DataFrame(
        {
            "source": ["fmp", "fmp", "financetoolkit"],
            "family": ["a", "b", "c"],
            "feature": ["fa", "fb", "fc"],
        }
    )

    features, filtered = _filter_selected_strategy_sources(
        ["fa", "fb", "fc"],
        metadata,
        strategy_sources=("fmp.b", "financetoolkit.c"),
    )

    assert features == ["fb", "fc"]
    assert set(filtered["family"]) == {"b", "c"}


def test_dagster_registers_ml_trading_experiment_job() -> None:
    job_def = defs.get_job_def("ml_trading_experiment_job")

    assert job_def.name == "ml_trading_experiment_job"


def test_dagster_registers_thetadata_options_backfill_job() -> None:
    job_def = defs.get_job_def("thetadata_options_backfill_job")

    assert job_def.name == "thetadata_options_backfill_job"


def test_trained_model_artifact_writer_saves_classifier_payload(tmp_path) -> None:
    classifier = {"model": "fake-rf", "classes": ["event_bullish", "oracle_long"]}

    _write_trained_model_artifacts(
        tmp_path,
        {
            ("fmp", "fmp_daily_mcap_yield"): {
                "classifier": classifier,
                "autoencoder": None,
                "feature_autoencoder": None,
                "features": ["price_to_sales", "price_to_book"],
                "raw_features": ["price_to_sales", "price_to_book"],
                "base_family": "fmp_daily_mcap_yield",
                "representation": "raw",
            }
        },
    )

    model_dir = tmp_path / "fmp.fmp_daily_mcap_yield"
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))

    assert manifest["models"][0]["strategy_source"] == "fmp.fmp_daily_mcap_yield"
    assert metadata["feature_count"] == 2
    assert metadata["representation"] == "raw"
    assert metadata["base_family"] == "fmp_daily_mcap_yield"
    assert metadata["classifier_serialized"] is True
    with (model_dir / "classifier.pkl").open("rb") as handle:
        assert pickle.load(handle) == classifier


def test_autoencoder_architecture_candidates_scale_down_for_small_families() -> None:
    config = LatentAutoencoderConfig(max_architecture_candidates=8)

    candidates = _candidate_architectures(5, config)

    assert candidates
    assert max(hidden_dim for hidden_dim, _ in candidates) <= 10
    assert max(bottleneck_dim for _, bottleneck_dim in candidates) <= 4
    assert (32, 32) not in candidates


def test_autoencoder_fit_records_selected_architecture_metadata() -> None:
    rng = np.random.default_rng(7)
    frame = pd.DataFrame(
        rng.normal(size=(80, 5)),
        columns=[f"feature_{idx}" for idx in range(5)],
    )
    frame["collapsed_label"] = np.where(frame["feature_0"].gt(0), "bullish", "bearish")
    features = [col for col in frame.columns if col.startswith("feature_")]
    config = LatentAutoencoderConfig(
        epochs=1,
        tuning_epochs=1,
        batch_size=80,
        device="cpu",
        validation_fraction=0.25,
        min_validation_rows=10,
        max_architecture_candidates=3,
    )

    index = LatentAutoencoderIndex.fit(frame, features=features, config=config)

    assert index is not None
    metadata = index.metadata()
    assert metadata["ae_architecture_tuned"] == 1
    assert metadata["ae_architecture_candidates"] == 3
    assert metadata["ae_validation_rows"] == 20
    assert metadata["ae_hidden_dim"] <= 10
    assert metadata["ae_bottleneck_dim"] <= 4
    assert metadata["ae_architecture_selection_metric"] == "latent_label_purity_minus_dim_penalty"
    assert np.isfinite(metadata["ae_validation_latent_label_purity"])
    assert sum(row["selected"] for row in index.architecture_diagnostics) == 1


def test_autoencoder_feature_representations_are_compact_classifier_inputs() -> None:
    rng = np.random.default_rng(11)
    frame = pd.DataFrame(rng.normal(size=(64, 4)), columns=[f"feature_{idx}" for idx in range(4)])
    frame["symbol"] = "AAPL"
    frame["date"] = pd.date_range("2024-01-01", periods=len(frame))
    frame["collapsed_label"] = np.where(frame["feature_0"].gt(0), "oracle_long", "oracle_short")
    features = [f"feature_{idx}" for idx in range(4)]
    config = LatentAutoencoderConfig(
        epochs=1,
        tuning_epochs=1,
        batch_size=64,
        device="cpu",
        tune_architecture=False,
        max_bottleneck_dim=2,
    )
    index = LatentAutoencoderIndex.fit(frame, features=features, config=config)

    ae_only, ae_features = _apply_feature_representation(frame, features, autoencoder=index, representation="ae_only")
    raw_plus_ae, raw_plus_features = _apply_feature_representation(frame, features, autoencoder=index, representation="raw_plus_ae")

    assert _normalized_feature_representations(("raw", "ae_only", "raw")) == ("raw", "ae_only")
    assert "ae_recon_error" in ae_features
    assert "ae_latent_distance" in ae_features
    assert any(col.startswith("ae_latent_") for col in ae_features)
    assert not set(features).intersection(ae_features)
    assert set(features).issubset(raw_plus_features)
    assert set(ae_features).issubset(raw_plus_features)
    assert len(ae_only) == len(frame)
    assert len(raw_plus_ae) == len(frame)


def test_classification_probability_diagnostics_reports_calibration_metrics() -> None:
    frame = pd.DataFrame({"collapsed_label": ["bullish", "bearish", "bullish", "bearish"]})
    proba = pd.DataFrame(
        {
            "prob__bullish": [0.90, 0.20, 0.55, 0.40],
            "prob__bearish": [0.10, 0.80, 0.45, 0.60],
        }
    )

    metrics = classification_probability_diagnostics(
        frame,
        proba,
        target_col="collapsed_label",
        labels=["bullish", "bearish"],
        n_bins=5,
    )

    assert metrics["rows"] == 4
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert 0.0 <= metrics["brier_macro"] < 0.25
    assert 0.0 <= metrics["expected_calibration_error"] <= 1.0


def test_expected_calibration_error_is_zero_for_perfect_confidence() -> None:
    assert expected_calibration_error([1.0, 1.0], [1.0, 1.0], n_bins=2) == 0.0


def test_model_vs_trading_summary_joins_best_strategy_per_family() -> None:
    model_results = pd.DataFrame(
        {
            "source": ["fmp", "fmp"],
            "family": ["income_mcap", "ratios"],
            "strategy_source": ["fmp.income_mcap", "fmp.ratios"],
            "status": ["ok", "ok"],
            "features": [10, 20],
            "oos_rows": [100, 120],
            "oos_macro_f1": [0.30, 0.40],
            "oos_expected_calibration_error": [0.10, 0.20],
        }
    )
    backtest_summary = pd.DataFrame(
        {
            "strategy_source": ["fmp.income_mcap", "fmp.income_mcap", "fmp.ratios", "ensemble_mean"],
            "variant": ["long_only", "long_short", "long_only", "long_only"],
            "top_k": [5, 10, 5, 5],
            "sharpe": [0.5, 1.2, 0.8, 2.0],
            "total_return": [0.1, 0.2, 0.3, 0.4],
            "max_drawdown": [-0.2, -0.3, -0.1, -0.1],
            "win_rate": [0.51, 0.52, 0.53, 0.54],
            "signal_events": [10, 11, 12, 13],
        }
    )

    summary = model_vs_trading_summary(model_results, backtest_summary)

    assert summary.loc[0, "strategy_source"] == "fmp.income_mcap"
    assert summary.loc[0, "variant"] == "long_short"
    assert summary.loc[0, "sharpe"] == 1.2
    assert "ensemble_mean" not in set(summary["strategy_source"])


def test_metric_correlation_summary_reports_pairwise_rows() -> None:
    frame = pd.DataFrame({"oos_macro_f1": [0.1, 0.2, 0.3], "sharpe": [1.0, 2.0, 4.0]})

    summary = metric_correlation_summary(frame, x_cols=["oos_macro_f1"], y_cols=["sharpe"])

    assert summary.loc[0, "rows"] == 3
    assert summary.loc[0, "spearman"] == 1.0


def test_ml_trading_artifact_writer_saves_diagnostic_tables(tmp_path) -> None:
    paths = write_ml_trading_artifact_files(
        model_results=pd.DataFrame({"model": [1]}),
        strategy_scores=pd.DataFrame({"score": [1]}),
        backtest_summary=pd.DataFrame({"sharpe": [1.0]}),
        trade_log=pd.DataFrame({"trade": [1]}),
        model_vs_trading=pd.DataFrame({"strategy_source": ["fmp.ratios"]}),
        metric_correlations=pd.DataFrame({"x": ["oos_macro_f1"], "y": ["sharpe"]}),
        yearly_backtest_summary=pd.DataFrame({"year": [2020], "sharpe": [1.0]}),
        symbol_strategy_summary=pd.DataFrame({"symbol": ["AAPL"], "beats_buy_hold": [True]}),
        symbol_robustness_summary=pd.DataFrame({"strategy_source": ["fmp.ratios"], "beat_buy_hold_rate": [1.0]}),
        backtesting_py_symbol_validation=pd.DataFrame({"framework": ["backtesting_py_signal"], "symbol": ["AAPL"]}),
        phase_timings=pd.DataFrame({"phase": ["train_family_models"], "seconds": [1.0]}),
        analysis_markdown="analysis",
        directory=tmp_path,
    )

    assert pd.read_csv(paths["model_vs_trading"]).loc[0, "strategy_source"] == "fmp.ratios"
    assert pd.read_csv(paths["metric_correlations"]).loc[0, "x"] == "oos_macro_f1"
    assert pd.read_csv(paths["yearly_backtest_summary"]).loc[0, "year"] == 2020
    assert pd.read_csv(paths["symbol_strategy_summary"]).loc[0, "symbol"] == "AAPL"
    assert pd.read_csv(paths["symbol_robustness_summary"]).loc[0, "beat_buy_hold_rate"] == 1.0
    assert pd.read_csv(paths["backtesting_py_symbol_validation"]).loc[0, "framework"] == "backtesting_py_signal"
    assert pd.read_csv(paths["phase_timings"]).loc[0, "phase"] == "train_family_models"


def test_single_symbol_positions_follow_threshold_rules() -> None:
    scores = pd.DataFrame(
        {
            "long_score": [0.6, 0.55, 0.4, 0.2],
            "short_score": [0.1, 0.7, 0.8, 0.2],
        },
        index=pd.date_range("2020-01-01", periods=4),
    )

    long_short = _single_symbol_positions(scores, variant="long_short", entry_threshold=0.5, exit_threshold=0.5)

    assert long_short.tolist() == [1.0, -1.0, -1.0, 0.0]


def test_symbol_rule_diagnostics_reports_buy_hold_beat_rate() -> None:
    dates = pd.date_range("2020-01-01", periods=6)
    strategy_scores = pd.DataFrame(
        {
            "strategy_source": ["fmp.ratios"] * 6,
            "source": ["fmp"] * 6,
            "family": ["ratios"] * 6,
            "symbol": ["AAPL"] * 6,
            "date": dates,
            "long_score": [0.8, 0.8, 0.2, 0.2, 0.8, 0.8],
            "short_score": [0.1] * 6,
        }
    )
    price_frames = {
        "AAPL": pd.DataFrame(
            {
                "close": [100.0, 110.0, 121.0, 90.0, 99.0, 108.9],
                "open": [100.0, 110.0, 121.0, 90.0, 99.0, 108.9],
                "high": [100.0, 110.0, 121.0, 90.0, 99.0, 108.9],
                "low": [100.0, 110.0, 121.0, 90.0, 99.0, 108.9],
                "volume": [1] * 6,
            },
            index=dates,
        )
    }
    config = MLTradingExperimentConfig(log_mlflow=False)

    symbol_summary, robustness = _run_symbol_rule_diagnostics(
        config,
        strategy_scores,
        price_frames,
        oos_start=pd.Timestamp("2020-01-01"),
    )

    assert not symbol_summary.empty
    assert not robustness.empty
    assert set(symbol_summary["variant"]) == {"long_only", "short_only", "long_short"}
    assert "beat_buy_hold_rate" in robustness.columns


def test_score_family_models_uses_score_start_window() -> None:
    feature_panel = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL", "AAPL"],
            "date": pd.to_datetime(["2020-01-01", "2026-01-02", "2026-01-03"]),
            "feature_a": [1.0, 2.0, 3.0],
        }
    )
    models = {
        ("fmp", "ratios"): {
            "classifier": _ConstantClassifier(),
            "autoencoder": None,
            "features": ["feature_a"],
        }
    }

    scores, mean_scores = _score_family_models(
        MLTradingExperimentConfig(log_mlflow=False),
        feature_panel,
        models,
        score_start=pd.Timestamp("2026-01-01"),
    )

    assert set(pd.to_datetime(scores["date"]).dt.strftime("%Y-%m-%d")) == {"2026-01-02", "2026-01-03"}
    assert len(mean_scores) == 2


def test_backtesting_py_ml_score_signal_frame_contains_entry_and_exit_signals() -> None:
    dates = pd.date_range("2020-01-01", periods=4)
    prices = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0, 103.0],
            "high": [101.0, 102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0, 102.0],
            "close": [100.0, 101.0, 102.0, 103.0],
            "volume": [1, 1, 1, 1],
        },
        index=dates,
    )
    scores = pd.DataFrame(
        {
            "date": dates,
            "long_score": [0.8, 0.8, 0.2, 0.2],
            "short_score": [0.1, 0.1, 0.9, 0.9],
        }
    )

    frame = build_ml_score_signal_frame(
        prices,
        scores,
        variant="long_only",
        entry_threshold=0.5,
        exit_threshold=0.5,
    )

    assert "signal_entry_size" in frame.columns
    assert "signal_exit_portion" in frame.columns
    assert frame["signal_entry_size"].gt(0).sum() == 1
    assert frame["signal_exit_portion"].gt(0).sum() == 1


def test_backtesting_py_ml_score_backtest_runs() -> None:
    dates = pd.date_range("2020-01-01", periods=8)
    prices = pd.DataFrame(
        {
            "open": np.linspace(100, 108, len(dates)),
            "high": np.linspace(101, 109, len(dates)),
            "low": np.linspace(99, 107, len(dates)),
            "close": np.linspace(100, 108, len(dates)),
            "volume": [100] * len(dates),
        },
        index=dates,
    )
    scores = pd.DataFrame(
        {
            "date": dates,
            "long_score": [0.8] * len(dates),
            "short_score": [0.1] * len(dates),
        }
    )

    result = run_ml_score_signal_backtest(
        prices,
        scores,
        symbol="AAPL",
        strategy_source="fmp.ratios",
        variant="long_only",
        cash=10_000,
    )

    assert result.summary["framework"] == "backtesting_py_signal"
    assert result.summary["symbol"] == "AAPL"
    assert result.summary["days"] == len(dates)
