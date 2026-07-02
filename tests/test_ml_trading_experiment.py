from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd

from quant_orchestrator.dagster_defs import defs
from quant_orchestrator.platforms.ml_frameworks.torch_autoencoder.latent_index import (
    LatentAutoencoderConfig,
    LatentAutoencoderIndex,
    _candidate_architectures,
)
from quant_orchestrator.research_tools import MLTradingExperimentConfig
from quant_orchestrator.research_tools.ml_trading_experiment import _write_trained_model_artifacts


def test_ml_trading_experiment_config_defaults_to_fast_1t_smoke() -> None:
    config = MLTradingExperimentConfig()

    assert config.experiment_name == "gpu_rf_shared_book_1t"
    assert config.mode == "classifier"
    assert config.min_market_cap == 1_000_000_000_000
    assert config.log_mlflow is True
    assert config.mlflow_experiment == "ml_trading"


def test_dagster_registers_ml_trading_experiment_job() -> None:
    job_def = defs.get_job_def("ml_trading_experiment_job")

    assert job_def.name == "ml_trading_experiment_job"


def test_trained_model_artifact_writer_saves_classifier_payload(tmp_path) -> None:
    classifier = {"model": "fake-rf", "classes": ["event_bullish", "oracle_long"]}

    _write_trained_model_artifacts(
        tmp_path,
        {
            ("fmp", "fmp_daily_mcap_yield"): {
                "classifier": classifier,
                "autoencoder": None,
                "features": ["price_to_sales", "price_to_book"],
            }
        },
    )

    model_dir = tmp_path / "fmp.fmp_daily_mcap_yield"
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))

    assert manifest["models"][0]["strategy_source"] == "fmp.fmp_daily_mcap_yield"
    assert metadata["feature_count"] == 2
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
