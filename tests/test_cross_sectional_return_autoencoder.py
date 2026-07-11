from __future__ import annotations

import numpy as np
import pandas as pd

from quant_orchestrator.research_tools.cross_sectional_return_autoencoder import (
    DEFAULT_RETURN_HORIZONS,
    CrossSectionalReturnMatrixAutoEncoder,
    ReturnHorizonSpec,
    build_multi_horizon_return_tensor,
    build_symbol_reconstruction_feature_frame,
    fit_multi_horizon_standardizer,
)


def test_build_multi_horizon_return_tensor_preserves_symbol_horizon_matrix() -> None:
    dates = pd.bdate_range("2023-12-27", periods=8)
    close = pd.DataFrame(
        {
            "AAPL": [100, 101, 102, 103, 104, 105, 106, 107],
            "MSFT": [200, 198, 202, 204, 206, 208, 210, 212],
        },
        index=dates,
    )
    horizons = (
        ReturnHorizonSpec("1d", 1),
        ReturnHorizonSpec("5d", 5),
        ReturnHorizonSpec("ytd", None, "ytd"),
    )

    tensor = build_multi_horizon_return_tensor(close, horizons=horizons)

    assert tensor.values.shape == (7, 2, 3)
    assert tensor.symbols == ("AAPL", "MSFT")
    assert tensor.horizons == ("1d", "5d", "ytd")
    assert tensor.dates[0] == dates[1]
    assert np.isclose(tensor.panel("1d").loc[dates[1], "AAPL"], 0.01)
    assert np.isclose(tensor.panel("5d").loc[dates[5], "AAPL"], 0.05)
    assert np.isnan(tensor.panel("ytd").loc[pd.Timestamp("2023-12-29"), "AAPL"])
    assert np.isclose(tensor.panel("ytd").loc[pd.Timestamp("2024-01-01"), "AAPL"], 103 / 102 - 1)


def test_symbol_reconstruction_feature_frame_aligns_date_symbol_rows() -> None:
    dates = pd.bdate_range("2024-01-01", periods=4)
    close = pd.DataFrame(
        {
            "AAPL": [100.0, 101.0, 103.0, 106.0],
            "MSFT": [50.0, 51.0, 52.0, 53.0],
        },
        index=dates,
    )
    tensor = build_multi_horizon_return_tensor(
        close,
        horizons=(ReturnHorizonSpec("1d", 1), ReturnHorizonSpec("2d", 2)),
    )
    reconstructed = np.nan_to_num(tensor.values, nan=0.0) * 0.5

    features = build_symbol_reconstruction_feature_frame(tensor, reconstructed)

    assert len(features) == len(tensor.dates) * len(tensor.symbols)
    row = features.loc[
        features["date"].eq(dates[2]) & features["symbol"].eq("AAPL")
    ].iloc[0]
    observed = close.loc[dates[2], "AAPL"] / close.loc[dates[1], "AAPL"] - 1
    assert np.isclose(row["csra_return_1d"], observed)
    assert np.isclose(row["csra_recon_1d"], observed * 0.5)
    assert np.isclose(row["csra_residual_1d"], observed * 0.5)
    assert np.isfinite(row["csra_signed_residual"])
    assert "csra_signed_sq_score" in features.columns
    assert "csra_residual_norm" in features.columns


def test_matrix_autoencoder_forward_preserves_matrix_shape() -> None:
    import torch

    model = CrossSectionalReturnMatrixAutoEncoder(
        symbol_count=3,
        horizon_count=2,
        hidden_dim=4,
        latent_dim=2,
    )
    x = torch.zeros((5, 3, 2), dtype=torch.float32)

    out = model(x)

    assert tuple(out.shape) == (5, 3, 2)


def test_multi_horizon_standardizer_uses_symbol_horizon_statistics() -> None:
    values = np.array(
        [
            [[1.0, 10.0], [2.0, 20.0]],
            [[2.0, 20.0], [4.0, 40.0]],
            [[3.0, 30.0], [6.0, 60.0]],
        ],
        dtype="float64",
    )

    standardized, standardizer = fit_multi_horizon_standardizer(values)

    assert standardized.shape == values.shape
    assert standardizer.center.shape == (2, 2)
    assert standardizer.scale.shape == (2, 2)


def test_default_horizons_include_requested_multi_scale_returns() -> None:
    assert tuple(spec.name for spec in DEFAULT_RETURN_HORIZONS) == (
        "1d",
        "5d",
        "30d",
        "6m",
        "ytd",
        "1y",
        "5y",
    )
