from __future__ import annotations

import pandas as pd
import pytest

from quant_orchestrator.research_tools.attribution import (
    attach_security_context,
    attribute_model_scores,
    attribute_strategy_returns,
)


def _context() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"symbol": "AAPL", "date": "2024-12-31", "calendar_year": 2024, "sector": "Technology", "industry": "Hardware"},
            {"symbol": "AAPL", "date": "2025-01-02", "calendar_year": 2025, "sector": "Technology", "industry": "Hardware"},
            {"symbol": "JPM", "date": "2025-01-02", "calendar_year": 2025, "sector": "Financials", "industry": "Banks"},
        ]
    )


def test_attribute_model_scores_groups_quality_by_year_sector_and_industry():
    scores = pd.DataFrame(
        [
            {"model_id": "quality", "symbol": "AAPL", "date": "2024-12-31", "long_score": 0.8, "short_score": 0.2},
            {"model_id": "quality", "symbol": "AAPL", "date": "2025-01-02", "long_score": 0.3, "short_score": 0.7},
            {"model_id": "quality", "symbol": "JPM", "date": "2025-01-02", "long_score": 0.6, "short_score": 0.4},
        ]
    )
    labels = pd.DataFrame(
        [
            {"symbol": "AAPL", "date": "2024-12-31", "collapsed_label": "oracle_long"},
            {"symbol": "AAPL", "date": "2025-01-02", "collapsed_label": "oracle_short"},
            {"symbol": "JPM", "date": "2025-01-02", "collapsed_label": "oracle_short"},
        ]
    )

    result = attribute_model_scores(scores, labels, _context())

    technology_2025 = result.loc[
        result["calendar_year"].eq(2025) & result["sector"].eq("Technology")
    ].iloc[0]
    financials_2025 = result.loc[result["sector"].eq("Financials")].iloc[0]
    assert technology_2025["directional_accuracy"] == 1.0
    assert financials_2025["directional_accuracy"] == 0.0
    assert result["observations"].sum() == 3


def test_attribute_strategy_returns_produces_additive_net_contribution():
    observations = pd.DataFrame(
        [
            {"strategy_id": "top_k", "symbol": "AAPL", "date": "2025-01-02", "asset_return": 0.10, "position": 0.5, "cost": 0.001},
            {"strategy_id": "top_k", "symbol": "JPM", "date": "2025-01-02", "asset_return": -0.04, "position": -0.5, "cost": 0.002},
        ]
    )

    result = attribute_strategy_returns(observations, _context())

    assert result["gross_contribution"].sum() == pytest.approx(0.07)
    assert result["costs"].sum() == pytest.approx(0.003)
    assert result["net_contribution"].sum() == pytest.approx(0.067)
    assert set(result["sector"]) == {"Technology", "Financials"}


def test_attach_security_context_rejects_duplicate_dimension_rows():
    duplicated = pd.concat([_context(), _context().iloc[[0]]], ignore_index=True)
    observations = pd.DataFrame([{"symbol": "AAPL", "date": "2024-12-31"}])

    with pytest.raises(ValueError, match="one row per symbol and date"):
        attach_security_context(observations, duplicated, dimensions=["sector"])


def test_attribute_model_scores_rejects_ambiguous_targets():
    scores = pd.DataFrame(
        [{"model_id": "m", "symbol": "AAPL", "date": "2025-01-02", "long_score": 0.5, "short_score": 0.5}]
    )
    labels = pd.DataFrame(
        [
            {"symbol": "AAPL", "date": "2025-01-02", "collapsed_label": "oracle_long"},
            {"symbol": "AAPL", "date": "2025-01-02", "collapsed_label": "oracle_short"},
        ]
    )

    with pytest.raises(ValueError, match="ambiguous long and short"):
        attribute_model_scores(scores, labels, _context())
