from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "train_multirate_mtl.py"
SPEC = importlib.util.spec_from_file_location("train_multirate_mtl", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_issuer_dte_selection_aggregates_contract_features_into_synthetic_rows() -> None:
    panel = pd.DataFrame(
        [
            {
                "symbol": "AAPL_C_100",
                "contract_symbol": "AAPL_C_100",
                "underlying_symbol": "AAPL",
                "entry_date": "2025-01-02",
                "option_type": "call",
                "side": "long",
                "dte": 30,
                "strike": 100.0,
                "entry_bid": 2.0,
                "entry_ask": 2.2,
                "execution_return": 0.10,
            },
            {
                "symbol": "AAPL_C_105",
                "contract_symbol": "AAPL_C_105",
                "underlying_symbol": "AAPL",
                "entry_date": "2025-01-02",
                "option_type": "call",
                "side": "long",
                "dte": 30,
                "strike": 105.0,
                "entry_bid": 3.0,
                "entry_ask": 3.2,
                "execution_return": 0.20,
            },
            {
                "symbol": "AAPL_P_100",
                "contract_symbol": "AAPL_P_100",
                "underlying_symbol": "AAPL",
                "entry_date": "2025-01-02",
                "option_type": "put",
                "side": "short",
                "dte": 30,
                "strike": 100.0,
                "entry_bid": 2.5,
                "entry_ask": 2.7,
                "execution_return": -0.10,
            },
        ]
    )
    taxonomy = pd.DataFrame({"issuer": ["AAPL"]}, index=pd.Index(["AAPL"], name="symbol"))

    result = MODULE._issuer_dte_bin_option_panel(panel, taxonomy, bin_count=1)

    assert len(result) == 2
    call = result.loc[result["option_type"].eq("call")].iloc[0]
    assert call["symbol"] == "OPT_SYNTH_AAPL_C_DTE30"
    assert call["contract_symbol"] == call["symbol"]
    assert bool(call["synthetic_option"])
    assert call["synthetic_contract_count"] == 2
    assert call["strike"] == pytest.approx(102.5)
    assert call["entry_bid"] == pytest.approx(2.5)
    assert call["execution_return"] == pytest.approx(0.15)


def test_issuer_dte_selection_uses_liquidity_weighted_bid_ask() -> None:
    panel = pd.DataFrame(
        [
            {
                "symbol": "AAPL_C_100", "contract_symbol": "AAPL_C_100",
                "underlying_symbol": "AAPL", "entry_date": "2025-01-02",
                "option_type": "call", "side": "long", "dte": 30,
                "entry_bid": 1.0, "entry_ask": 2.0, "volume": 1.0,
            },
            {
                "symbol": "AAPL_C_105", "contract_symbol": "AAPL_C_105",
                "underlying_symbol": "AAPL", "entry_date": "2025-01-02",
                "option_type": "call", "side": "long", "dte": 30,
                "entry_bid": 3.0, "entry_ask": 4.0, "volume": 3.0,
            },
        ]
    )
    taxonomy = pd.DataFrame({"issuer": ["AAPL"]}, index=pd.Index(["AAPL"], name="symbol"))

    result = MODULE._issuer_dte_bin_option_panel(panel, taxonomy, bin_count=1)
    call = result.iloc[0]

    assert call["entry_bid"] == pytest.approx(2.5)
    assert call["entry_ask"] == pytest.approx(3.5)
