from __future__ import annotations

import pandas as pd
import pytest
import json

from quant_orchestrator.platforms.backtesting_frameworks.scored_panel_replay import (
    ScoredPanelTopKReplayConfig,
    replay_scored_panel_top_k,
)
from quant_orchestrator.platforms.backtesting_frameworks.strategy_artifacts import (
    StrategyArtifactBundle,
    combine_trade_lists,
    normalize_trade_list,
    read_trade_list_artifact,
    read_strategy_artifacts,
    validate_strategy_artifact_frame,
    write_trade_list_artifact,
    write_strategy_artifacts,
)


def test_strategy_artifact_bundle_round_trips_core_contract(tmp_path) -> None:
    feature_panel = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-01")],
            "symbol": ["aaa"],
            "feature": [1.0],
        }
    )
    scored_panel = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-01")],
            "symbol": ["aaa"],
            "close": [100.0],
            "prob_buy": [0.9],
        }
    )
    action_tape = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-02")],
            "symbol": ["aaa"],
            "action": ["BUY"],
            "price": [101.0],
        }
    )
    trade_windows = pd.DataFrame(
        {
            "trade_id": ["t1"],
            "symbol": ["aaa"],
            "side": ["long"],
            "entry_date": [pd.Timestamp("2024-01-02")],
            "exit_date": [pd.Timestamp("2024-01-03")],
        }
    )

    legacy_scored_path = tmp_path / "legacy_scored.csv"
    legacy_scored_path.write_text("legacy", encoding="utf-8")
    paths = write_strategy_artifacts(
        StrategyArtifactBundle(
            feature_panel=feature_panel,
            scored_panel=scored_panel,
            action_tape=action_tape,
            trade_windows=trade_windows,
            summary={"total_return_pct": 1.0},
            strategy_name="test_strategy",
        ),
        tmp_path,
        extra_paths={"scored_panel": legacy_scored_path},
    )
    loaded = read_strategy_artifacts(tmp_path)
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))

    assert paths["manifest"].exists()
    assert manifest["artifacts"]["scored_panel"]["rows"] == 1
    assert manifest["artifacts"]["trade_list"]["alias_of"] == "trade_windows"
    assert manifest["artifacts"]["trade_list"]["path"] == "trade_windows.parquet"
    assert manifest["artifacts"]["legacy_scored_panel"]["path"] == str(legacy_scored_path)
    assert loaded.strategy_name == "test_strategy"
    assert loaded.summary == {"total_return_pct": 1.0}
    assert loaded.manifest_path == tmp_path / "strategy_artifacts_manifest.json"
    assert loaded.trade_windows_path == tmp_path / "trade_windows.parquet"
    assert loaded.trade_list_path == tmp_path / "trade_windows.parquet"
    assert loaded.scored_panel["symbol"].tolist() == ["AAA"]
    assert loaded.action_tape["action"].tolist() == ["buy"]
    assert loaded.trade_windows["entry_date"].tolist() == [pd.Timestamp("2024-01-02")]
    assert loaded.trade_list is loaded.trade_windows


def test_strategy_artifact_validation_rejects_missing_required_columns() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        validate_strategy_artifact_frame(
            "scored_panel",
            pd.DataFrame({"date": [pd.Timestamp("2024-01-01")], "symbol": ["AAA"]}),
            required_columns=("date", "symbol", "close"),
        )


def test_trade_window_validation_rejects_invalid_contract_values() -> None:
    base = pd.DataFrame(
        {
            "trade_id": ["t1"],
            "symbol": ["AAA"],
            "side": ["long"],
            "entry_date": [pd.Timestamp("2024-01-02")],
            "exit_date": [pd.Timestamp("2024-01-10")],
            "equity_entry_notional": [100.0],
        }
    )

    with pytest.raises(ValueError, match="invalid side"):
        validate_strategy_artifact_frame(
            "trade_windows",
            base.assign(side=["buy"]),
            required_columns=("trade_id", "symbol", "side", "entry_date", "exit_date"),
        )
    with pytest.raises(ValueError, match="exit_date before entry_date"):
        validate_strategy_artifact_frame(
            "trade_windows",
            base.assign(exit_date=[pd.Timestamp("2024-01-01")]),
            required_columns=("trade_id", "symbol", "side", "entry_date", "exit_date"),
        )
    with pytest.raises(ValueError, match="negative equity_entry_notional"):
        validate_strategy_artifact_frame(
            "trade_windows",
            base.assign(equity_entry_notional=[-1.0]),
            required_columns=("trade_id", "symbol", "side", "entry_date", "exit_date"),
        )


def test_scored_panel_top_k_replay_writes_strategy_contract(tmp_path) -> None:
    panel = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"],
            "symbol": ["AAA", "BBB", "AAA", "BBB", "AAA", "BBB"],
            "close": [100.0, 100.0, 110.0, 100.0, 120.0, 90.0],
            "signal": [0.9, 0.4, 0.3, 0.8, 0.3, 0.8],
        }
    )

    result = replay_scored_panel_top_k(
        panel,
        config=ScoredPanelTopKReplayConfig(
            score_col="signal",
            top_k=1,
            threshold=0.5,
            fee_bps=0.0,
            slippage_bps=0.0,
            strategy_name="unit_signal",
            output_dir=tmp_path,
        ),
    )
    loaded = read_strategy_artifacts(tmp_path)

    assert result.rule_replay.trade_windows.loc[0, "symbol"] == "AAA"
    assert result.rule_replay.trade_windows.loc[0, "ret_dec"] == pytest.approx((120.0 / 110.0) - 1.0)
    assert loaded.strategy_name == "unit_signal"
    assert loaded.scored_panel is not None
    assert loaded.action_tape is not None
    assert loaded.trade_windows is not None


def test_trade_list_helpers_load_manifest_direct_file_and_mixed_sources(tmp_path) -> None:
    trades = pd.DataFrame(
        {
            "trade_id": ["b", "a"],
            "symbol": ["msft", "aapl"],
            "side": ["LONG", "short"],
            "entry_date": ["2024-01-03", "2024-01-02"],
            "exit_date": ["2024-01-10", "2024-01-09"],
            "equity_entry_notional": [200.0, 100.0],
        }
    )

    paths = write_trade_list_artifact(
        trades,
        tmp_path / "manifest_source",
        strategy_name="unit_trades",
        summary={"producer": "unit"},
    )
    direct_path = tmp_path / "direct.parquet"
    normalize_trade_list(trades).to_parquet(direct_path, index=False)

    manifest_trades = read_trade_list_artifact(tmp_path / "manifest_source")
    direct_trades = read_trade_list_artifact(direct_path)
    combined = combine_trade_lists(
        {
            "manifest": tmp_path / "manifest_source",
            "direct": direct_path,
            "memory": trades,
        }
    )

    assert paths["trade_windows"].name == "trade_windows.parquet"
    assert manifest_trades["symbol"].tolist() == ["AAPL", "MSFT"]
    assert direct_trades["side"].tolist() == ["short", "long"]
    assert combined["artifact_source"].value_counts().to_dict() == {
        "direct": 2,
        "manifest": 2,
        "memory": 2,
    }
