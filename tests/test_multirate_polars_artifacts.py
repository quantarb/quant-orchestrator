from datetime import datetime
import polars as pl
import pytest
from quant_orchestrator.artifact_contracts import StrategyArtifactBundle, write_strategy_artifacts


def test_polars_trade_artifact_never_uses_pandas_bridge(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Pandas bridge used")

    monkeypatch.setattr(pl.DataFrame, "to_pandas", forbidden)
    trades = pl.DataFrame(
        {
            "trade_id": [1],
            "symbol": ["DUKB"],
            "side": ["long"],
            "entry_date": [datetime(2024, 1, 2)],
            "exit_date": [datetime(2024, 1, 3)],
        }
    )
    paths = write_strategy_artifacts(StrategyArtifactBundle(trade_list=trades), tmp_path)
    assert pl.read_parquet(paths["trade_list"])["symbol"].to_list() == ["DUKB"]
    with pytest.raises(ValueError, match="entry/exit"):
        write_strategy_artifacts(
            StrategyArtifactBundle(
                trade_list=trades.with_columns(pl.lit(datetime(2020, 1, 1)).alias("exit_date"))
            ),
            tmp_path,
        )
