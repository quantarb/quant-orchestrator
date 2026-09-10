from datetime import datetime
import polars as pl
import pytest
from quant_orchestrator.platforms.backtesting_frameworks.multirate_replay import replay_multirate


class Warehouse:
    backend = None

    def __init__(self, prices, distributions):
        self.prices = prices
        self.distributions = distributions

    def read_prices(self, symbol, **kwargs):
        return self.prices

    def read_fundamentals(self, symbol, **kwargs):
        return self.distributions


def scores(path, symbols, dates, directions):
    rows = []
    for symbol in symbols:
        for date, direction in zip(dates, directions):
            rows.append(
                dict(
                    symbol=symbol,
                    date=date,
                    oracle_is_buy=direction,
                    oracle_is_short=1 - direction,
                    oracle_is_sell=0.0,
                    hits_long_return_hub=0.9 if symbol == "AC" else 0.1,
                )
            )
    pl.DataFrame(rows).write_csv(path)


def test_previous_session_execution_and_expiry_intrinsic(tmp_path, monkeypatch):
    root = tmp_path / "corpus"
    root.mkdir()
    dates = [datetime(2024, 1, d) for d in [2, 3, 4]]
    pl.DataFrame(
        [
            dict(symbol="A", issuer="A", underlying_symbol="A", asset_class="equity"),
            dict(
                symbol="AC",
                issuer="A",
                underlying_symbol="A",
                asset_class="option",
                expiration="2024-01-04",
                strike=100.0,
                option_type="call",
            ),
        ],
        infer_schema_length=None,
    ).write_csv(root / "taxonomy.csv")
    q = pl.DataFrame(
        {
            "snapshot_date": dates,
            "contract_symbol": ["AC"] * 3,
            "bid": [1.0, 2.0, 5.0],
            "ask": [2.0, 3.0, 6.0],
        }
    )
    monkeypatch.setattr(
        "quant_warehouse.platforms.data_providers.thetadata.options.read_thetadata_eod_option_chain",
        lambda *args, **kwargs: q,
    )
    warehouse = Warehouse(
        pl.DataFrame({"date": dates, "close": [100.0, 110.0, 120.0]}), pl.DataFrame()
    )
    score_path = tmp_path / "scores.csv"
    scores(score_path, ["A", "AC"], dates, [0.9, 0.9, 0.9])
    result = replay_multirate(
        root,
        score_path,
        tmp_path / "replay",
        start="2024-01-02",
        end="2024-01-04",
        initial_cash=1000,
        fee_bps=0,
        slippage_bps=0,
        warehouse=warehouse,
    )
    assert result["final_equity"] == pytest.approx(6100.0)
    trades = pl.read_parquet(tmp_path / "replay/trade_list.parquet")
    assert trades.height == 1
    assert trades["entry_date"][0] == datetime(2024, 1, 3)
    assert trades["exit_price"][0] == 20.0


def test_distribution_entitlement_survives_sale_before_payment(tmp_path):
    root = tmp_path / "corpus"
    root.mkdir()
    dates = [datetime(2024, 1, d) for d in [2, 3, 4, 5]]
    pl.DataFrame(
        {"symbol": ["A"], "issuer": ["A"], "underlying_symbol": ["A"], "asset_class": ["equity"]}
    ).write_csv(root / "taxonomy.csv")
    distribution = pl.DataFrame(
        {"ex_dividend_date": [dates[2]], "payment_date": [dates[3]], "amount": [1.0]}
    )
    warehouse = Warehouse(pl.DataFrame({"date": dates, "close": [100.0] * 4}), distribution)
    score_path = tmp_path / "scores.csv"
    scores(score_path, ["A"], dates, [0.9, 0.1, 0.1, 0.1])
    result = replay_multirate(
        root,
        score_path,
        tmp_path / "replay",
        start="2024-01-02",
        end="2024-01-05",
        initial_cash=1000,
        fee_bps=0,
        slippage_bps=0,
        warehouse=warehouse,
    )
    assert result["final_equity"] == pytest.approx(1010.0)
    curve = pl.read_parquet(tmp_path / "replay/equity_curve.parquet")
    assert curve["receivables"].to_list() == [0.0, 0.0, 10.0, 0.0]
