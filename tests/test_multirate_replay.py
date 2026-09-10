from datetime import datetime
import polars as pl
import pytest
from quant_orchestrator.platforms.backtesting_frameworks.multirate_replay import replay_multirate


class Warehouse:
    backend = None

    def __init__(self, prices, distributions):
        self.prices = prices
        self.distributions = distributions
        self.price_adjustments = []

    def read_prices(self, symbol, **kwargs):
        self.price_adjustments.append(kwargs["adjustment"])
        return self.prices

    def read_fundamentals(self, symbol, **kwargs):
        if kwargs.get("section") == "historical_splits":
            return pl.DataFrame()
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


@pytest.mark.parametrize("session_days", [(2, 3, 4), (5, 8, 9)])
def test_previous_session_execution_and_expiry_intrinsic(tmp_path, monkeypatch, session_days):
    root = tmp_path / "corpus"
    root.mkdir()
    dates = [datetime(2024, 1, d) for d in session_days]
    pl.DataFrame(
        [
            dict(symbol="A", issuer="A", underlying_symbol="A", asset_class="equity"),
            dict(
                symbol="AC",
                issuer="A",
                underlying_symbol="A",
                asset_class="option",
                expiration=dates[-1].strftime("%Y-%m-%d"),
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
        start=dates[0].strftime("%Y-%m-%d"),
        end=dates[-1].strftime("%Y-%m-%d"),
        initial_cash=1000,
        fee_bps=0,
        slippage_bps=0,
        warehouse=warehouse,
    )
    assert result["final_equity"] == pytest.approx(6100.0)
    assert warehouse.price_adjustments == ["splits_and_dividends", "unadjusted"]
    trades = pl.read_parquet(tmp_path / "replay/trade_list.parquet")
    assert trades.height == 1
    assert trades["entry_date"][0] == dates[1]
    assert trades["exit_price"][0] == 20.0


def test_adjusted_prices_do_not_double_count_cash_distributions(tmp_path):
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
    assert result["final_equity"] == pytest.approx(1000.0)
    curve = pl.read_parquet(tmp_path / "replay/equity_curve.parquet")
    assert curve["receivables"].to_list() == [0.0, 0.0, 0.0, 0.0]
    assert warehouse.price_adjustments == ["splits_and_dividends"]
