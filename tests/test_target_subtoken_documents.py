import pandas as pd

from quant_orchestrator.platforms.ml_frameworks.torch import (
    build_analyst_rating_subtoken_documents,
    build_earnings_report_subtoken_documents,
    build_ownership_insider_trading_subtoken_documents,
)


def test_insider_events_same_day_remain_distinct_subtokens():
    frame = pd.DataFrame({
        "symbol": ["AAPL", "AAPL", "AAPL"],
        "event_date": ["2024-01-02", "2024-01-02", "2024-01-03"],
        "owner_name": ["ONE", "TWO", "ONE"],
        "transaction_shares": [100.0, 200.0, 50.0],
        "transaction_price": [10.0, 11.0, 12.0],
        "transaction_value": [1000.0, 2200.0, 600.0],
    })
    corpus = build_ownership_insider_trading_subtoken_documents(
        frame,
        {"equity.ownership.insider_trading": (
            "owner_name", "transaction_shares", "transaction_price", "transaction_value",
        )},
    )

    assert len(corpus.documents) == 1
    assert len(corpus.subtokens) == 3
    assert len(corpus.document_subtokens) == 3
    assert corpus.subtokens["event_id"].is_unique
    assert corpus.prototype_targets.loc[0, "subtoken_count"] == 3


def test_earnings_endpoint_rows_are_sparse_subtokens():
    frame = pd.DataFrame({
        "symbol": ["AAPL", "AAPL"],
        "report_date": ["2024-01-02", "2024-04-02"],
        "eps_consensus": [1.0, 1.1],
        "eps_actual": [1.2, 0.9],
        "revenue_consensus": [1000.0, 1100.0],
        "revenue_actual": [1020.0, 1080.0],
    })
    corpus = build_earnings_report_subtoken_documents(
        frame,
        {"equity.calendar.earnings": (
            "eps_consensus", "eps_actual", "revenue_consensus", "revenue_actual",
        )},
    )

    assert len(corpus.documents) == 1
    assert len(corpus.subtokens) == 2
    assert set(corpus.subtokens["event_role"]) == {"happened"}
    assert corpus.subtokens["availability_timestamp"].equals(corpus.subtokens["event_timestamp"])


def test_analyst_rating_endpoint_preserves_each_row():
    frame = pd.DataFrame({
        "symbol": ["AAPL", "AAPL"],
        "published_date": ["2024-01-02 10:00:00", "2024-01-02 11:00:00"],
        "analyst_name": ["Analyst A", "Analyst B"],
        "analyst_firm": ["Firm A", "Firm B"],
        "price_target": [200.0, 210.0],
        "news_title": ["target raised", "target cut"],
    })
    corpus = build_analyst_rating_subtoken_documents(
        frame,
        {"equity.estimates.price_target": (
            "analyst_name", "analyst_firm", "price_target", "news_title",
        )},
    )

    assert len(corpus.documents) == 1
    assert len(corpus.subtokens) == 2
    assert corpus.subtokens["news_title"].tolist() == ["target raised", "target cut"]
