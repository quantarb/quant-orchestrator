import pandas as pd
import torch

from quant_orchestrator.platforms.ml_frameworks.torch import (
    build_1t_subtoken_documents,
    mean_document_embeddings,
)


def test_documents_are_symbol_family_and_annual_subtoken_is_not_duplicated():
    frame = pd.DataFrame({
        "symbol": ["AAPL", "AAPL", "AAPL", "AAPL"],
        "timestamp": [
            "2024-01-02 09:30:00+00:00",
            "2024-01-02 09:31:00+00:00",
            "2024-01-03 09:30:00+00:00",
            "2024-04-01 00:00:00+00:00",
        ],
        "feature_family": ["prices", "prices", "prices", "fundamentals"],
        "close": [100.0, 100.1, 100.2, None],
        "employee_count": [None, None, None, 10000.0],
    })
    corpus = build_1t_subtoken_documents(
        frame, {"prices": ("close",), "fundamentals": ("employee_count",)}
    )
    assert len(corpus.documents) == 2
    assert set(map(tuple, corpus.documents[["symbol", "feature_family"]].to_numpy())) == {
        ("AAPL", "prices"), ("AAPL", "fundamentals")
    }
    assert len(corpus.subtokens) == 4
    assert len(corpus.document_subtokens) == 4


def test_mean_document_embeddings_computes_family_prototypes():
    embeddings = torch.tensor([[1.0, 3.0], [3.0, 5.0], [10.0, 14.0]])
    document_ids = torch.tensor([0, 0, 1])
    prototypes = mean_document_embeddings(embeddings, document_ids)
    assert torch.equal(prototypes, torch.tensor([[2.0, 4.0], [10.0, 14.0]]))
