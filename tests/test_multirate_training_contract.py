from datetime import datetime

import polars as pl
import pytest

from scripts.train_multirate_mtl import _IndexedTable, _as_datetime
from quant_orchestrator.research_tools.multirate_validation import validate_supervision
from quant_orchestrator.research_tools.streaming_context import StreamingContext
from quant_orchestrator.research_tools.multirate_supervision import StreamingSupervision, instrument_asset_groups


def test_issuer_linkage_does_not_turn_debt_and_preferred_into_options():
    taxonomy = pl.DataFrame({
        'symbol': ['A', 'A_CALL', 'A_NOTE', 'A_PREF'],
        'underlying_symbol': ['A'] * 4,
        'asset_class': ['equity', 'option', 'note_bond', 'preferred'],
    })
    assert instrument_asset_groups(taxonomy) == {
        'equity': {'A'}, 'option': {'A_CALL'},
        'note_bond': {'A_NOTE'}, 'preferred': {'A_PREF'},
    }


def test_instrument_type_cannot_be_guessed_from_issuer_linkage():
    with pytest.raises(ValueError, match='asset_class'):
        instrument_asset_groups(pl.DataFrame({'symbol': ['A_NOTE'], 'underlying_symbol': ['A']}))


def test_equity_events_cannot_satisfy_bond_label_coverage():
    events = pl.DataFrame({
        'symbol': ['A'], 'date': [datetime(2023, 12, 31)],
        'event_date': [datetime(2023, 1, 2)],
        'target_family': ['equity.strategy.oracle_trades'],
        'signal_value': [1.], **{f'text_{i}': [0.] for i in range(7)},
    })
    store = StreamingSupervision(events.lazy())
    with pytest.raises(ValueError, match='note_bond'):
        store.coverage(symbols_by_asset={'equity': {'A'}, 'note_bond': {'A_NOTE'}},
                       required_tasks=('oracle_is_buy',))
    debt_events = events.with_columns(pl.lit('A_NOTE').alias('symbol'))
    store = StreamingSupervision(pl.concat([events, debt_events]).lazy())
    assert store.coverage(symbols_by_asset={'equity': {'A'}, 'note_bond': {'A_NOTE'}},
                          required_tasks=('oracle_is_buy',)) == {
        'equity': {'oracle_is_buy': 1}, 'note_bond': {'oracle_is_buy': 1},
    }


@pytest.mark.parametrize("unit", ["ns", "us", "ms"])
def test_historical_window_cannot_select_future_row(unit):
    frame = pl.DataFrame({"symbol": ["AAPL", "AAPL"],
                          "date": pl.Series([datetime(2021, 1, 4), datetime(2026, 9, 9)], dtype=pl.Datetime(unit)),
                          "value": [1., 999.]})
    values, padding, dates, _ = _IndexedTable(frame, ["value"]).window("AAPL", datetime(2021, 1, 4), 2)
    assert values[~padding].flatten().tolist() == [1.]
    assert [_as_datetime(value) for value in dates] == [datetime(2021, 1, 4)]


def test_equity_labels_do_not_satisfy_option_supervision():
    targets = {("AAPL", datetime(2021, 1, 4)): {"buy": 1.}}
    with pytest.raises(ValueError, match="option"):
        validate_supervision(targets, equity_symbols={"AAPL"}, option_symbols={"OPT_AAPL"}, required_tasks=("buy",))


def test_empty_corpus_cannot_train_trading_heads():
    with pytest.raises(ValueError, match="equity"):
        validate_supervision({}, equity_symbols={"AAPL"}, option_symbols=set(), required_tasks=("buy",))


def test_streaming_context_is_bounded_and_asof(tmp_path):
    path = tmp_path / "history.parquet"
    pl.DataFrame({"symbol": ["AAPL"] * 3 + ["MSFT"],
                  "date": [datetime(2021, 1, 4), datetime(2021, 1, 5), datetime(2026, 9, 9), datetime(2021, 1, 5)],
                  "value": [1., 2., 999., 555.]}).write_parquet(path)
    index = StreamingContext(pl.scan_parquet(path), ["value"])
    values, padding, dates, _ = index.window("AAPL", datetime(2021, 1, 5), 1)
    assert values.tolist() == [[2.]]
    assert not padding.any()
    assert len(dates) == 1
    assert index.version("AAPL", datetime(2021, 1, 5)) == int(dates[0])


def test_normalization_excludes_future_and_reuses_checkpoint():
    from scripts.train_multirate_mtl import _normalization_stats

    history = pl.DataFrame({
        "date": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2025, 1, 1)],
        "signal_value": [1., 3., 999.],
    })
    fitted = _normalization_stats(history, ["signal_value"], cutoff="2021-01-01")
    assert fitted == ([2.], [1.])
    # Inference must reuse training statistics even with a different corpus.
    assert _normalization_stats(history.tail(1), ["signal_value"], saved=fitted) == fitted


def test_normalization_handles_absent_and_nonfinite_observations():
    from scripts.train_multirate_mtl import _normalization_stats

    history = pl.DataFrame({
        "date": [datetime(2020, 1, 1)] * 4,
        "value": [float("inf"), float("nan"), None, 4.],
    })
    assert _normalization_stats(history, ["value"]) == ([4.], [1.])
    assert _normalization_stats(history.head(0), ["value"]) == ([0.], [1.])


def test_checkpoint_normalization_rejects_wrong_dimensions():
    from scripts.train_multirate_mtl import _normalization_stats

    with pytest.raises(ValueError, match="dimensions"):
        _normalization_stats(None, ["value"], saved=([1., 2.], [1., 1.]))


def test_inference_rejects_checkpoint_without_training_preprocessing():
    from scripts.train_multirate_mtl import _validate_inference_checkpoint

    with pytest.raises(ValueError, match="Restore the exact training metadata"):
        _validate_inference_checkpoint({"state_dict": {}, "metrics": {"epoch": 3}})
    checkpoint = {
        "labels": {"family": ["prices"]},
        "configuration": {"d_model": 8},
        "normalization": {rate: ([0.], [1.]) for rate in ("annual", "quarterly", "daily")},
    }
    with pytest.raises(ValueError, match="normalization.sparse"):
        _validate_inference_checkpoint(checkpoint)
    checkpoint["normalization"]["sparse"] = ([0.], [1.])
    _validate_inference_checkpoint(checkpoint)


def test_inference_labels_keep_training_ids_for_subset_and_new_dates():
    from scripts.train_multirate_mtl import _encode_labels

    ids, names, _ = _encode_labels(pl.Series(["MSFT", "NEW"]), ["AAPL", "MSFT"])
    assert ids.tolist() == [1, -100]
    assert names == ["AAPL", "MSFT"]


def test_next_observation_skips_missing_family_and_same_date():
    import torch
    from quant_orchestrator.research_tools.multirate_objectives import next_observation_targets
    values = torch.tensor([[[1.], [2.], [99.], [4.]]])
    valid = torch.tensor([[[True], [True], [False], [True]]])
    target, observed = next_observation_targets(values, valid, torch.tensor([[1, 1, 2, 3]]))
    assert target[0, :2, 0].tolist() == [4., 4.]
    assert observed[0, :, 0].tolist() == [True, True, False, False]


def test_streaming_window_preserves_union_of_same_date_families(tmp_path):
    path = tmp_path / 'families.parquet'
    pl.DataFrame({'symbol': ['A', 'A'], 'date': [datetime(2020, 1, 1)] * 2,
                  'balance': [1., None], 'income': [None, 2.]}).write_parquet(path)
    values, padding, _, _ = StreamingContext(pl.scan_parquet(path), ['balance', 'income']).window('A', datetime(2020, 1, 1), 1)
    assert values.tolist() == [[1., 2.]]
    assert not padding.any()


def test_lazy_samples_release_materialized_arrays_after_batch():
    from scripts.train_multirate_mtl import _LazySample
    calls = []
    def load():
        calls.append(1)
        return {'daily': [1, 2, 3]}
    sample = _LazySample({'symbol': 'A'}, load)
    assert sample['daily'] == [1, 2, 3]
    sample.release()
    assert list(sample) == ['symbol']
    assert sample['daily'] == [1, 2, 3]
    assert len(calls) == 2


def test_streamed_supervision_respects_availability_and_exact_event_date():
    from quant_orchestrator.research_tools.multirate_supervision import StreamingSupervision
    events = pl.DataFrame([
        {'symbol': 'A', 'date': datetime(2023, 12, 31), 'event_date': datetime(2023, 1, 2),
         'target_family': 'equity.strategy.oracle_trades', 'signal_value': 1.,
         **{f'text_{i}': 0. for i in range(7)}},
        {'symbol': 'A', 'date': datetime(2024, 12, 31), 'event_date': datetime(2023, 1, 3),
         'target_family': 'equity.strategy.oracle_trades', 'signal_value': 999.,
         **{f'text_{i}': 0. for i in range(7)}},
    ])
    store = StreamingSupervision(events.lazy(), cutoff=datetime(2024, 1, 1))
    assert store.get(('A', datetime(2023, 1, 2)))['oracle_is_buy'] == 1.
    assert store.get(('A', datetime(2023, 1, 3))) == {}
    assert store.get(('A', datetime(2023, 1, 4))) == {}
    assert store.anchors().height == 1


def test_batched_transformer_engine_mask_preserves_individual_dates():
    import torch
    from quant_orchestrator.platforms.ml_frameworks.torch.models.transformers.multirate.transformer_engine import _blocked_mask
    mask = torch.tensor([[[False, True], [False, False]], [[False, False], [False, False]]])
    blocked = _blocked_mask(mask, None, batch_size=2, query_length=2, key_length=2, device=torch.device('cpu'))
    assert blocked.shape == (2, 1, 2, 2)
    assert bool(blocked[0, 0, 0, 1])
    assert not bool(blocked[1, 0, 0, 1])


def test_sparse_channels_fit_independent_scales_without_future_values():
    from scripts.train_multirate_mtl import _normalization_stats
    frame = pl.DataFrame({'date': [datetime(2023, 1, 1), datetime(2023, 1, 2), datetime(2025, 1, 1)],
                          'insider__text_0': [1e6, 3e6, 1e12], 'hits__text_0': [0.1, 0.3, 0.9]})
    mean, scale = _normalization_stats(frame, ['insider__text_0','hits__text_0'], cutoff='2024-01-01')
    assert mean == pytest.approx([2e6, .2])
    assert scale == pytest.approx([1e6, .1])
    assert [(v-m)/s for v,m,s in zip([3e6,.3],mean,scale)] == pytest.approx([1.,1.])
