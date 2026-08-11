import pytest
import torch

from quant_orchestrator.platforms.ml_frameworks.torch import (
    AutoFeatureEngineer,
    MultiRatePredictionTaskSpec,
    MultiRateTaskSpec,
    MultiRateTransformer,
    MultiRateTransformerConfig,
    IssuerContextCache,
    build_attention_mask,
)
from quant_orchestrator.platforms.ml_frameworks.torch.models.transformers.multirate import (
    CoverageAwareInput,
    MultiRateTransformer as NestedMultiRateTransformer,
    TEMPORAL_MTL_TASK_NAMES,
    add_subtoken_temporal_tasks,
    SUBTOKEN_PREDICTION_TASK_NAMES,
    TOKEN_PREDICTION_TASK_NAMES,
)


def _tasks():
    return (
        MultiRateTaskSpec("daily_hits", "token", output_dim=6),
        MultiRateTaskSpec("sector", "document", output_dim=4, source="fused"),
        MultiRateTaskSpec("annual_context", "document", output_dim=2, source="annual"),
    )


def test_subtoken_temporal_task_factory_defines_exact_contract():
    labels = {name: ("a", "b") for name in ("issuer", "symbol", "industry", "sector", "subsector", "date")}
    bundle = add_subtoken_temporal_tasks(({"row": 1},), ("feature_family", "target_family"), labels)
    assert bundle.task_names == TEMPORAL_MTL_TASK_NAMES
    assert len(bundle.corpus) == 1
    assert len(bundle.document_tasks) == 7
    assert len(bundle.supervised_tasks) == 22
    assert len(bundle.prediction_tasks) == 16
    assert {task.level for task in bundle.prediction_tasks if task.task_name in SUBTOKEN_PREDICTION_TASK_NAMES} == {"subtoken"}
    assert {task.level for task in bundle.prediction_tasks if task.task_name in TOKEN_PREDICTION_TASK_NAMES} == {"token"}


def test_temporal_mask_blocks_future_and_cross_sectional_mask_shares_dates():
    temporal = build_attention_mask(torch.tensor([0, 1, 2]), mode="temporal")
    assert torch.isfinite(temporal[2, 0])
    assert torch.isneginf(temporal[0, 1])

    cross_sectional = build_attention_mask(torch.tensor([10, 10, 11]), mode="cross_sectional")
    assert torch.isfinite(cross_sectional[0, 1])
    assert torch.isneginf(cross_sectional[0, 2])
    assert torch.isfinite(cross_sectional[2, 0])


def test_multirate_model_is_owned_by_nested_transformer_module():
    assert MultiRateTransformer is NestedMultiRateTransformer


@pytest.mark.parametrize("backbone", ["encoder_only", "decoder_only", "encoder_decoder"])
@pytest.mark.parametrize("attention_mode", ["temporal", "cross_sectional"])
def test_all_backbones_return_token_and_document_tasks(backbone, attention_mode):
    config = MultiRateTransformerConfig(
        backbone=backbone,
        d_model=16,
        num_heads=4,
        layers=1,
        document_pool="mean",
    )
    model = MultiRateTransformer(
        {"annual": 3, "quarterly": 4, "daily": 5},
        config=config,
        tasks=_tasks(),
    )
    output = model(
        torch.randn(2, 5, 5),
        torch.randn(2, 2, 3),
        torch.randn(2, 3, 4),
        attention_mode=attention_mode,
        daily_dates=torch.tensor([0, 0, 1, 1, 2]),
    )
    assert output["token_states"].shape == (2, 5, 16)
    assert output["document_state"].shape == (2, 16)
    assert output["token_outputs"]["daily_hits"].shape == (2, 5, 6)
    assert output["document_outputs"]["sector"].shape == (2, 4)
    assert output["document_outputs"]["annual_context"].shape == (2, 2)


def test_padding_mask_is_respected_by_document_pooling():
    config = MultiRateTransformerConfig(d_model=16, num_heads=4, layers=1, document_pool="last")
    model = MultiRateTransformer(
        {"annual": 2, "quarterly": 2, "daily": 2}, config=config, tasks=_tasks()[:1]
    )
    output = model(
        torch.randn(1, 3, 2), torch.randn(1, 1, 2), torch.randn(1, 1, 2),
        daily_padding_mask=torch.tensor([[False, True, True]]),
    )
    assert output["document_state"].shape == (1, 16)
    assert torch.isfinite(output["document_state"]).all()


def test_coverage_input_distinguishes_missing_from_observed_zero():
    torch.manual_seed(7)
    layer = CoverageAwareInput({"prices": 2, "fundamentals": 1}, 8)
    values = torch.tensor([[[0.0, 0.0, 0.0]]])
    observed = layer(values, family_presence=torch.tensor([[[1.0, 0.0]]]))
    missing = layer(values, family_presence=torch.tensor([[[0.0, 1.0]]]))
    assert not torch.equal(observed, missing)


def test_model_accepts_family_presence_and_nan_imputation_path():
    model = MultiRateTransformer(
        {"annual": 3, "quarterly": 3, "daily": 3},
        config=MultiRateTransformerConfig(d_model=12, num_heads=3, layers=1),
        feature_families={
            "annual": {"fundamentals": 2, "prices": 1},
            "quarterly": {"fundamentals": 2, "prices": 1},
            "daily": {"fundamentals": 2, "prices": 1},
        },
        tasks=_tasks()[:1],
    )
    output = model(
        torch.tensor([[[float("nan"), 1.0, 0.0]]]),
        torch.zeros(1, 1, 3),
        torch.zeros(1, 1, 3),
        daily_family_presence=torch.tensor([[[0.0, 1.0]]]),
    )
    assert torch.isfinite(output["token_states"]).all()


def test_four_rate_cache_reuses_all_intermediate_states():
    config = MultiRateTransformerConfig(
        backbone="encoder_decoder", rates=("annual", "quarterly", "daily", "sparse"),
        d_model=12, num_heads=3, layers=1, dropout=0.0,
    )
    model = MultiRateTransformer(
        {rate: 2 for rate in config.rates}, config=config, tasks=(),
    ).eval()
    values = {rate: torch.randn(1, length, 2) for rate, length in {
        "annual": 2, "quarterly": 3, "daily": 4, "sparse": 2,
    }.items()}
    with torch.no_grad():
        first = model(values["daily"], values["annual"], values["quarterly"], values["sparse"], compute_document_outputs=False)
    assert set(first["rate_cache"]) == set(config.rates)

    original_project = model._project
    model._project = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("cached rate was projected again"))
    try:
        with torch.no_grad():
            second = model(
                values["daily"], values["annual"], values["quarterly"], values["sparse"],
                rate_cache=first["rate_cache"], compute_document_outputs=False,
            )
    finally:
        model._project = original_project
    assert torch.allclose(first["rate_states"]["daily"], second["rate_states"]["daily"])
    assert torch.allclose(first["rate_states"]["annual"], second["rate_states"]["annual"])
    assert torch.allclose(first["rate_states"]["quarterly"], second["rate_states"]["quarterly"])
    assert torch.allclose(first["rate_states"]["sparse"], second["rate_states"]["sparse"])


def test_real_1t_corpus_builds_and_reuses_multirate_cache():
    from pathlib import Path
    import pandas as pd

    root = Path(__file__).parents[1] / "artifacts" / "multi-rate-mtl" / "inputs" / "1T"
    paths = {rate: root / f"{rate}.parquet" for rate in ("daily", "annual", "quarterly")}
    if not all(path.exists() for path in paths.values()):
        pytest.skip("local 1T corpus is not available")
    frames = {rate: pd.read_parquet(path) for rate, path in paths.items()}
    common = set(frames["daily"].symbol.astype(str)) & set(frames["annual"].symbol.astype(str)) & set(frames["quarterly"].symbol.astype(str))
    symbol = sorted(common)[0]
    selected = [column for column in frames["daily"].columns if column.startswith("value__")][:8]
    prediction_date = min(pd.to_datetime(frames[rate].loc[frames[rate].symbol.eq(symbol), "date"]).max() for rate in frames)
    arrays = {}
    lengths = {"daily": 8, "annual": 3, "quarterly": 4}
    for rate, frame in frames.items():
        rows = frame.loc[frame.symbol.eq(symbol) & pd.to_datetime(frame.date).le(prediction_date)].sort_values("date").tail(lengths[rate])
        arrays[rate] = torch.tensor(rows[selected].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(), dtype=torch.float32).unsqueeze(0)
    config = MultiRateTransformerConfig(
        backbone="encoder_decoder", rates=("annual", "quarterly", "daily"),
        d_model=16, num_heads=4, layers=1, dropout=0.0,
    )
    model = MultiRateTransformer({rate: len(selected) for rate in config.rates}, config=config, tasks=()).eval()
    with torch.no_grad():
        first = model(arrays["daily"], arrays["annual"], arrays["quarterly"], compute_document_outputs=False)
        second = model(
            arrays["daily"], arrays["annual"], arrays["quarterly"],
            rate_cache=first["rate_cache"], compute_document_outputs=False,
        )
    assert set(first["rate_cache"]) == {"annual", "quarterly", "daily"}
    assert torch.isfinite(second["token_states"]).all()
    assert torch.allclose(first["rate_states"]["daily"], second["rate_states"]["daily"])


def test_fixed_32_call_put_universe_reuses_issuer_states_for_real_daily_documents():
    """Exercise fixed option instruments against real issuer rate documents."""
    from pathlib import Path
    import pandas as pd

    corpus_root = Path(__file__).parents[1] / "artifacts" / "multi-rate-mtl" / "inputs" / "1T"
    option_path = Path(__file__).parents[1] / "notebooks" / "ml_trading" / "mlruns" / "2" / "12019faf781b4160a92d52490d68f806" / "artifacts" / "option_panel" / "option_candidate_panel.parquet"
    required = [corpus_root / f"{rate}.parquet" for rate in ("daily", "annual", "quarterly")]
    if not option_path.exists() or not all(path.exists() for path in required):
        pytest.skip("real option/1T corpus is not available")

    options = pd.read_parquet(option_path)
    options["entry_date"] = pd.to_datetime(options["entry_date"])
    options["expiration"] = pd.to_datetime(options["expiration"])
    # The source panel has calls and puts on different entry dates. Select
    # backwards from the newest date that has at least 32 contracts for each
    # side, then freeze that universe for the backward-running experiment.
    frames = {rate: pd.read_parquet(corpus_root / f"{rate}.parquet") for rate in ("daily", "annual", "quarterly")}
    coverage = {}
    for symbol in options.symbol.astype(str).str.upper().unique():
        counts = {}
        for rate, frame in frames.items():
            rows = frame.loc[frame.symbol.astype(str).str.upper().eq(symbol)]
            counts[rate] = int(rows.shape[0])
        # Daily coverage dominates, but all three streams must exist. This
        # prevents selecting a high-DTE contract whose issuer history is thin.
        coverage[symbol] = min(counts["daily"] / 100.0, counts["quarterly"] / 4.0, counts["annual"] / 2.0)

    opening = options.loc[
        options.entry_date.ge("2026-01-01")
        & options.option_type.isin(("call", "put"))
    ].copy()
    selected_parts = []
    for option_type in ("call", "put"):
        side = opening.loc[opening.option_type.eq(option_type)]
        eligible_dates = (
            side.groupby("entry_date").size().loc[lambda values: values >= 32]
        )
        selection_date = eligible_dates.index.max()
        dated = side.loc[side.entry_date.eq(selection_date)].copy()
        dated["symbol_coverage"] = dated.symbol.astype(str).str.upper().map(coverage).fillna(0.0)
        symbol_counts = dated.groupby("symbol").size()
        eligible_symbols = symbol_counts.loc[symbol_counts >= 32].index
        if len(eligible_symbols):
            dated = dated.loc[dated.symbol.isin(eligible_symbols)]
        chosen_symbol = (
            dated.groupby("symbol", as_index=False)["symbol_coverage"]
            .max().sort_values(["symbol_coverage", "symbol"], ascending=[False, True])
            .iloc[0]["symbol"]
        )
        selected_parts.append(
            dated.loc[dated.symbol.eq(chosen_symbol)]
            .sort_values(["volume", "contract_symbol"], ascending=[False, True])
            .head(32)
        )
    fixed = pd.concat(selected_parts, ignore_index=True)
    assert len(fixed) == 64
    assert fixed.contract_symbol.nunique() == 64
    assert fixed.groupby("option_type").size().to_dict() == {"call": 32, "put": 32}

    prediction_date = pd.Timestamp("2026-04-10")
    value_columns = [column for column in frames["daily"].columns if column.startswith("value__")][:8]
    arrays_by_issuer = {}
    for issuer in fixed.symbol.astype(str).str.upper().unique():
        arrays_by_issuer[issuer] = {}
        for rate, length in (("daily", 8), ("annual", 3), ("quarterly", 4)):
            frame = frames[rate]
            rows = frame.loc[
                frame.symbol.astype(str).str.upper().eq(issuer)
                & pd.to_datetime(frame.date).le(prediction_date)
            ].sort_values("date").tail(length)
            arrays_by_issuer[issuer][rate] = torch.tensor(
                rows[value_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(),
                dtype=torch.float32,
            ).unsqueeze(0)

    config = MultiRateTransformerConfig(
        backbone="encoder_decoder", rates=("annual", "quarterly", "daily"),
        d_model=16, num_heads=4, layers=1, dropout=0.0,
    )
    model = MultiRateTransformer(
        {"annual": 8, "quarterly": 8, "daily": 14}, config=config, tasks=()
    ).eval()
    issuer_cache = IssuerContextCache(max_entries=4)
    outputs = []
    with torch.no_grad():
        for _, option in fixed.iterrows():
            spot = max(float(option.get("underlying_spot_entry", 1.0)), 1e-6)
            option_features = torch.tensor([
                float(option.get("strike", spot)) / spot,
                float(pd.to_numeric(option.get("volume"), errors="coerce") or 0.0),
                float(option.get("dte", 0.0)) / 100.0,
                float(option.get("moneyness", 0.0)),
                float(option.get("spread_pct", 0.0)),
                float(option.get("entry_mid", 0.0)) / spot,
            ], dtype=torch.float32)
            option_features[1] = torch.log1p(option_features[1].clamp_min(0.0))
            issuer_arrays = arrays_by_issuer[str(option.symbol).upper()]
            daily = torch.cat([
                issuer_arrays["daily"], option_features.view(1, 1, -1).expand(1, issuer_arrays["daily"].shape[1], -1)
            ], dim=-1)
            outputs.append(model(
                daily, arrays_by_issuer[str(option.symbol).upper()]["annual"], arrays_by_issuer[str(option.symbol).upper()]["quarterly"],
                issuer_context_cache=issuer_cache,
                issuer_context_key=(str(option.symbol).upper(), prediction_date.isoformat()),
                compute_document_outputs=False,
            ))
    assert len(issuer_cache) == fixed.symbol.nunique()
    for issuer, group in fixed.groupby(fixed.symbol.astype(str).str.upper()):
        indices = list(group.index)
        first = outputs[indices[0]]
        assert all(torch.allclose(first["rate_states"]["annual"], outputs[index]["rate_states"]["annual"]) for index in indices)
        assert all(torch.allclose(first["rate_states"]["quarterly"], outputs[index]["rate_states"]["quarterly"]) for index in indices)
    instrument_states = torch.cat([output["instrument_states"][:, -1] for output in outputs], dim=0)
    assert torch.unique(instrument_states, dim=0).shape[0] > 1


def test_auto_feature_engineer_has_temporal_and_cross_sectional_paths():
    torch.manual_seed(3)
    block = AutoFeatureEngineer(12)
    values = torch.randn(2, 4, 12)
    families = torch.randn(2, 4, 2, 12)
    presence = torch.ones(2, 4, 2)
    temporal = block(values, mode="temporal", family_states=families, family_presence=presence)
    cross = block(
        values,
        mode="cross_sectional",
        dates=torch.tensor([0, 0, 1, 1]),
        family_states=families,
        family_presence=presence,
    )
    assert temporal.shape == values.shape
    assert cross.shape == values.shape
    assert not torch.equal(temporal, cross)


def test_auto_feature_engineer_keeps_family_documents_isolated():
    torch.manual_seed(11)
    block = AutoFeatureEngineer(12)
    values = torch.randn(1, 4, 12)
    families = torch.randn(1, 4, 2, 12)
    changed = families.clone()
    changed[:, :, 1] += 100.0
    presence = torch.ones(1, 4, 2)
    _, first = block(
        values, mode="temporal", family_states=families,
        family_presence=presence, return_subtoken_states=True,
    )
    _, changed_first = block(
        values, mode="temporal", family_states=changed,
        family_presence=presence, return_subtoken_states=True,
    )
    assert torch.allclose(first[:, :, 0], changed_first[:, :, 0])


def test_auto_feature_engineer_has_cross_rate_path():
    block = AutoFeatureEngineer(12)
    rates = tuple(torch.randn(2, 12) for _ in range(3))
    output = block.cross_rate_features(rates)
    assert output.shape == (2, 12)


def test_prediction_heads_support_token_and_subtoken_objectives():
    model = MultiRateTransformer(
        {"annual": 3, "quarterly": 3, "daily": 3},
        config=MultiRateTransformerConfig(d_model=12, num_heads=3, layers=1),
        feature_families={
            "annual": {"fundamentals": 2, "prices": 1},
            "quarterly": {"fundamentals": 2, "prices": 1},
            "daily": {"fundamentals": 2, "prices": 1},
        },
        prediction_tasks=(
            MultiRatePredictionTaskSpec("next_daily_token", "next_token", "token", output_dim=3),
            MultiRatePredictionTaskSpec("masked_daily_family", "masked_token", "subtoken", output_dim=2),
        ),
    )
    output = model(
        torch.randn(2, 4, 3), torch.randn(2, 2, 3), torch.randn(2, 3, 3),
    )
    assert output["prediction_outputs"]["next_daily_token"].shape == (2, 4, 3)
    assert output["prediction_outputs"]["masked_daily_family"].shape == (2, 4, 2, 2)


def test_token_prediction_states_are_pooled_from_subtokens():
    config = MultiRateTransformerConfig(d_model=8, num_heads=2, layers=1, document_pool="mean")
    model = MultiRateTransformer(
        {"annual": 2, "quarterly": 2, "daily": 2},
        config=config,
        feature_families={
            "annual": {"a": 1, "b": 1},
            "quarterly": {"a": 1, "b": 1},
            "daily": {"a": 1, "b": 1},
        },
        prediction_tasks=(MultiRatePredictionTaskSpec("next_daily_token", "next_token", "token", output_dim=1, source="daily"),),
    )
    output = model(
        torch.randn(1, 3, 2), torch.randn(1, 2, 2), torch.randn(1, 2, 2),
        daily_family_presence=torch.tensor([[[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]]),
    )
    assert output["token_states"].shape == (1, 3, 8)
    assert output["prediction_outputs"]["next_daily_token"].shape == (1, 3, 1)


def test_document_pooling_means_valid_subtokens_and_excludes_missing_families():
    states = torch.tensor([[[[1.0, 3.0], [100.0, 100.0]], [[5.0, 7.0], [9.0, 11.0]]]])
    padding = torch.tensor([[False, False]])
    presence = torch.tensor([[[True, False], [True, True]]])
    pooled = MultiRateTransformer._pool_subtokens(states, padding, presence)
    assert torch.allclose(pooled, torch.tensor([[5.0, 7.0]]))


def test_family_classification_is_document_level():
    model = MultiRateTransformer(
        {"annual": 2, "quarterly": 2, "daily": 2},
        config=MultiRateTransformerConfig(d_model=12, num_heads=3, layers=1),
        feature_families={
            "annual": {"fundamentals": 1, "prices": 1},
            "quarterly": {"fundamentals": 1, "prices": 1},
            "daily": {"fundamentals": 1, "prices": 1},
        },
        tasks=(MultiRateTaskSpec("sector", "document", output_dim=3),),
        family_classification_dim=5,
    )
    output = model(
        torch.randn(2, 3, 2), torch.randn(2, 2, 2), torch.randn(2, 2, 2),
    )
    assert output["family_outputs"].shape == (2, 5)
    assert output["document_outputs"]["sector"].shape == (2, 3)


def test_family_documents_are_a_regular_document_task_source():
    model = MultiRateTransformer(
        {"annual": 2, "quarterly": 2, "daily": 2},
        config=MultiRateTransformerConfig(d_model=12, num_heads=3, layers=1),
        feature_families={
            "annual": {"balance_sheet": 1, "prices": 1},
            "quarterly": {"balance_sheet": 1, "prices": 1},
            "daily": {"balance_sheet": 1, "prices": 1},
        },
        tasks=(MultiRateTaskSpec("family", "document", output_dim=2, source="family"),),
    )
    output = model(
        torch.randn(2, 3, 2), torch.randn(2, 2, 2), torch.randn(2, 2, 2),
    )
    assert output["document_outputs"]["family"].shape == (2, 2, 2)
