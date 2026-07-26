# Multi-rate Transformer

The reusable PyTorch implementation lives at
`quant_orchestrator.platforms.ml_frameworks.torch.models.transformers.multirate`.

`MultiRateTransformer` supports three traditional layouts:

- `encoder_only`: independent annual, quarterly, and daily encoders followed
  by rate fusion;
- `decoder_only`: one self-attention stack over concatenated rate tokens;
- `encoder_decoder`: annual and quarterly encoders with a daily decoder.

All layouts use `build_attention_mask`:

- both modes use the same date-causal policy: same-date tokens attend
  bidirectionally, earlier dates are visible, and future dates are blocked;
- `attention_mode` remains a task label, while the date mask is the invariant
  that prevents temporal leakage.

Annual and quarterly inputs must already be restricted to observations that
were available at the prediction timestamp. The model does not infer filing or
release availability from fiscal dates.

Task heads are declared with `MultiRateTaskSpec` and may be `token` or
`document` level. Token heads consume rate token states; document heads consume
pooled daily, annual, quarterly, or fused states. Training loops, walk-forward
splits, target construction, and strategy/backtest selection remain caller-owned
orchestrator workflows.

## Required coverage path

Every rate is projected through `CoverageAwareInput`; this is not an optional
model feature. It provides:

- consumption of provider-produced outer-joined endpoint families, preserving
  the union of `(symbol, date)` rows;
- family presence masks, with finite-value inference when a caller has not
  supplied the pre-imputation mask;
- missingness embeddings, so an absent value is distinct from an observed zero;
- one learned `feature_family_adapter` and coverage gate per endpoint-level
  feature family;
- one learned adapter per asset/modality, selected with `*_modality_ids`;
- padding masks for attention and document pooling.

For multiple endpoint families, pass `feature_families` to the constructor, for
example `{"daily": {"equity_ohlcv": 8, "option_greeks": 12}, ...}`. The
dimensions must sum to the corresponding rate's `feature_dims`. Keep all
columns from one endpoint in the same family; do not split options data by
column. At inference, pass
`daily_family_presence` (and the annual/quarterly equivalents) captured before
imputation. A dense tensor without that argument is treated as one family and
uses its finite-value mask automatically.

## Learned auto-feature engineering

Every rate is passed through `AutoFeatureEngineer` before the backbone. It
shares one implementation across issuers, instruments, and asset classes:

- endpoint feature families become tokens, so the model can learn within-family,
  cross-family, cross-sectional, and temporal relationships in one attention
  space;
- same-date family/instrument tokens can interact in both directions, while
  different-date interactions remain past-only;
- learned attention can discover useful lookbacks, ratios, ranks, and other
  transformations without hardcoded SMA/RSI/rank operators or materializing
  every feature combination;
- pooled annual, quarterly, and daily states receive learned bidirectional
  cross-rate attention before fused document tasks.

These are learned latent features for the supervised tasks; they are not
persisted as an unbounded set of generated columns.
