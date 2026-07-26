"""Build sparse 1T symbol/family documents from Quant Warehouse features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from quant_warehouse.research_tools.feature_family_eval import (
    FamilyEvaluationConfig,
    build_fundamental_feature_panel,
)
from quant_orchestrator.platforms.ml_frameworks.torch.models.transformers.multirate.documents import (
    build_1t_subtoken_documents,
)


TA_PREFIXES = ("technical_", "ta_")
TA_FAMILIES = {"technical_candles", "technical_cycles", "technical_math", "technical_momentum", "technical_overlap", "technical_performance"}


def _is_ta(source_family: str) -> bool:
    family = source_family.split(".", 1)[-1].lower()
    return family in TA_FAMILIES or family.startswith(TA_PREFIXES)


def build(index_path: Path, output_dir: Path) -> dict[str, object]:
    index = pd.read_csv(index_path)
    requested = [str(value) for value in index.strategy_source if not _is_ta(str(value))]
    first_panel = pd.read_parquet(index.iloc[0].panel_path, columns=["symbol"])
    symbols = tuple(sorted(first_panel["symbol"].astype(str).str.upper().unique()))
    config = FamilyEvaluationConfig(
        market_cap_min=1_000_000_000_000,
        start_date="2018-01-01",
        end_date=None,
        screen_limit=max(5_000, len(symbols)),
    )
    panel, metadata, diagnostics, timings = build_fundamental_feature_panel(
        symbols,
        config,
        strategy_sources=requested,
        broadcast_to_target=False,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for family, family_meta in metadata.groupby("family", sort=True):
        columns = [str(value) for value in family_meta.feature if str(value) in panel.columns]
        if not columns:
            continue
        source = str(family_meta.source.iloc[0])
        source_family = f"{source}.{family}"
        if _is_ta(source_family):
            continue
        family_frame = panel[["symbol", "date", *columns]].dropna(subset=columns, how="all")
        if family_frame.empty:
            continue
        corpus = build_1t_subtoken_documents(
            family_frame,
            {source_family: tuple(columns)},
        )
        family_dir = output_dir / source_family.replace("/", "_")
        family_dir.mkdir(parents=True, exist_ok=True)
        corpus.documents.to_parquet(family_dir / "documents.parquet", index=False)
        corpus.subtokens.to_parquet(family_dir / "subtokens.parquet", index=False)
        corpus.document_subtokens.to_parquet(family_dir / "document_subtokens.parquet", index=False)
        corpus.prototype_targets.to_parquet(family_dir / "prototype_targets.parquet", index=False)
        rows.append({
            "feature_family": source_family,
            "features": len(columns),
            "subtokens": len(corpus.subtokens),
            "documents": len(corpus.documents),
            "symbols": int(corpus.documents.symbol.nunique()),
            "first_date": str(corpus.subtokens.timestamp.min()),
            "last_date": str(corpus.subtokens.timestamp.max()),
        })
    summary = pd.DataFrame(rows).sort_values("feature_family").reset_index(drop=True)
    summary.to_csv(output_dir / "family_summary.csv", index=False)
    diagnostics.to_csv(output_dir / "diagnostics.csv", index=False)
    manifest = {
        "source_index": str(index_path),
        "universe": "1T",
        "sampling": "source observation dates; no daily broadcast",
        "document_key": ["symbol", "feature_family"],
        "prototype": "mean_subtoken_embedding",
        "requested_non_ta_families": len(requested),
        "built_families": int(len(summary)),
        "excluded_ta_families": int(len(index) - len(requested)),
        "total_subtokens": int(summary.subtokens.sum()) if not summary.empty else 0,
        "total_documents": int(summary.documents.sum()) if not summary.empty else 0,
        "timings": timings,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.index, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
