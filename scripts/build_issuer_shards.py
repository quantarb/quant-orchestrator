"""Build issuer-partitioned multi-rate tables for sequential training."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", str(value))[:160]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.corpus
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    taxonomy = pd.read_csv(root / "taxonomy.csv", usecols=["symbol", "issuer"])
    taxonomy["symbol"] = taxonomy["symbol"].astype(str).str.upper()
    taxonomy["issuer"] = taxonomy["issuer"].astype(str)
    symbol_to_issuer = taxonomy.set_index("symbol")["issuer"].to_dict()
    issuer_names = sorted(taxonomy["issuer"].unique())
    issuer_dirs = {issuer: output / _safe_name(issuer) for issuer in issuer_names}
    counts: dict[str, dict[str, int]] = {issuer: {} for issuer in issuer_names}
    for rate in ("annual", "quarterly", "daily", "sparse_events"):
        source = root / f"{rate}.parquet"
        table = pd.read_parquet(source)
        table["symbol"] = table["symbol"].astype(str).str.upper()
        table["issuer"] = table["symbol"].map(symbol_to_issuer)
        table = table.dropna(subset=["issuer"])
        for issuer, group in table.groupby("issuer", sort=False):
            directory = issuer_dirs[str(issuer)]
            directory.mkdir(parents=True, exist_ok=True)
            group.drop(columns=["issuer"]).to_parquet(directory / f"{rate}.parquet", index=False)
            counts[str(issuer)][rate] = int(len(group))
    (output / "manifest.json").write_text(json.dumps({
        "source_corpus": str(root),
        "issuers": [{"issuer": issuer, "directory": issuer_dirs[issuer].name, **counts[issuer]} for issuer in issuer_names],
    }, indent=2))
    print(json.dumps({"issuers": len(issuer_names), "output": str(output), "rates": list(counts.values())[0].keys() if counts else []}, indent=2, default=list))


if __name__ == "__main__":
    main()
