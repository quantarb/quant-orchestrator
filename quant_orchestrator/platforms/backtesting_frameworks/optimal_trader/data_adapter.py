from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_strategy_dataset_artifact(path: str | Path) -> pd.DataFrame:
    """Load a saved optimal_trader strategy dataset for backtest replay."""

    source = Path(path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Missing strategy dataset artifact: {source}")
    suffix = source.suffix.lower()
    if suffix in {".pkl", ".pickle"}:
        frame = pd.read_pickle(source)
    elif suffix in {".parquet", ".pq"}:
        frame = pd.read_parquet(source)
    else:
        frame = pd.read_csv(source)
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"Strategy dataset artifact did not load as a DataFrame: {source}")
    return normalize_strategy_dataset_frame(frame)


def normalize_strategy_dataset_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    out = frame.copy()
    out = out.rename(columns={column: _canonical_column_name(column) for column in out.columns})
    required = {"date", "symbol", "target_weight"}
    missing = sorted(required.difference(out.columns))
    if missing:
        raise ValueError(f"Strategy dataset artifact missing required columns: {missing}")
    if "ret_1" not in out.columns and "asset_return" not in out.columns:
        raise ValueError("Strategy dataset artifact requires 'ret_1' or 'asset_return'")
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["symbol"] = out["symbol"].astype(str).str.strip().str.upper()
    out = out.loc[out["date"].notna() & out["symbol"].ne("")].copy()
    return out.reset_index(drop=True)


def _canonical_column_name(column: object) -> str:
    text = str(column).strip()
    mapping = {
        "Date": "date",
        "Symbol": "symbol",
        "Target Weight": "target_weight",
        "TargetWeight": "target_weight",
        "Asset Return": "asset_return",
        "AssetReturn": "asset_return",
        "Return": "ret_1",
    }
    return mapping.get(text, text)
