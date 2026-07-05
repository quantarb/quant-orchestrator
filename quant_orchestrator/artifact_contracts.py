from __future__ import annotations

"""Produced-artifact contracts shared across research and framework adapters."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


ARTIFACT_SCHEMA_VERSION = 1

FEATURE_PANEL_REQUIRED_COLUMNS = ("date", "symbol")
SCORED_PANEL_REQUIRED_COLUMNS = ("date", "symbol", "close")
ACTION_TAPE_REQUIRED_COLUMNS = ("date", "symbol", "action", "price")
TRADE_LIST_REQUIRED_COLUMNS = ("trade_id", "symbol", "side", "entry_date", "exit_date")
TRADE_LIST_SIDES = frozenset({"long", "short"})
TRADE_LIST_NON_NEGATIVE_COLUMNS = ("equity_entry_notional",)
TRADE_LIST_ARTIFACT_NAME = "trade_list"


@dataclass(frozen=True)
class StrategyArtifactBundle:
    feature_panel: pd.DataFrame | None = None
    scored_panel: pd.DataFrame | None = None
    action_tape: pd.DataFrame | None = None
    trade_list: pd.DataFrame | None = None
    summary: Mapping[str, Any] = field(default_factory=dict)
    strategy_name: str = ""
    base_path: Path | None = None
    feature_panel_path: Path | None = None
    scored_panel_path: Path | None = None
    action_tape_path: Path | None = None
    trade_list_path: Path | None = None
    summary_path: Path | None = None
    manifest_path: Path | None = None


def write_strategy_artifacts(
    bundle: StrategyArtifactBundle,
    output_dir: str | Path,
    *,
    extra_paths: Mapping[str, str | Path] | None = None,
) -> dict[str, Path]:
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    manifest_artifacts: dict[str, dict[str, Any]] = {}

    for name, frame, required in (
        ("feature_panel", bundle.feature_panel, FEATURE_PANEL_REQUIRED_COLUMNS),
        ("scored_panel", bundle.scored_panel, SCORED_PANEL_REQUIRED_COLUMNS),
        ("action_tape", bundle.action_tape, ACTION_TAPE_REQUIRED_COLUMNS),
        (TRADE_LIST_ARTIFACT_NAME, bundle.trade_list, TRADE_LIST_REQUIRED_COLUMNS),
    ):
        if frame is None:
            continue
        validated = validate_strategy_artifact_frame(name, frame, required_columns=required)
        path = out_dir / f"{name}.parquet"
        validated.to_parquet(path, index=False)
        paths[name] = path
        manifest_artifacts[name] = {
            "path": path.name,
            "rows": int(len(validated)),
            "columns": list(map(str, validated.columns)),
        }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(dict(bundle.summary), indent=2, default=str), encoding="utf-8")
    paths["summary"] = summary_path
    manifest_artifacts["summary"] = {"path": summary_path.name}

    if extra_paths:
        for name, path_value in extra_paths.items():
            manifest_name = str(name)
            if manifest_name in manifest_artifacts:
                manifest_name = f"extra_{manifest_name}"
            path = Path(path_value)
            paths[str(name)] = path
            manifest_artifacts[manifest_name] = {"path": str(path)}

    manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "strategy_name": str(bundle.strategy_name or ""),
        "artifacts": manifest_artifacts,
    }
    manifest_path = out_dir / "strategy_artifacts_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    paths["manifest"] = manifest_path
    return paths


def read_strategy_artifacts(path: str | Path) -> StrategyArtifactBundle:
    base = Path(path).expanduser().resolve()
    manifest_path = base / "strategy_artifacts_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing strategy artifact manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = dict(manifest.get("artifacts") or {})

    resolved_paths: dict[str, Path] = {}

    def artifact_path(name: str) -> Path | None:
        entry = artifacts.get(name)
        if not entry:
            return None
        frame_path = base / str(entry.get("path"))
        resolved_paths[name] = frame_path
        return frame_path

    def read_frame(name: str, required: tuple[str, ...]) -> pd.DataFrame | None:
        frame_path = artifact_path(name)
        if frame_path is None:
            return None
        if not frame_path.exists():
            raise FileNotFoundError(f"Missing {name} artifact: {frame_path}")
        return validate_strategy_artifact_frame(name, pd.read_parquet(frame_path), required_columns=required)

    summary_path = base / str((artifacts.get("summary") or {}).get("path", "summary.json"))
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    return StrategyArtifactBundle(
        feature_panel=read_frame("feature_panel", FEATURE_PANEL_REQUIRED_COLUMNS),
        scored_panel=read_frame("scored_panel", SCORED_PANEL_REQUIRED_COLUMNS),
        action_tape=read_frame("action_tape", ACTION_TAPE_REQUIRED_COLUMNS),
        trade_list=read_frame(TRADE_LIST_ARTIFACT_NAME, TRADE_LIST_REQUIRED_COLUMNS),
        summary=summary,
        strategy_name=str(manifest.get("strategy_name") or ""),
        base_path=base,
        feature_panel_path=resolved_paths.get("feature_panel"),
        scored_panel_path=resolved_paths.get("scored_panel"),
        action_tape_path=resolved_paths.get("action_tape"),
        trade_list_path=resolved_paths.get("trade_list"),
        summary_path=summary_path,
        manifest_path=manifest_path,
    )


def validate_strategy_artifact_frame(
    artifact_name: str,
    frame: pd.DataFrame,
    *,
    required_columns: tuple[str, ...],
) -> pd.DataFrame:
    if frame is None:
        raise TypeError(f"{artifact_name} artifact frame is None")
    out = _frame_with_date_symbol_columns(frame)
    missing = [column for column in required_columns if column not in out.columns]
    if missing and out.empty:
        for column in missing:
            out[column] = pd.Series(dtype="object")
        missing = []
    if missing:
        raise ValueError(f"{artifact_name} artifact missing required columns: {missing}")
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
        out = out.dropna(subset=["date"])
    if "entry_date" in out.columns:
        out["entry_date"] = pd.to_datetime(out["entry_date"], errors="coerce").dt.normalize()
    if "exit_date" in out.columns:
        out["exit_date"] = pd.to_datetime(out["exit_date"], errors="coerce").dt.normalize()
    if "symbol" in out.columns:
        out["symbol"] = out["symbol"].astype(str).str.strip().str.upper()
        out = out.loc[out["symbol"].ne("")]
    if "action" in out.columns:
        out["action"] = out["action"].astype(str).str.strip().str.lower()
    if artifact_name == TRADE_LIST_ARTIFACT_NAME:
        out = _validate_trade_list(out)
    sort_cols = [column for column in ("date", "entry_date", "symbol", "action") if column in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols, kind="stable")
    return out.reset_index(drop=True)


def normalize_trade_list(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize a reusable list of closed equity trades.

    This is the standard downstream handoff for Monte Carlo, option-equivalent
    replay, equity-curve analysis, and ensemble/mixing experiments. Producers
    are intentionally unconstrained: a notebook, native backtesting framework,
    saved optimal_trader artifact, or external system may create this table as
    long as the emitted artifact satisfies this schema.
    """

    return validate_strategy_artifact_frame(
        TRADE_LIST_ARTIFACT_NAME,
        frame,
        required_columns=TRADE_LIST_REQUIRED_COLUMNS,
    )


def write_trade_list_artifact(
    trades: pd.DataFrame,
    output_dir: str | Path,
    *,
    strategy_name: str = "",
    summary: Mapping[str, Any] | None = None,
    extra_paths: Mapping[str, str | Path] | None = None,
) -> dict[str, Path]:
    """Write only the canonical trade-list contract.

    The physical parquet is ``trade_list.parquet`` and the manifest exposes the
    same canonical ``trade_list`` artifact name.
    """

    return write_strategy_artifacts(
        StrategyArtifactBundle(
            trade_list=trades,
            summary=dict(summary or {}),
            strategy_name=str(strategy_name or ""),
        ),
        output_dir,
        extra_paths=extra_paths,
    )


def read_trade_list_artifact(path: str | Path) -> pd.DataFrame:
    """Load a trade list from a manifest directory or a direct dataframe file."""

    source = Path(path).expanduser().resolve()
    if source.is_dir():
        bundle = read_strategy_artifacts(source)
        if bundle.trade_list is None:
            raise FileNotFoundError(f"Strategy artifact manifest has no trade_list artifact: {source}")
        return bundle.trade_list
    if not source.exists():
        raise FileNotFoundError(f"Missing trade-list artifact: {source}")
    if source.suffix.lower() in {".parquet", ".pq"}:
        frame = pd.read_parquet(source)
    elif source.suffix.lower() == ".csv":
        frame = pd.read_csv(source)
    elif source.suffix.lower() in {".pkl", ".pickle"}:
        frame = pd.read_pickle(source)
    else:
        raise ValueError(f"Unsupported trade-list artifact format: {source.suffix}")
    return normalize_trade_list(frame)


def combine_trade_lists(
    sources: Mapping[str, pd.DataFrame | str | Path],
    *,
    source_column: str = "artifact_source",
) -> pd.DataFrame:
    """Load and stack multiple trade-list artifacts with source attribution."""

    frames: list[pd.DataFrame] = []
    for source_name, source in sources.items():
        if isinstance(source, pd.DataFrame):
            frame = normalize_trade_list(source)
        else:
            frame = read_trade_list_artifact(source)
        frame = frame.copy()
        if source_column not in frame.columns:
            frame[source_column] = str(source_name)
        frames.append(frame)
    if not frames:
        return normalize_trade_list(pd.DataFrame(columns=TRADE_LIST_REQUIRED_COLUMNS))
    return normalize_trade_list(pd.concat(frames, ignore_index=True, sort=False))


def _validate_trade_list(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if out.empty:
        return out
    null_date = out["entry_date"].isna() | out["exit_date"].isna()
    if bool(null_date.any()):
        raise ValueError("trade_list contains null or invalid entry_date/exit_date")
    reversed_dates = out["exit_date"].lt(out["entry_date"])
    if bool(reversed_dates.any()):
        raise ValueError("trade_list contains exit_date before entry_date")
    out["side"] = out["side"].astype(str).str.strip().str.lower()
    bad_side = out["side"].notna() & ~out["side"].isin(TRADE_LIST_SIDES)
    if bool(bad_side.any()):
        invalid = sorted(out.loc[bad_side, "side"].dropna().unique().tolist())
        raise ValueError(f"trade_list contains invalid side values: {invalid}")
    if "trade_id" in out.columns:
        out["trade_id"] = out["trade_id"].astype(str).str.strip()
        if bool(out["trade_id"].eq("").any()):
            raise ValueError("trade_list contains blank trade_id values")
    for column in TRADE_LIST_NON_NEGATIVE_COLUMNS:
        if column not in out.columns:
            continue
        numeric = pd.to_numeric(out[column], errors="coerce")
        invalid = numeric.notna() & numeric.lt(0.0)
        if bool(invalid.any()):
            raise ValueError(f"trade_list contains negative {column} values")
        out[column] = numeric
    return out


def _frame_with_date_symbol_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if isinstance(out.index, pd.MultiIndex) and {"date", "symbol"}.issubset(set(out.index.names)):
        out = out.reset_index()
    elif isinstance(out.index, pd.DatetimeIndex) and "date" not in out.columns:
        out = out.reset_index(names="date")
    return out.loc[:, ~out.columns.duplicated()].copy()
