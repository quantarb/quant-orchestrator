from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from quant_orchestrator.platforms.backtesting_frameworks.panel_weight import (
    SyntheticOptionsBacktestConfig,
    SyntheticOptionsBacktestResult,
    run_synthetic_options_backtest,
)


@dataclass(frozen=True)
class SyntheticOptionsBacktestRunConfig:
    experiment_name: str = "traditional_ml_synthetic_options_backtest"
    panel_path: str | None = None
    artifact_dir: str | None = None
    backtest: SyntheticOptionsBacktestConfig = SyntheticOptionsBacktestConfig()


@dataclass(frozen=True)
class SyntheticOptionsBacktestRunResult:
    config: SyntheticOptionsBacktestRunConfig
    result: SyntheticOptionsBacktestResult
    artifact_paths: dict[str, Path]


def run_synthetic_options_backtest_experiment(
    config: SyntheticOptionsBacktestRunConfig,
    *,
    bt_panel: pd.DataFrame | None = None,
) -> SyntheticOptionsBacktestRunResult:
    if bt_panel is None:
        if not config.panel_path:
            raise ValueError("Either bt_panel or config.panel_path is required")
        bt_panel = read_backtest_panel(config.panel_path)
    result = run_synthetic_options_backtest(bt_panel, config=config.backtest)
    artifact_paths = write_synthetic_options_backtest_artifacts(
        result,
        artifact_dir=_artifact_dir(config),
    )
    return SyntheticOptionsBacktestRunResult(config=config, result=result, artifact_paths=artifact_paths)


def read_backtest_panel(path: str | Path) -> pd.DataFrame:
    source = Path(path).expanduser()
    if source.suffix.lower() in {".parquet", ".pq"}:
        frame = pd.read_parquet(source)
    else:
        frame = pd.read_csv(source)
    if isinstance(frame.index, pd.MultiIndex) and {"date", "symbol"}.issubset(frame.index.names):
        return frame.sort_index()
    missing = [column for column in ("date", "symbol") if column not in frame.columns]
    if missing:
        raise ValueError(f"Backtest panel file is missing required columns: {missing}")
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out["symbol"] = out["symbol"].astype(str).str.upper()
    return out.dropna(subset=["date", "symbol"]).set_index(["date", "symbol"]).sort_index()


def write_synthetic_options_backtest_artifacts(
    result: SyntheticOptionsBacktestResult,
    *,
    artifact_dir: str | Path,
) -> dict[str, Path]:
    root = Path(artifact_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": root / "summary.csv",
        "yearly_summary": root / "yearly_summary.csv",
        "close_panel": root / "close_panel.parquet",
        "realized_vol_panel": root / "realized_vol_panel.parquet",
        "synthetic_price_metadata": root / "synthetic_price_metadata.csv",
    }
    result.summary.to_csv(paths["summary"], index=False)
    result.yearly_summary.to_csv(paths["yearly_summary"], index=False)
    result.close_panel.to_parquet(paths["close_panel"])
    result.realized_vol_panel.to_parquet(paths["realized_vol_panel"])
    _synthetic_price_metadata(result).to_csv(paths["synthetic_price_metadata"], index=False)
    if not result.real_quote_coverage.empty:
        paths["real_quote_coverage"] = root / "real_quote_coverage.csv"
        result.real_quote_coverage.to_csv(paths["real_quote_coverage"], index=False)

    equity_dir = root / "equity_curves"
    positions_dir = root / "positions"
    equity_dir.mkdir(exist_ok=True)
    positions_dir.mkdir(exist_ok=True)
    for (strategy, instrument, top_k), payload in result.variant_runs.items():
        stem = _artifact_stem(strategy, instrument, top_k)
        summary = payload["summary"]
        backtest = payload["backtest"]
        positions = payload["positions"]
        if isinstance(summary, dict) and "equity_curve" in summary:
            pd.Series(summary["equity_curve"]).rename("equity").to_frame().to_csv(
                equity_dir / f"{stem}.csv",
            )
        if isinstance(backtest, dict) and "returns" in backtest:
            pd.Series(backtest["returns"]).rename("returns").to_frame().to_csv(
                equity_dir / f"{stem}_returns.csv",
            )
        if isinstance(positions, pd.DataFrame):
            positions.to_parquet(positions_dir / f"{stem}.parquet")
    paths["equity_curves"] = equity_dir
    paths["positions"] = positions_dir
    return paths


def _artifact_dir(config: SyntheticOptionsBacktestRunConfig) -> Path:
    if config.artifact_dir:
        return Path(config.artifact_dir).expanduser()
    return Path("artifacts") / "synthetic_options" / config.experiment_name


def _artifact_stem(strategy: str, instrument: str, top_k: int) -> str:
    raw = f"{strategy}_{instrument}_top_{int(top_k)}"
    return "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in raw)


def _synthetic_price_metadata(result: SyntheticOptionsBacktestResult) -> pd.DataFrame:
    rows = []
    for instrument, payload in result.synthetic_price_panels.items():
        call = payload.get("call")
        put = payload.get("put")
        rows.append(
            {
                "instrument": instrument,
                "long_strike_multiplier": payload.get("long_strike_multiplier"),
                "short_strike_multiplier": payload.get("short_strike_multiplier"),
                "call_rows": len(call) if isinstance(call, pd.DataFrame) else 0,
                "put_rows": len(put) if isinstance(put, pd.DataFrame) else 0,
                "call_symbols": len(call.columns) if isinstance(call, pd.DataFrame) else 0,
                "put_symbols": len(put.columns) if isinstance(put, pd.DataFrame) else 0,
            },
        )
    return pd.DataFrame(rows)
