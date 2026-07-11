from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quant_orchestrator.platforms.backtesting_frameworks.shared_book import (
    build_shared_book_weights,
    run_shared_book_backtest,
    shared_book_performance_metrics,
)
from quant_orchestrator.research_tools.ml_trading import (
    build_family_prediction_frame,
    build_strategy_score_frame,
)
from quant_orchestrator.research_tools.ml_trading_experiment import (
    _load_price_frames,
    _prepare_quant_warehouse_import,
    _warehouse_imports,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_dirs", nargs="+")
    parser.add_argument("--oos-start", default="2021-01-01")
    parser.add_argument("--provider", default="fmp")
    parser.add_argument("--price-start", default="1900-01-01")
    parser.add_argument("--cost-bps", type=float, default=5.5)
    parser.add_argument("--capital-base", type=float, default=1_000_000.0)
    parser.add_argument("--top-k", default="5,10,20,40")
    args = parser.parse_args()

    _prepare_quant_warehouse_import("/home/jlee153232/PycharmProjects/quant-warehouse")
    Warehouse, *_ = _warehouse_imports()
    warehouse = Warehouse()
    top_k_values = tuple(int(value) for value in str(args.top_k).split(",") if str(value).strip())

    for raw_artifact_dir in args.artifact_dirs:
        artifact_dir = Path(raw_artifact_dir)
        summary = run_returns(
            artifact_dir,
            warehouse=warehouse,
            provider=args.provider,
            price_start=args.price_start,
            oos_start=pd.Timestamp(args.oos_start),
            cost_bps=float(args.cost_bps),
            capital_base=float(args.capital_base),
            top_k_values=top_k_values,
        )
        print(f"\n## {artifact_dir}")
        print(summary.sort_values(["sharpe", "total_return"], ascending=False).head(12).to_string(index=False))


def run_returns(
    artifact_dir: Path,
    *,
    warehouse,
    provider: str,
    price_start: str,
    oos_start: pd.Timestamp,
    cost_bps: float,
    capital_base: float,
    top_k_values: tuple[int, ...],
) -> pd.DataFrame:
    metadata = json.loads((artifact_dir / "metadata.json").read_text())
    symbols = tuple(str(symbol).upper() for symbol in metadata["symbols"])
    source = str(metadata["source"])
    family = str(metadata["family"])

    feature_panel = pd.read_parquet(artifact_dir / "feature_panel.parquet")
    feature_panel["symbol"] = feature_panel["symbol"].astype(str).str.upper()
    feature_panel["date"] = pd.to_datetime(feature_panel["date"], errors="coerce").dt.normalize()
    feature_panel = feature_panel.loc[feature_panel["date"].ge(oos_start)].copy()
    features = [col for col in feature_panel.columns if col not in {"symbol", "date"}]

    with (artifact_dir / "classifier.pkl").open("rb") as handle:
        classifier = pickle.load(handle)

    prediction_frame = build_family_prediction_frame(
        feature_panel,
        features,
        min_feature_coverage=0.50,
    )
    proba = classifier.predict_proba_frame(prediction_frame, features)
    strategy_scores = build_strategy_score_frame(
        source=source,
        family=family,
        prediction_frame=prediction_frame,
        probability_frame=proba,
        ae_familiarity_frame=None,
        apply_ae_to_exits=False,
    )
    strategy_scores.to_parquet(artifact_dir / "strategy_scores.parquet", index=False)

    price_frames = _load_price_frames(
        warehouse,
        symbols,
        provider=provider,
        start=price_start,
        end=None,
    )
    wide_close = pd.DataFrame({symbol: frame["close"] for symbol, frame in price_frames.items()}).sort_index().ffill()
    next_returns = wide_close.pct_change().shift(-1)
    dates = pd.DatetimeIndex(sorted(set(strategy_scores["date"]).intersection(next_returns.index)))
    dates = dates[dates >= oos_start]
    trade_symbols = tuple(sorted(set(strategy_scores["symbol"]).intersection(next_returns.columns)))

    rows = []
    trade_logs = []
    for variant in ("long_only", "short_only", "long_short"):
        for top_k in top_k_values:
            weights, trades = build_shared_book_weights(
                strategy_scores,
                trade_symbols,
                dates,
                top_k=top_k,
                variant=variant,
                entry_threshold=0.50,
                exit_threshold=0.50,
                long_exit_score_col="long_exit_score",
                short_exit_score_col="short_exit_score",
            )
            returns, equity, _turnover = run_shared_book_backtest(
                weights,
                next_returns,
                cost_bps=cost_bps,
                capital_base=capital_base,
            )
            row = shared_book_performance_metrics(
                returns,
                equity,
                weights,
                trades,
                framework="vectorized_shared_book",
                variant=variant,
                top_k=top_k,
                cost_bps=cost_bps,
            )
            row.update(
                {
                    "strategy_source": f"{source}.{family}",
                    "symbols": len(trade_symbols),
                    "score_rows": len(strategy_scores),
                    "score_dates": strategy_scores["date"].nunique(),
                }
            )
            rows.append(row)
            if not trades.empty:
                trade_logs.append(trades.assign(variant=variant, top_k=top_k))

    summary = pd.DataFrame(rows).sort_values(["variant", "top_k"]).reset_index(drop=True)
    summary.to_csv(artifact_dir / "backtest_summary_vectorized.csv", index=False)
    if trade_logs:
        pd.concat(trade_logs, ignore_index=True).to_csv(artifact_dir / "trade_log_vectorized.csv", index=False)
    return summary


if __name__ == "__main__":
    main()
