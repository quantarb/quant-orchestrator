from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import pandas as pd

from quant_orchestrator.research_tools.ml_trading import write_ml_trading_artifact_files
from quant_orchestrator.research_tools.ml_trading_experiment import (
    MLTradingExperimentConfig,
    run_ml_trading_experiment,
)
from quant_orchestrator.research_tools.options_experiment import (
    OptionMvBasketConfig,
    OptionRetrievalConfig,
    OptionWindowBuildConfig,
    OptopsyExecutionConfig,
    OracleOptionExperimentConfig,
    SharedSplitConfig,
    build_option_window_dataset,
    run_trade_window_option_experiment,
)


CLEAN_FEATURE_SOURCES = (
    "fmp.fmp_income_mcap",
    "fmp.fmp_balance_mcap",
    "fmp.fmp_cash_mcap",
    "fmp.fmp_daily_mcap_multiple",
    "fmp.fmp_daily_mcap_yield",
    "fmp.fmp_daily_ev_multiple",
    "fmp.fmp_daily_ev_yield",
    "financetoolkit.ft_growth_income",
    "financetoolkit.ft_growth_balance",
    "financetoolkit.ft_growth_cash",
    "financetoolkit.ft_ratios_profitability",
    "financetoolkit.ft_ratios_efficiency",
    "financetoolkit.ft_ratios_valuation",
    "financetoolkit.ft_ratios_solvency",
    "financetoolkit.ft_ratios_liquidity",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", default="artifacts/clean_oracle_1t_options")
    parser.add_argument("--log-mlflow", action="store_true")
    parser.add_argument("--skip-equity", action="store_true")
    parser.add_argument("--equity-artifact-dir", default="")
    parser.add_argument("--option-groups", default="")
    parser.add_argument("--max-option-groups", type=int, default=0)
    parser.add_argument("--run-rule", action="store_true", default=True)
    parser.add_argument("--run-full-chain", action="store_true", default=True)
    args = parser.parse_args()

    started = perf_counter()
    root = Path(args.artifact_root)
    root.mkdir(parents=True, exist_ok=True)

    if args.skip_equity:
        equity_dir = Path(args.equity_artifact_dir)
        if not equity_dir.exists():
            raise FileNotFoundError("--equity-artifact-dir is required with --skip-equity")
        strategy_scores = pd.read_csv(equity_dir / "strategy_scores.csv")
        backtest_summary = pd.read_csv(equity_dir / "backtest_summary.csv")
        equity_metrics = {}
    else:
        equity_config = MLTradingExperimentConfig(
            experiment_name="clean_oracle_1t_feature_families",
            min_market_cap=1_000_000_000_000,
            start_date="1900-01-01",
            train_end="2020-12-31",
            oos_start="2021-01-01",
            score_start="1900-01-01",
            top_k_values=(5, 10, 20, 40),
            strategy_sources=CLEAN_FEATURE_SOURCES,
            target_label_mode="oracle_only",
            oracle_frequencies=("YE",),
            oracle_k_min=1,
            oracle_k_max=12,
            log_mlflow=bool(args.log_mlflow),
        )
        equity_result = run_ml_trading_experiment(equity_config)
        equity_dir = root / "equity"
        equity_dir.mkdir(parents=True, exist_ok=True)
        write_ml_trading_artifact_files(
            model_results=equity_result.model_results,
            strategy_scores=equity_result.strategy_scores,
            backtest_summary=equity_result.backtest_summary,
            trade_log=equity_result.trade_log,
            model_vs_trading=equity_result.model_vs_trading,
            metric_correlations=equity_result.metric_correlations,
            yearly_backtest_summary=equity_result.yearly_backtest_summary,
            symbol_strategy_summary=equity_result.symbol_strategy_summary,
            symbol_robustness_summary=equity_result.symbol_robustness_summary,
            backtesting_py_symbol_validation=equity_result.backtesting_py_symbol_validation,
            phase_timings=equity_result.phase_timings,
            analysis_markdown=equity_result.analysis_markdown,
            directory=equity_dir,
        )
        strategy_scores = equity_result.strategy_scores
        backtest_summary = equity_result.backtest_summary
        equity_metrics = equity_result.metrics

    window_config = OptionWindowBuildConfig(
        variant="long_short",
        top_k=5,
        entry_threshold=0.50,
        exit_threshold=0.50,
    )
    window_dataset = build_option_window_dataset(
        strategy_scores,
        backtest_summary=backtest_summary,
        config=window_config,
        include_groups=("individual_feature_families", "all_feature_families"),
    )
    window_dataset.window_summary.to_csv(root / "option_window_summary.csv", index=False)
    window_dataset.source_ranking.to_csv(root / "option_source_ranking.csv", index=False)

    requested_groups = tuple(group.strip() for group in args.option_groups.split(",") if group.strip())
    group_names = [group for group in window_dataset.windows_by_group if not requested_groups or group in requested_groups]
    if args.max_option_groups > 0:
        group_names = group_names[: args.max_option_groups]

    option_runs = []
    symbols = tuple(sorted(strategy_scores["symbol"].dropna().astype(str).str.upper().unique().tolist()))
    for group_name in group_names:
        windows = window_dataset.windows_by_group[group_name]
        if windows.empty:
            continue
        if args.run_rule:
            option_runs.append(
                _run_option_group(
                    root=root,
                    group_name=group_name,
                    variant_name="option_rule_atm_90d",
                    symbols=symbols,
                    windows=windows,
                    retrieval=OptionRetrievalConfig(
                        option_universe="filtered",
                        min_dte=60,
                        max_dte=120,
                        target_dte=90,
                        max_candidates_per_trade=40,
                    ),
                    log_mlflow=bool(args.log_mlflow),
                )
            )
        if args.run_full_chain:
            option_runs.append(
                _run_option_group(
                    root=root,
                    group_name=group_name,
                    variant_name="option_full_chain_actions",
                    symbols=symbols,
                    windows=windows,
                    retrieval=OptionRetrievalConfig(
                        option_universe="full_chain_actions",
                        target_dte=90,
                        require_affordable=False,
                        min_entry_mid=0.0,
                        max_entry_spread_pct=10_000.0,
                        max_abs_moneyness=10_000.0,
                        max_candidates_per_trade=1_000_000,
                    ),
                    log_mlflow=bool(args.log_mlflow),
                )
            )

    summary = {
        "elapsed_seconds": perf_counter() - started,
        "equity_artifact_dir": str(equity_dir),
        "equity_metrics": equity_metrics,
        "option_runs": option_runs,
    }
    (root / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")


def _run_option_group(
    *,
    root: Path,
    group_name: str,
    variant_name: str,
    symbols: tuple[str, ...],
    windows: pd.DataFrame,
    retrieval: OptionRetrievalConfig,
    log_mlflow: bool,
) -> dict[str, object]:
    artifact_dir = root / "options" / variant_name / group_name.replace("/", "_")
    config = OracleOptionExperimentConfig(
        experiment_name=f"clean_oracle_1t_{variant_name}_{group_name}",
        symbols=symbols,
        price_start="1900-01-01",
        price_end=None,
        split=SharedSplitConfig(insample_end="2020-12-31"),
        retrieval=retrieval,
        execution=OptopsyExecutionConfig(
            capital=100_000.0,
            max_positions=5,
            sizing_mode="portfolio_fraction",
            top_k_column="top_k",
        ),
        mv_basket=OptionMvBasketConfig(enabled=True, max_legs=4),
        artifact_dir=str(artifact_dir),
        log_mlflow=log_mlflow,
    )
    result = run_trade_window_option_experiment(config, windows)
    return {
        "group": group_name,
        "variant": variant_name,
        "artifact_dir": str(artifact_dir),
        "trade_windows": len(result.oracle_trades),
        "option_rows": len(result.option_panel),
        "train_rows": len(result.train_panel),
        "eval_rows": len(result.eval_panel),
        "elapsed_seconds": result.elapsed_seconds,
        "metrics": result.metrics,
    }


if __name__ == "__main__":
    main()
