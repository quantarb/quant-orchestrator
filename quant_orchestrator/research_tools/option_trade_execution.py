from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OptionTradeExecutionBatch:
    selected_option_trades: pd.DataFrame
    selected_option_paths: pd.DataFrame
    trade_status: pd.DataFrame
    metrics: dict[str, float]


class OptionSelectionRetriever(Protocol):
    def select_rule_atm_entry(self, trade: pd.Series) -> pd.DataFrame: ...

    def price_selected_options_with_paths(
        self,
        selected: pd.DataFrame,
        *,
        selector_name: str | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]: ...

    def metrics(self) -> dict[str, float]: ...


class OptionTradeExecutor:
    def __init__(
        self,
        retriever_factory,
        *,
        selector_name: str = "rule_atm_90d",
        workers: int = 1,
    ):
        if selector_name != "rule_atm_90d":
            raise ValueError(
                "fast option execution currently supports selector_name='rule_atm_90d', "
                f"got {selector_name!r}"
            )
        self.retriever_factory = retriever_factory
        self.selector_name = selector_name
        self.workers = max(1, int(workers))

    def execute(self, trade_list: pd.DataFrame) -> OptionTradeExecutionBatch:
        if trade_list.empty:
            return OptionTradeExecutionBatch(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {})
        if self.workers <= 1 or len(trade_list) <= 1:
            return self._execute_chunk(trade_list)
        chunks = _split_frame_for_workers(trade_list, self.workers)
        batches: list[OptionTradeExecutionBatch] = []
        with ThreadPoolExecutor(max_workers=min(self.workers, len(chunks))) as pool:
            futures = [pool.submit(self._execute_chunk, chunk) for chunk in chunks if not chunk.empty]
            for future in as_completed(futures):
                batches.append(future.result())
        return _merge_option_execution_batches(batches)

    def _execute_chunk(self, trade_list: pd.DataFrame) -> OptionTradeExecutionBatch:
        retriever: OptionSelectionRetriever = self.retriever_factory()
        selected_frames = []
        path_frames = []
        status_rows: list[dict[str, Any]] = []
        for trade in trade_list.itertuples(index=False):
            selected, paths, status = self._execute_trade(pd.Series(trade._asdict()), retriever)
            if not selected.empty:
                selected_frames.append(selected)
            if not paths.empty:
                path_frames.append(paths)
            status_rows.append(status)
        selected_frame = pd.concat(selected_frames, ignore_index=True) if selected_frames else pd.DataFrame()
        path_frame = pd.concat(path_frames, ignore_index=True) if path_frames else pd.DataFrame()
        status_frame = pd.DataFrame(status_rows)
        return OptionTradeExecutionBatch(selected_frame, path_frame, status_frame, retriever.metrics())

    def _execute_trade(
        self,
        payload: pd.Series,
        retriever: OptionSelectionRetriever,
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        trade_id = payload.get("trade_id")
        symbol = str(payload.get("symbol", "")).upper()
        entry_date = pd.Timestamp(payload.get("entry_date")).normalize()
        exit_date = pd.Timestamp(payload.get("exit_date")).normalize()
        status_base = {
            "trade_id": trade_id,
            "symbol": symbol,
            "side": payload.get("side"),
            "entry_date": entry_date,
            "exit_date": exit_date,
        }
        try:
            selected = retriever.select_rule_atm_entry(payload)
        except Exception as exc:
            return pd.DataFrame(), pd.DataFrame(), {**status_base, "status": "entry_error", "message": str(exc)}
        if selected is None or selected.empty:
            return pd.DataFrame(), pd.DataFrame(), {**status_base, "status": "no_entry_candidates", "message": ""}
        priced, paths = retriever.price_selected_options_with_paths(selected, selector_name=self.selector_name)
        if priced.empty:
            contract = selected["contract_symbol"].iloc[0] if "contract_symbol" in selected.columns and len(selected) else ""
            return (
                pd.DataFrame(),
                pd.DataFrame(),
                {**status_base, "status": "selected_unpriced", "contract_symbol": contract, "message": ""},
            )
        first = priced.iloc[0]
        status = {
            **status_base,
            "status": "selected_priced",
            "contract_symbol": first.get("contract_symbol"),
            "option_exit_date": first.get("option_exit_date"),
            "expiration": first.get("expiration"),
            "option_action": first.get("option_action"),
            "option_return": first.get("option_return"),
            "path_observations": first.get("path_observations", np.nan),
            "message": "",
        }
        return priced, paths, status


def execute_rule_trade_list(
    trade_list: pd.DataFrame,
    retriever: OptionSelectionRetriever,
    *,
    selector_name: str = "rule_atm_90d",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    batch = OptionTradeExecutor(lambda: retriever, selector_name=selector_name, workers=1).execute(trade_list)
    return batch.selected_option_trades, batch.selected_option_paths, batch.trade_status


def _split_frame_for_workers(frame: pd.DataFrame, workers: int) -> list[pd.DataFrame]:
    worker_count = min(max(1, int(workers)), max(1, len(frame)))
    positions = np.array_split(np.arange(len(frame)), worker_count)
    return [frame.iloc[pos].copy() for pos in positions if len(pos)]


def _merge_option_execution_batches(batches: list[OptionTradeExecutionBatch]) -> OptionTradeExecutionBatch:
    if not batches:
        return OptionTradeExecutionBatch(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {})
    selected = [batch.selected_option_trades for batch in batches if not batch.selected_option_trades.empty]
    paths = [batch.selected_option_paths for batch in batches if not batch.selected_option_paths.empty]
    statuses = [batch.trade_status for batch in batches if not batch.trade_status.empty]
    return OptionTradeExecutionBatch(
        pd.concat(selected, ignore_index=True) if selected else pd.DataFrame(),
        pd.concat(paths, ignore_index=True) if paths else pd.DataFrame(),
        pd.concat(statuses, ignore_index=True) if statuses else pd.DataFrame(),
        _sum_metric_dicts([batch.metrics for batch in batches]),
    )


def _sum_metric_dicts(metrics: list[dict[str, float]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for metric in metrics:
        for key, value in metric.items():
            if isinstance(value, (int, float, np.integer, np.floating)) and pd.notna(value):
                out[key] = out.get(key, 0.0) + float(value)
    return out
