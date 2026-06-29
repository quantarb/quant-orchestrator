from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd


WindowMode = Literal["fixed", "anchored", "rolling"]


@dataclass(frozen=True)
class UniverseSplit:
    """Symbol universes used by a research or production experiment."""

    train: tuple[str, ...] = ()
    validation: tuple[str, ...] = ()
    backtest: tuple[str, ...] = ()
    paper: tuple[str, ...] = ()

    def train_symbols(self) -> tuple[str, ...] | None:
        return _normalize_symbols(self.train)

    def backtest_symbols(self) -> tuple[str, ...] | None:
        return _normalize_symbols(self.backtest)


@dataclass(frozen=True)
class WindowSpec:
    """Walk-forward window definition.

    `fixed` emits one train/test split.
    `anchored` keeps `train_start` fixed and advances each test window.
    `rolling` advances both train and test windows with a fixed train length.
    """

    mode: WindowMode = "fixed"
    train_start: str = "2020-01-01"
    train_end: str = "2020-12-31"
    test_start: str = "2021-01-01"
    test_end: str | None = None
    step: str = "30D"
    train_lookback: str = "365D"
    test_length: str = "30D"


@dataclass(frozen=True)
class WalkForwardWindow:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    @property
    def label(self) -> str:
        return (
            f"train_{self.train_start.date()}_{self.train_end.date()}"
            f"_test_{self.test_start.date()}_{self.test_end.date()}"
        )


@dataclass(frozen=True)
class MonteCarloSpec:
    enabled: bool = False
    iterations: int = 1_000
    horizon: int | None = None
    seed: int = 1337
    block_size: int = 1


def build_walk_forward_windows(spec: WindowSpec) -> list[WalkForwardWindow]:
    train_start = pd.Timestamp(spec.train_start).normalize()
    train_end = pd.Timestamp(spec.train_end).normalize()
    test_start = pd.Timestamp(spec.test_start).normalize()
    test_end = (
        pd.Timestamp(spec.test_end).normalize()
        if spec.test_end is not None
        else test_start + pd.Timedelta(spec.test_length) - pd.Timedelta(days=1)
    )
    step = pd.Timedelta(spec.step)
    test_length = pd.Timedelta(spec.test_length)
    train_lookback = pd.Timedelta(spec.train_lookback)

    if spec.mode == "fixed":
        return [WalkForwardWindow(train_start, train_end, test_start, test_end)]

    windows = []
    current_test_start = test_start
    while current_test_start <= test_end:
        current_test_end = min(current_test_start + test_length - pd.Timedelta(days=1), test_end)
        current_train_end = current_test_start - pd.Timedelta(days=1)
        current_train_start = (
            train_start if spec.mode == "anchored" else current_train_end - train_lookback
        )
        current_train_start = max(current_train_start.normalize(), train_start)
        if current_train_start <= current_train_end:
            windows.append(
                WalkForwardWindow(
                    current_train_start,
                    current_train_end.normalize(),
                    current_test_start.normalize(),
                    current_test_end.normalize(),
                )
            )
        current_test_start = current_test_start + step

    return windows


def _normalize_symbols(symbols: tuple[str, ...]) -> tuple[str, ...] | None:
    cleaned = tuple(sorted({symbol.strip().upper() for symbol in symbols if symbol.strip()}))
    return cleaned or None
