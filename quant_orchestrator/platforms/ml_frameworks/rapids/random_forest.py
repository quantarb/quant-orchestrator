from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder


DEFAULT_RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": 16,
    "max_features": "sqrt",
    "n_bins": 128,
    "n_streams": 8,
}


@dataclass
class RapidsRandomForestClassifier:
    """Small adapter around cuML RandomForestClassifier for notebook experiments."""

    params: dict
    model: object
    encoder: LabelEncoder

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        *,
        features: Iterable[str],
        target_col: str,
        random_state: int,
        params: dict | None = None,
    ) -> "RapidsRandomForestClassifier":
        import cudf
        from cuml.ensemble import RandomForestClassifier as CuRandomForestClassifier

        feature_list = list(features)
        model_params = {**DEFAULT_RF_PARAMS, **dict(params or {}), "random_state": random_state}
        encoder = LabelEncoder()
        y = encoder.fit_transform(frame[target_col].astype(str))
        model = CuRandomForestClassifier(**model_params)
        model.fit(cudf.from_pandas(frame[feature_list].astype("float32")), cudf.Series(y.astype("int32")))
        return cls(params=model_params, model=model, encoder=encoder)

    def predict_proba_frame(self, frame: pd.DataFrame, features: Iterable[str]) -> pd.DataFrame:
        import cupy as cp
        import cudf

        feature_list = list(features)
        proba = self.model.predict_proba(cudf.from_pandas(frame[feature_list].astype("float32")))
        proba_np = proba.to_numpy() if hasattr(proba, "to_numpy") else cp.asnumpy(proba)
        return pd.DataFrame(
            proba_np,
            columns=[f"prob__{label}" for label in self.encoder.classes_],
            index=frame.index,
        )

    def score(
        self,
        frame: pd.DataFrame,
        *,
        features: Iterable[str],
        target_col: str,
    ) -> dict[str, float | int]:
        proba = self.predict_proba_frame(frame, features)
        class_cols = [f"prob__{label}" for label in self.encoder.classes_]
        y_true = frame[target_col].astype(str).to_numpy()
        y_pred = self.encoder.inverse_transform(np.asarray(proba[class_cols].to_numpy().argmax(axis=1)).astype(int))
        return {
            "rows": int(len(frame)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        }


def ensure_probability_columns(frame: pd.DataFrame, labels: Iterable[str]) -> pd.DataFrame:
    out = frame.copy()
    for label in labels:
        col = f"prob__{label}"
        if col not in out.columns:
            out[col] = 0.0
    return out
