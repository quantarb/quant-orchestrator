"""Target-free classification of feature-family documents.

The document label is the source feature family itself.  This module does not
load trading labels, target-engineering data, or trading-score projections.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
import torch
from torch.utils.data import DataLoader, TensorDataset

from quant_orchestrator.platforms.ml_frameworks.torch.models.transformers.multirate import (
    MultiRateTaskSpec,
    MultiRateTransformer,
    MultiRateTransformerConfig,
)


DOCUMENT_FEATURES = (
    "document_mean",
    "document_std",
    "document_min",
    "document_max",
    "document_abs_mean",
    "document_coverage",
)


@dataclass(frozen=True)
class FeatureFamilyClassificationConfig:
    train_end: str = "2020-12-31"
    score_start: str = "2021-01-01"
    random_seed: int = 20260702
    min_feature_coverage: float = 0.50
    epochs: int = 8
    batch_size: int = 8192
    learning_rate: float = 0.001
    d_model: int = 64
    num_heads: int = 4
    layers: int = 2
    device: str = "auto"


def build_feature_family_documents(
    index_path: str | Path,
    *,
    excluded_families: set[str] | None = None,
    config: FeatureFamilyClassificationConfig | None = None,
) -> pd.DataFrame:
    """Build one target-free document row per ``(symbol, date, family)``."""

    cfg = config or FeatureFamilyClassificationConfig()
    index = pd.read_csv(index_path)
    excluded = {str(value) for value in (excluded_families or set())}
    index = index.loc[~index["family"].astype(str).isin(excluded)].copy()
    frames: list[pd.DataFrame] = []
    for row in index.sort_values(["source", "family"]).to_dict("records"):
        family = f"{row['source']}.{row['family']}"
        if "option" in family.lower():
            continue
        panel = pd.read_parquet(row["panel_path"])
        metadata = pd.read_parquet(row["metadata_path"])
        features = tuple(
            str(value)
            for value in metadata.loc[
                metadata["feature"].isin(panel.columns), "feature"
            ].drop_duplicates()
        )
        if not features:
            continue
        numeric = (
            panel[list(features)]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
        )
        coverage = numeric.notna().mean(axis=1)
        keep = coverage.ge(float(cfg.min_feature_coverage))
        numeric = numeric.loc[keep]
        if numeric.empty:
            continue
        document = pd.DataFrame(
            {
                "symbol": panel.loc[keep, "symbol"].astype(str).str.upper().to_numpy(),
                "date": pd.to_datetime(panel.loc[keep, "date"], errors="coerce")
                .dt.normalize()
                .to_numpy(),
                "feature_family": family,
                "document_mean": numeric.mean(axis=1).to_numpy(),
                "document_std": numeric.std(axis=1).fillna(0.0).to_numpy(),
                "document_min": numeric.min(axis=1).to_numpy(),
                "document_max": numeric.max(axis=1).to_numpy(),
                "document_abs_mean": numeric.abs().mean(axis=1).to_numpy(),
                "document_coverage": numeric.notna().mean(axis=1).to_numpy(),
            }
        )
        frames.append(document.dropna(subset=["date"]))
    if not frames:
        return pd.DataFrame(columns=["symbol", "date", "feature_family", *DOCUMENT_FEATURES])
    return pd.concat(frames, ignore_index=True).sort_values(
        ["date", "symbol", "feature_family"]
    ).reset_index(drop=True)


def run_feature_family_classification(
    documents: pd.DataFrame,
    output_dir: str | Path,
    *,
    config: FeatureFamilyClassificationConfig | None = None,
) -> dict[str, pd.DataFrame | Path]:
    """Train and evaluate a family-identity classifier."""

    cfg = config or FeatureFamilyClassificationConfig()
    required = {"date", "feature_family", *DOCUMENT_FEATURES}
    missing = required.difference(documents.columns)
    if missing:
        raise KeyError(f"family documents missing columns: {sorted(missing)}")
    frame = documents.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame = frame.dropna(subset=["date", "feature_family"])
    train = frame.loc[frame["date"].le(pd.Timestamp(cfg.train_end))].copy()
    oos = frame.loc[frame["date"].ge(pd.Timestamp(cfg.score_start))].copy()
    feature_columns = list(DOCUMENT_FEATURES)
    medians = train[feature_columns].median().fillna(0.0)
    train.loc[:, feature_columns] = train[feature_columns].fillna(medians)
    oos.loc[:, feature_columns] = oos[feature_columns].fillna(medians)
    means = train[feature_columns].mean().fillna(0.0)
    scales = train[feature_columns].std().replace(0.0, 1.0).fillna(1.0)
    train[feature_columns] = ((train[feature_columns] - means) / scales).astype("float32")
    oos[feature_columns] = ((oos[feature_columns] - means) / scales).astype("float32")
    # Keep OOS-only families in the report.  They cannot be learned without
    # pre-2021 examples, but silently dropping them hides coverage problems.
    labels = sorted(frame["feature_family"].astype(str).unique())
    label_to_id = {label: index for index, label in enumerate(labels)}
    train_x = torch.tensor(train[feature_columns].to_numpy(dtype="float32"))
    train_y = torch.tensor(train["feature_family"].map(label_to_id).to_numpy(dtype="int64"))
    if cfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)
    torch.manual_seed(int(cfg.random_seed))
    model = MultiRateTransformer(
        {"annual": len(feature_columns), "quarterly": len(feature_columns), "daily": len(feature_columns)},
        config=MultiRateTransformerConfig(
            backbone="encoder_only",
            d_model=int(cfg.d_model),
            num_heads=int(cfg.num_heads),
            layers=int(cfg.layers),
            document_pool="mean",
            max_position=1,
        ),
        tasks=(MultiRateTaskSpec("feature_family", "document", output_dim=len(labels), source="daily"),),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.learning_rate), weight_decay=1e-4)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=int(cfg.batch_size), shuffle=True)
    def inputs(batch_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # A family document is one daily document token.  The other rates are
        # empty context placeholders; no values are duplicated across rates.
        batch_x = batch_x.to(device).unsqueeze(1)
        empty = torch.zeros_like(batch_x)
        return batch_x, empty, empty

    model.train()
    for _ in range(int(cfg.epochs)):
        for batch_x, batch_y in loader:
            daily_x, quarterly_x, annual_x = inputs(batch_x)
            optimizer.zero_grad(set_to_none=True)
            output = model(
                daily_x,
                annual_x,
                quarterly_x,
                attention_mode="cross_sectional",
                daily_dates=torch.zeros(1, dtype=torch.long, device=device),
                annual_dates=torch.zeros(1, dtype=torch.long, device=device),
                quarterly_dates=torch.zeros(1, dtype=torch.long, device=device),
            )
            loss = torch.nn.functional.cross_entropy(
                output["document_outputs"]["feature_family"], batch_y.to(device)
            )
            loss.backward()
            optimizer.step()
    rows: list[dict[str, object]] = []
    for split, part in (("train", train), ("oos", oos)):
        model.eval()
        with torch.no_grad():
            part_x = torch.tensor(part[feature_columns].to_numpy(dtype="float32"))
            daily_x, quarterly_x, annual_x = inputs(part_x)
            logits = model(
                daily_x,
                annual_x,
                quarterly_x,
                attention_mode="cross_sectional",
                daily_dates=torch.zeros(1, dtype=torch.long, device=device),
                annual_dates=torch.zeros(1, dtype=torch.long, device=device),
                quarterly_dates=torch.zeros(1, dtype=torch.long, device=device),
            )["document_outputs"]["feature_family"]
            predicted_ids = logits.argmax(dim=1).cpu().numpy()
        predicted = np.asarray(labels, dtype=object)[predicted_ids]
        report = classification_report(
            part["feature_family"],
            predicted,
            labels=labels,
            output_dict=True,
            zero_division=0,
        )
        for label, values in report.items():
            if isinstance(values, dict):
                rows.append(
                    {
                        "split": split,
                        "feature_family": label,
                        "precision": float(values["precision"]),
                        "recall": float(values["recall"]),
                        "f1": float(values["f1-score"]),
                        "support": int(values["support"]),
                    }
                )
    report_frame = pd.DataFrame(rows)
    summary = pd.DataFrame(
        [
            {"split": "train", "rows": len(train), "families": train["feature_family"].nunique()},
            {"split": "oos", "rows": len(oos), "families": oos["feature_family"].nunique()},
        ]
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report_frame.to_csv(output / "classification_report.csv", index=False)
    summary.to_csv(output / "dataset_summary.csv", index=False)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "labels": labels,
            "feature_columns": feature_columns,
            "model_config": {
                "backbone": "encoder_only",
                "d_model": int(cfg.d_model),
                "num_heads": int(cfg.num_heads),
                "layers": int(cfg.layers),
                "document_pool": "mean",
            },
        },
        output / "document_classifier.pt",
    )
    (output / "config.json").write_text(
        json.dumps({**cfg.__dict__, "device_used": str(device)}, indent=2), encoding="utf-8"
    )
    return {"classification_report": report_frame, "dataset_summary": summary, "output_dir": output}


def run_multitask_taxonomy_classification(
    documents: pd.DataFrame,
    taxonomy: pd.DataFrame,
    output_dir: str | Path,
    *,
    config: FeatureFamilyClassificationConfig | None = None,
) -> dict[str, pd.DataFrame | Path]:
    """Train family, sector, subsector, and industry heads jointly.

    ``taxonomy`` is symbol metadata, not a trading target.  The four heads
    share one document encoder while retaining independent label vocabularies.
    """
    cfg = config or FeatureFamilyClassificationConfig()
    required = {"symbol", "date", "feature_family", *DOCUMENT_FEATURES}
    missing = required.difference(documents.columns)
    if missing:
        raise KeyError(f"family documents missing columns: {sorted(missing)}")
    taxonomy_required = {"symbol", "sector", "subsector", "industry"}
    missing = taxonomy_required.difference(taxonomy.columns)
    if missing:
        raise KeyError(f"taxonomy missing columns: {sorted(missing)}")
    frame = documents.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    taxonomy_frame = taxonomy[list(taxonomy_required)].copy()
    taxonomy_frame["symbol"] = taxonomy_frame["symbol"].astype(str).str.upper()
    frame = frame.merge(taxonomy_frame.drop_duplicates("symbol"), on="symbol", how="inner")
    train = frame.loc[frame["date"].le(pd.Timestamp(cfg.train_end))].copy()
    oos = frame.loc[frame["date"].ge(pd.Timestamp(cfg.score_start))].copy()
    feature_columns = list(DOCUMENT_FEATURES)
    medians = train[feature_columns].median().fillna(0.0)
    means = train[feature_columns].fillna(medians).mean().fillna(0.0)
    scales = train[feature_columns].fillna(medians).std().replace(0.0, 1.0).fillna(1.0)
    train[feature_columns] = ((train[feature_columns].fillna(medians) - means) / scales).astype("float32")
    oos[feature_columns] = ((oos[feature_columns].fillna(medians) - means) / scales).astype("float32")
    task_columns = {"feature_family": "feature_family", "sector": "sector", "subsector": "subsector", "industry": "industry"}
    labels = {task: sorted(frame[column].astype(str).unique()) for task, column in task_columns.items()}
    mappings = {task: {label: i for i, label in enumerate(values)} for task, values in labels.items()}
    device = torch.device("cuda" if cfg.device == "auto" and torch.cuda.is_available() else cfg.device if cfg.device != "auto" else "cpu")
    torch.manual_seed(int(cfg.random_seed))
    model = MultiRateTransformer(
        {"annual": len(feature_columns), "quarterly": len(feature_columns), "daily": len(feature_columns)},
        config=MultiRateTransformerConfig(backbone="encoder_only", d_model=int(cfg.d_model), num_heads=int(cfg.num_heads), layers=int(cfg.layers), document_pool="mean", max_position=1),
        tasks=tuple(MultiRateTaskSpec(task, "document", output_dim=len(values), source="daily") for task, values in labels.items()),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.learning_rate), weight_decay=1e-4)
    train_x = torch.tensor(train[feature_columns].to_numpy("float32"))
    train_y = {task: torch.tensor(train[column].astype(str).map(mappings[task]).to_numpy("int64")) for task, column in task_columns.items()}
    loader = DataLoader(TensorDataset(train_x, *[train_y[task] for task in task_columns]), batch_size=int(cfg.batch_size), shuffle=True)

    def inputs(batch_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        daily = batch_x.to(device).unsqueeze(1)
        empty = torch.zeros_like(daily)
        return daily, empty, empty

    model.train()
    for _ in range(int(cfg.epochs)):
        for batch in loader:
            batch_x, *batch_y = batch
            daily, quarterly, annual = inputs(batch_x)
            output = model(daily, annual, quarterly, attention_mode="cross_sectional", daily_dates=torch.zeros(1, dtype=torch.long, device=device), annual_dates=torch.zeros(1, dtype=torch.long, device=device), quarterly_dates=torch.zeros(1, dtype=torch.long, device=device))["document_outputs"]
            loss = sum(torch.nn.functional.cross_entropy(output[task], target.to(device)) for task, target in zip(task_columns, batch_y))
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()

    rows: list[dict[str, object]] = []
    for split, part in (("train", train), ("oos", oos)):
        model.eval()
        with torch.no_grad():
            all_predictions = {task: [] for task in task_columns}
            for start in range(0, len(part), int(cfg.batch_size)):
                batch_x = torch.tensor(part.iloc[start:start + int(cfg.batch_size)][feature_columns].to_numpy("float32"))
                daily, quarterly, annual = inputs(batch_x)
                outputs = model(daily, annual, quarterly, attention_mode="cross_sectional", daily_dates=torch.zeros(1, dtype=torch.long, device=device), annual_dates=torch.zeros(1, dtype=torch.long, device=device), quarterly_dates=torch.zeros(1, dtype=torch.long, device=device))["document_outputs"]
                for task in task_columns:
                    all_predictions[task].append(outputs[task].argmax(dim=1).cpu().numpy())
            for task, column in task_columns.items():
                predicted_ids = np.concatenate(all_predictions[task])
                predicted = np.asarray(labels[task], dtype=object)[predicted_ids]
                report = classification_report(part[column].astype(str), predicted, labels=labels[task], output_dict=True, zero_division=0)
                for label, values in report.items():
                    if isinstance(values, dict):
                        rows.append({"split": split, "task": task, "label": label, "precision": float(values["precision"]), "recall": float(values["recall"]), "f1": float(values["f1-score"]), "support": int(values["support"])})
    report_frame = pd.DataFrame(rows)
    summary = pd.DataFrame([{"split": "train", "rows": len(train), "families": train.feature_family.nunique()}, {"split": "oos", "rows": len(oos), "families": oos.feature_family.nunique()}])
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    report_frame.to_csv(output / "multitask_classification_report.csv", index=False); summary.to_csv(output / "dataset_summary.csv", index=False)
    torch.save({"state_dict": model.state_dict(), "labels": labels, "feature_columns": feature_columns, "tasks": list(task_columns), "model_config": {"backbone": "encoder_only", "d_model": int(cfg.d_model), "num_heads": int(cfg.num_heads), "layers": int(cfg.layers), "document_pool": "mean"}}, output / "multitask_document_classifier.pt")
    (output / "config.json").write_text(json.dumps({**cfg.__dict__, "device_used": str(device), "tasks": list(task_columns)}, indent=2), encoding="utf-8")
    return {"classification_report": report_frame, "dataset_summary": summary, "output_dir": output}
