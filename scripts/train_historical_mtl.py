"""Train the compact historical-document MTL baseline and prototype map.

The next-subtoken target is constructed strictly within each
``(symbol, feature_family)`` history, so its context is directional.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from quant_orchestrator.platforms.ml_frameworks.torch.models.transformers.multirate.entity_embeddings import (
    GlobalEntityEmbedding,
    GlobalEntityVocabulary,
)


FEATURES = [
    "document_mean", "document_std", "document_min", "document_max",
    "document_abs_mean", "document_coverage", "daily_observations",
]
TASK_COLUMNS = ("feature_family", "sector", "subsector", "industry")
ENTITY_COLUMNS = (
    "symbol", "feature_family", "sector", "subsector", "industry",
    "event_type", "actor_type", "actor_name", "actor_firm", "actor_role",
    "actor_chamber", "source",
)


class HistoricalMTL(nn.Module):
    def __init__(self, class_counts: dict[str, int], year_count: int, entity_vocabulary_size: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(9, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Linear(128, 64), nn.LayerNorm(64), nn.GELU(),
        )
        self.entity_embedding = GlobalEntityEmbedding(entity_vocabulary_size, 64)
        self.heads = nn.ModuleDict({name: nn.Linear(64, count) for name, count in class_counts.items()})
        self.year_embedding = nn.Embedding(year_count, 64)
        self.next_subtoken = nn.Linear(64, 1)
        self.masked_token = nn.Linear(64, 1)

    def forward(self, values: torch.Tensor, entity_ids: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        state = self.backbone(values)
        if entity_ids is not None:
            state = state + self.entity_embedding(entity_ids)
        outputs = {name: head(state) for name, head in self.heads.items()}
        outputs["next_subtoken"] = self.next_subtoken(state).squeeze(-1)
        outputs["masked_token"] = self.masked_token(state).squeeze(-1)
        return state, outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--documents", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--target-events", type=Path, default=None)
    parser.add_argument("--earnings-events", type=Path, default=None)
    parser.add_argument("--target-family-events", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8192)
    args = parser.parse_args()

    args.artifact_root.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(args.documents).merge(pd.read_csv(args.taxonomy), on="symbol", how="left")
    for column in ("sector", "subsector", "industry"):
        frame[column] = frame[column].fillna("Unknown").astype(str)
    frame = frame.sort_values(["symbol", "feature_family", "year"]).reset_index(drop=True)
    grouped = frame.groupby(["symbol", "feature_family"], sort=False)
    frame["previous_mean"] = grouped["document_mean"].shift(1)
    frame["next_mean"] = grouped["document_mean"].shift(-1)
    frame["has_next"] = grouped["year"].shift(-1).notna()
    # A same-date document contains one token per observed feature family.
    # The masked token may use all other families from that same date, but no
    # future date is introduced into this context.
    date_group = frame.groupby(["symbol", "year"], sort=False)["document_mean"]
    date_sum = date_group.transform("sum")
    date_count = date_group.transform("count")
    frame["same_date_family_mean"] = ((date_sum - frame["document_mean"]) / (date_count - 1).clip(lower=1)).fillna(0.0)
    frame["document_id"] = pd.factorize(pd.MultiIndex.from_frame(frame[["symbol", "year"]]))[0]
    frame["document_family_count"] = date_count.astype("int64")

    target_documents = pd.DataFrame()
    target_event_index = pd.DataFrame()
    if args.target_events is not None and args.target_events.exists():
        events = pd.read_parquet(args.target_events).copy()
        events["event_date"] = pd.to_datetime(events["event_date"], errors="coerce", utc=True)
        events = events.loc[events["event_date"].notna()].copy()
        events["reported_date"] = pd.to_datetime(events.get("reported_date"), errors="coerce", utc=True)
        events["source_event_id"] = np.arange(len(events), dtype="int64")
        for column in ("transaction_value", "transaction_shares", "transaction_price"):
            events[column] = pd.to_numeric(events.get(column), errors="coerce").fillna(0.0)

        happened = events.copy()
        happened["event_role"] = "happened"
        happened["event_timestamp"] = happened["event_date"]
        happened["availability_timestamp"] = happened["reported_date"].fillna(happened["event_date"])
        disclosed = events.loc[events["reported_date"].notna()].copy()
        disclosed["event_role"] = "disclosed"
        disclosed["event_timestamp"] = disclosed["reported_date"]
        disclosed["availability_timestamp"] = disclosed["reported_date"]
        event_rows = pd.concat([happened, disclosed], ignore_index=True)
        event_rows["year"] = event_rows["event_timestamp"].dt.year.astype(int)
        event_rows["feature_family"] = "equity.ownership.insider_trading"
        target_event_index = event_rows[[
            "source_event_id", "symbol", "event_role", "event_timestamp", "availability_timestamp",
        ]].sort_values(["symbol", "event_timestamp", "event_role"]).reset_index(drop=True)
        target_documents = event_rows.groupby(["symbol", "year", "event_role"], sort=True).agg(
            document_mean=("transaction_value", "mean"),
            document_std=("transaction_value", "std"),
            document_min=("transaction_value", "min"),
            document_max=("transaction_value", "max"),
            document_abs_mean=("transaction_value", lambda values: values.abs().mean()),
            document_coverage=("transaction_value", lambda values: values.notna().mean()),
            daily_observations=("transaction_value", "size"),
        ).reset_index()
        target_documents["document_std"] = target_documents["document_std"].fillna(0.0)
        target_documents["feature_family"] = "equity.ownership.insider_trading"
        target_documents["sector"] = "Unknown"
        target_documents["subsector"] = "Unknown"
        target_documents["industry"] = "Unknown"

    if args.earnings_events is not None and args.earnings_events.exists():
        earnings = pd.read_parquet(args.earnings_events).copy()
        earnings["report_date"] = pd.to_datetime(earnings["report_date"], errors="coerce", utc=True)
        earnings = earnings.loc[earnings["report_date"].notna()].copy()
        earnings["symbol"] = earnings["symbol"].astype(str).str.upper()
        numeric = ["eps_consensus", "eps_actual", "revenue_consensus", "revenue_actual"]
        for column in numeric:
            if column not in earnings:
                earnings[column] = 0.0
            earnings[column] = pd.to_numeric(earnings[column], errors="coerce")
        # Keep the endpoint values in the source artifact; use actual EPS as
        # the compact scalar for the existing prototype heads and fall back to
        # actual revenue where EPS is unavailable.
        earnings["signal_value"] = earnings["eps_actual"].fillna(earnings["revenue_actual"]).fillna(0.0)
        earnings["event_role"] = "happened"
        earnings["event_timestamp"] = earnings["report_date"]
        earnings["availability_timestamp"] = earnings["report_date"]
        earnings["year"] = earnings["report_date"].dt.year.astype(int)
        earnings["feature_family"] = "equity.calendar.earnings"
        earnings["source_event_id"] = np.arange(len(earnings), dtype="int64")
        earnings_target_documents = earnings.groupby(["symbol", "year", "event_role"], sort=True).agg(
            document_mean=("signal_value", "mean"),
            document_std=("signal_value", "std"),
            document_min=("signal_value", "min"),
            document_max=("signal_value", "max"),
            document_abs_mean=("signal_value", lambda values: values.abs().mean()),
            document_coverage=("signal_value", lambda values: values.notna().mean()),
            daily_observations=("signal_value", "size"),
        ).reset_index()
        earnings_target_documents["document_std"] = earnings_target_documents["document_std"].fillna(0.0)
        earnings_target_documents["feature_family"] = "equity.calendar.earnings"
        earnings_target_documents["sector"] = "Unknown"
        earnings_target_documents["subsector"] = "Unknown"
        earnings_target_documents["industry"] = "Unknown"
        target_documents = pd.concat([target_documents, earnings_target_documents], ignore_index=True)
        earnings[["source_event_id", "symbol", "event_role", "event_timestamp", "availability_timestamp"]].to_parquet(
            args.artifact_root / "equity_calendar_earnings_event_index.parquet", index=False,
        )

    if args.target_family_events is not None and args.target_family_events.exists():
        endpoint_events = pd.read_parquet(args.target_family_events).copy()
        endpoint_events["event_date"] = pd.to_datetime(endpoint_events["event_date"], errors="coerce", utc=True)
        endpoint_events["reported_date"] = pd.to_datetime(endpoint_events.get("reported_date"), errors="coerce", utc=True)
        endpoint_events = endpoint_events.loc[
            endpoint_events["event_date"].notna() & endpoint_events["target_family"].notna()
        ].copy()
        endpoint_events["symbol"] = endpoint_events["symbol"].astype(str).str.upper()
        numeric_candidates = [
            "strength", "transaction_value", "transaction_shares", "transaction_price",
            "surprise", "target_value", "split_ratio", "dividend_value", "delta", "amount",
        ]
        numeric_values = []
        for column in numeric_candidates:
            if column in endpoint_events:
                numeric_values.append(pd.to_numeric(endpoint_events[column], errors="coerce"))
        if numeric_values:
            endpoint_events["signal_value"] = pd.concat(numeric_values, axis=1).bfill(axis=1).iloc[:, 0]
        else:
            endpoint_events["signal_value"] = np.nan
        endpoint_events["signal_observed"] = endpoint_events["signal_value"].notna()
        endpoint_events["signal_value"] = endpoint_events["signal_value"].fillna(0.0)
        endpoint_events["source_event_id"] = np.arange(len(endpoint_events), dtype="int64")

        happened = endpoint_events.copy()
        happened["event_role"] = "happened"
        happened["event_timestamp"] = happened["event_date"]
        happened["availability_timestamp"] = happened["reported_date"].fillna(happened["event_date"])
        disclosed = endpoint_events.loc[endpoint_events["reported_date"].notna()].copy()
        disclosed["event_role"] = "disclosed"
        disclosed["event_timestamp"] = disclosed["reported_date"]
        disclosed["availability_timestamp"] = disclosed["reported_date"]
        endpoint_events = pd.concat([happened, disclosed], ignore_index=True)
        endpoint_events["year"] = endpoint_events["event_timestamp"].dt.year.astype(int)
        endpoint_documents = endpoint_events.groupby(
            ["symbol", "year", "target_family", "event_role"], sort=True,
        ).agg(
            document_mean=("signal_value", "mean"),
            document_std=("signal_value", "std"),
            document_min=("signal_value", "min"),
            document_max=("signal_value", "max"),
            document_abs_mean=("signal_value", lambda values: values.abs().mean()),
            document_coverage=("signal_observed", "mean"),
            daily_observations=("signal_value", "size"),
        ).reset_index()
        endpoint_documents["document_std"] = endpoint_documents["document_std"].fillna(0.0)
        endpoint_documents["feature_family"] = endpoint_documents["target_family"]
        endpoint_documents["sector"] = "Unknown"
        endpoint_documents["subsector"] = "Unknown"
        endpoint_documents["industry"] = "Unknown"
        endpoint_keys = ["symbol", "year", "target_family", "event_role"]
        endpoint_entity_columns = [column for column in ENTITY_COLUMNS if column in endpoint_events.columns and column not in endpoint_keys]
        if endpoint_entity_columns:
            entity_values = endpoint_events.groupby(endpoint_keys, sort=True)[endpoint_entity_columns].agg(
                lambda values: next((value for value in values if pd.notna(value)), None)
            ).reset_index()
            endpoint_documents = endpoint_documents.merge(entity_values, on=endpoint_keys, how="left", validate="one_to_one")
        target_documents = pd.concat([target_documents, endpoint_documents], ignore_index=True)
        endpoint_events[[
            "source_event_id", "symbol", "target_family", "event_role",
            "event_timestamp", "availability_timestamp",
        ]].to_parquet(args.artifact_root / "all_target_family_event_index.parquet", index=False)

    entity_vocabulary = GlobalEntityVocabulary()
    entity_vocabulary.fit_frame(frame, ENTITY_COLUMNS)
    if args.target_family_events is not None and args.target_family_events.exists():
        entity_vocabulary.fit_frame(endpoint_events, ENTITY_COLUMNS)
    frame_entity_ids = entity_vocabulary.encode_rows(frame, ENTITY_COLUMNS)
    target_entity_ids = entity_vocabulary.encode_rows(target_documents, ENTITY_COLUMNS) if not target_documents.empty else np.empty((0, len(ENTITY_COLUMNS)), dtype="int64")

    values = frame[FEATURES].to_numpy("float32")
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std[std < 1e-6] = 1.0
    values = np.clip((values - mean) / std, -8.0, 8.0)
    previous = frame["previous_mean"].fillna(0.0).to_numpy("float32")
    previous = np.nan_to_num(previous, nan=0.0, posinf=0.0, neginf=0.0)
    previous = np.clip((previous - mean[0]) / std[0], -8.0, 8.0)
    same_date = frame["same_date_family_mean"].fillna(0.0).to_numpy("float32")
    same_date = np.nan_to_num(same_date, nan=0.0, posinf=0.0, neginf=0.0)
    same_date = np.clip((same_date - mean[0]) / std[0], -8.0, 8.0)
    model_values = np.concatenate([values, previous[:, None], same_date[:, None]], axis=1)
    masked_values = model_values.copy()
    masked_values[:, 0] = 0.0

    labels: dict[str, list[str]] = {}
    tensors: list[torch.Tensor] = [
        torch.from_numpy(model_values), torch.from_numpy(masked_values),
        torch.from_numpy(frame_entity_ids), torch.from_numpy(frame_entity_ids.copy()),
    ]
    class_counts: dict[str, int] = {}
    for column in TASK_COLUMNS:
        label_values = set(frame[column].unique())
        if column == "feature_family" and not target_documents.empty:
            label_values.update(target_documents[column].unique())
        labels[column] = sorted(label_values)
        encoded = frame[column].map({value: i for i, value in enumerate(labels[column])}).to_numpy("int64")
        tensors.append(torch.from_numpy(encoded))
        class_counts[column] = len(labels[column])
    family_mapping = {value: i for i, value in enumerate(labels["feature_family"])}
    family_inputs = torch.from_numpy(model_values)
    family_targets = frame["feature_family"].map(family_mapping).to_numpy("int64")
    if not target_documents.empty:
        target_values = target_documents[FEATURES].to_numpy("float32")
        target_values = np.nan_to_num(target_values, nan=0.0, posinf=0.0, neginf=0.0)
        target_values = np.clip((target_values - mean) / std, -8.0, 8.0)
        target_inputs = np.concatenate([target_values, np.zeros((len(target_values), 2), dtype="float32")], axis=1)
        family_inputs = torch.cat([family_inputs, torch.from_numpy(target_inputs)], dim=0)
        family_entity_ids = torch.cat([torch.from_numpy(frame_entity_ids), torch.from_numpy(target_entity_ids)], dim=0)
        family_targets = np.concatenate([
            family_targets,
            target_documents["feature_family"].map(family_mapping).to_numpy("int64"),
        ])
    years = sorted(frame["year"].astype(int).unique())
    year_codes = {value: i for i, value in enumerate(years)}
    cross_sectional = frame.groupby("year", sort=True)[FEATURES].mean().reindex(years)
    cross_values = cross_sectional.to_numpy("float32")
    cross_values = np.nan_to_num(cross_values, nan=0.0, posinf=0.0, neginf=0.0)
    cross_values = np.clip((cross_values - mean) / std, -8.0, 8.0)
    cross_inputs = np.concatenate([cross_values, np.zeros((len(years), 2), dtype="float32")], axis=1)
    cross_targets = torch.arange(len(years), dtype=torch.long)
    target = frame["next_mean"].fillna(0.0).to_numpy("float32")
    target = np.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
    target_mean, target_std = float(target.mean()), float(target.std() or 1.0)
    target = np.clip((target - target_mean) / target_std, -8.0, 8.0)
    masked_target = values[:, 0].copy()
    tensors.extend([
        torch.from_numpy(target), torch.from_numpy(frame["has_next"].to_numpy()),
        torch.from_numpy(masked_target),
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_counts["cross_sectional_year"] = len(years)
    if target_documents.empty:
        family_entity_ids = torch.from_numpy(frame_entity_ids)
    model = HistoricalMTL(class_counts, len(years), entity_vocabulary.size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(*tensors), batch_size=args.batch_size, shuffle=True, pin_memory=True)
    cross_loader = DataLoader(TensorDataset(torch.from_numpy(cross_inputs), cross_targets), batch_size=min(args.batch_size, len(years)), shuffle=True)
    family_loader = DataLoader(TensorDataset(family_inputs, family_entity_ids, torch.from_numpy(family_targets)), batch_size=args.batch_size, shuffle=True, pin_memory=True)
    losses: list[float] = []
    cross_losses: list[float] = []
    family_losses: list[float] = []
    model.train()
    for epoch in range(args.epochs):
        total = 0.0
        count = 0
        for batch in loader:
            batch = [part.to(device, non_blocking=True) for part in batch]
            values_batch, masked_batch, entity_batch, masked_entity_batch, family, sector, subsector, industry, next_value, has_next, masked_target_batch = batch
            _, output = model(values_batch, entity_batch)
            loss = sum(
                nn.functional.cross_entropy(output[name], target_value)
                for name, target_value in {
                    "feature_family": family, "sector": sector,
                    "subsector": subsector, "industry": industry,
                }.items()
            )
            if has_next.any():
                loss = loss + nn.functional.mse_loss(output["next_subtoken"][has_next], next_value[has_next])
            _, masked_output = model(masked_batch, masked_entity_batch)
            loss = loss + nn.functional.mse_loss(masked_output["masked_token"], masked_target_batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss) * len(values_batch)
            count += len(values_batch)
        cross_total = 0.0
        cross_count = 0
        for cross_batch in cross_loader:
            cross_values_batch, cross_target_batch = [part.to(device) for part in cross_batch]
            _, cross_output = model(cross_values_batch)
            cross_loss = nn.functional.cross_entropy(cross_output["cross_sectional_year"], cross_target_batch)
            optimizer.zero_grad(set_to_none=True)
            cross_loss.backward()
            optimizer.step()
            cross_total += float(cross_loss) * len(cross_values_batch)
            cross_count += len(cross_values_batch)
        family_total = 0.0
        family_count = 0
        for family_batch in family_loader:
            family_values_batch, family_entity_batch, family_target_batch = [part.to(device, non_blocking=True) for part in family_batch]
            family_state, _ = model(family_values_batch, family_entity_batch)
            family_logits = model.heads["feature_family"](family_state)
            family_loss = nn.functional.cross_entropy(family_logits, family_target_batch)
            optimizer.zero_grad(set_to_none=True)
            family_loss.backward()
            optimizer.step()
            family_total += float(family_loss) * len(family_values_batch)
            family_count += len(family_values_batch)
        losses.append(total / count)
        cross_losses.append(cross_total / cross_count)
        family_losses.append(family_total / family_count)
        print(f"epoch {epoch + 1}/{args.epochs} loss={losses[-1]:.6f} cross_sectional_year={cross_losses[-1]:.6f} family_union={family_losses[-1]:.6f}", flush=True)

    checkpoint = {
        "state_dict": model.state_dict(), "labels": {**labels, "year": [str(value) for value in years],
                                                        "cross_sectional_year": [str(value) for value in years]},
        "tasks": [*TASK_COLUMNS, "year", "cross_sectional_year", "next_subtoken", "masked_token"], "feature_columns": FEATURES,
        "metrics": {"device": str(device), "documents": len(frame), "losses": losses, "cross_sectional_losses": cross_losses, "family_union_losses": family_losses,
                     "next_subtoken_examples": int(frame["has_next"].sum()),
                     "masked_token_examples": len(frame), "same_date_documents": int(frame["document_id"].nunique()),
                     "cross_sectional_documents": len(years),
                     "target_family_documents": len(target_documents),
                     "target_mean": target_mean,
                     "target_std": target_std,
                     "global_entity_vocabulary_size": entity_vocabulary.size},
        "entity_vocabulary": entity_vocabulary.state(),
    }
    torch.save(checkpoint, args.artifact_root / "historical_mtl_model.pt")
    frame[["document_id", "symbol", "year", "feature_family", "document_family_count"]].assign(
        subtokens=frame["daily_observations"], same_date_family_mean=frame["same_date_family_mean"],
    ).to_parquet(
        args.artifact_root / "historical_document_index.parquet", index=False,
    )
    if not target_event_index.empty:
        target_event_index.to_parquet(args.artifact_root / "equity_ownership_insider_trading_event_index.parquet", index=False)
    pd.DataFrame({"year": years, "token_count": frame.groupby("year").size().reindex(years).to_numpy()}).to_parquet(
        args.artifact_root / "cross_sectional_year_corpus.parquet", index=False,
    )
    (args.artifact_root / "training_summary.json").write_text(json.dumps(checkpoint["metrics"], indent=2))

    rows: list[dict[str, object]] = []
    vectors: list[np.ndarray] = []
    state = model.state_dict()
    for task, column in (("family", "feature_family"), ("sector", "sector"), ("subsector", "subsector"), ("industry", "industry")):
        weights = state[f"heads.{column}.weight"].detach().cpu().numpy()
        for index, label in enumerate(labels[column]):
            support = int((frame[column] == label).sum())
            if column == "feature_family" and not target_documents.empty:
                support += int((target_documents[column] == label).sum())
            rows.append({"task": task, "label": label, "support": support})
            vectors.append(weights[index])
    year_weights = state["year_embedding.weight"].detach().cpu().numpy()
    for index, year in enumerate(years):
        rows.append({"task": "year", "label": str(year), "support": int((frame["year"] == year).sum())})
        vectors.append(year_weights[index])
    cross_weights = state["heads.cross_sectional_year.weight"].detach().cpu().numpy()
    for index, year in enumerate(years):
        rows.append({"task": "cross_sectional_year", "label": str(year), "support": int((frame["year"] == year).sum())})
        vectors.append(cross_weights[index])
    coordinates = TSNE(n_components=3, perplexity=min(30.0, len(vectors) - 1), init="pca",
                       learning_rate="auto", max_iter=1500, random_state=42).fit_transform(np.asarray(vectors, np.float32))
    plot_frame = pd.DataFrame(rows)
    plot_frame[["x", "y", "z"]] = coordinates
    plot_dir = args.artifact_root / "all_prototypes_tsne_3d"
    plot_dir.mkdir(exist_ok=True)
    plot_frame.to_csv(plot_dir / "all_prototype_coordinates_3d.csv", index=False)

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    styles = {"family": ("tab:blue", "o", "family prototypes"), "sector": ("tab:orange", "s", "sector prototypes"),
              "subsector": ("tab:green", "^", "subsector prototypes"), "industry": ("tab:red", "D", "industry prototypes"),
              "year": ("tab:purple", "P", "year prototypes"),
              "cross_sectional_year": ("tab:cyan", "X", "cross-sectional year prototypes")}
    figure = plt.figure(figsize=(15, 15), dpi=180)
    axis = figure.add_subplot(111, projection="3d")
    for task, (color, marker, _) in styles.items():
        part = plot_frame[plot_frame.task == task]
        axis.scatter(part.x, part.y, part.z, s=28, color=color, marker=marker, alpha=0.85, depthshade=True)
        for _, row in part.iterrows():
            axis.text(row.x, row.y, row.z, row.label, fontsize=4.0, alpha=0.85)
    axis.legend(handles=[Line2D([0], [0], marker=marker, color="w", markerfacecolor=color, markersize=7, label=label)
                        for color, marker, label in styles.values()], loc="upper left", bbox_to_anchor=(0.01, 0.99))
    axis.set_title("10B all prototypical classes — 3D t-SNE")
    axis.set_xlabel("t-SNE 1"); axis.set_ylabel("t-SNE 2"); axis.set_zlabel("t-SNE 3")
    figure.tight_layout(); figure.savefig(plot_dir / "all_prototype_tsne_3d.png", bbox_inches="tight"); plt.close(figure)
    (plot_dir / "metadata.json").write_text(json.dumps({"tasks": checkpoint["tasks"], "n_prototypes": len(rows),
        "prototype_counts": plot_frame.groupby("task").size().to_dict(), "label_policy": "label every plotted point"}, indent=2))


if __name__ == "__main__":
    main()
