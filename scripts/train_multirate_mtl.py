"""Train the four-rate MultiRateTransformer MTL corpus."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE
from torch import nn

from quant_orchestrator.platforms.ml_frameworks.torch.models.transformers.multirate import (
    MultiRatePredictionTaskSpec,
    MultiRateTaskSpec,
    MultiRateTransformer,
    MultiRateTransformerConfig,
)


DAILY_WINDOW = 252  # one trading year for daily self-supervision


def _encode_labels(values: pd.Series) -> tuple[np.ndarray, list[str], dict[str, int]]:
    labels = sorted(values.astype(str).fillna("Unknown").unique())
    mapping = {value: index for index, value in enumerate(labels)}
    return values.astype(str).fillna("Unknown").map(mapping).to_numpy("int64"), labels, mapping


def _window(table: pd.DataFrame, symbol: str, anchor: pd.Timestamp, value_columns: list[str], length: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = table.loc[(table["symbol"].eq(symbol)) & (table["date"].le(anchor))].tail(length)
    values = rows[value_columns].to_numpy("float32") if len(rows) else np.empty((0, len(value_columns)), dtype="float32")
    dates = pd.to_datetime(rows["date"], errors="coerce").to_numpy() if len(rows) else np.empty(0, dtype="datetime64[ns]")
    padding = np.ones(length, dtype=bool)
    output = np.full((length, len(value_columns)), np.nan, dtype="float32")
    if len(rows):
        output[-len(rows):] = values
        padding[-len(rows):] = False
    return output, padding, dates


def _sparse_window(table: pd.DataFrame, symbol: str, anchor: pd.Timestamp, value_columns: list[str], length: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = table.loc[(table["symbol"].eq(symbol)) & (table["date"].le(anchor))].tail(length)
    values = np.full((length, len(value_columns)), np.nan, dtype="float32")
    labels = np.full(length, -1, dtype="int64")
    padding = np.ones(length, dtype=bool)
    dates = np.empty(0, dtype="datetime64[ns]")
    if len(rows):
        values[-len(rows):] = rows[value_columns].to_numpy("float32")
        labels[-len(rows):] = rows["target_id"].to_numpy("int64")
        padding[-len(rows):] = False
        dates = pd.to_datetime(rows["date"], errors="coerce").to_numpy()
    return values, padding, labels, dates


def _relative_dates(dates: np.ndarray, length: int) -> torch.Tensor:
    # Windows are causal and left-padded. Relative ordering is shared across
    # a batch; sparse rows are aggregated to one row per availability date.
    result = torch.arange(length, dtype=torch.long)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--min-delta", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    root = args.corpus
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root / "manifest.json").read_text())
    feature_families = list(manifest["feature_families"])
    target_families = list(manifest["target_families"])
    annual = pd.read_parquet(root / "annual.parquet")
    quarterly = pd.read_parquet(root / "quarterly.parquet")
    daily = pd.read_parquet(root / "daily.parquet")
    sparse = pd.read_parquet(root / "sparse_events.parquet")
    taxonomy = pd.read_csv(root / "taxonomy.csv").set_index("symbol")
    for table in (annual, quarterly, daily, sparse):
        table["symbol"] = table["symbol"].astype(str).str.upper()
        table["date"] = pd.to_datetime(table["date"], errors="coerce", utc=True).dt.tz_localize(None)

    daily_value_columns = [f"value__{family}" for family in feature_families]
    annual_value_columns = daily_value_columns
    quarterly_value_columns = daily_value_columns
    sparse = sparse.sort_values(["symbol", "date", "event_date"] if "event_date" in sparse else ["symbol", "date"])
    # Multiple same-day disclosures become one sparse token while retaining
    # the first available family and the mean numeric/text representation.
    sparse["target_id"] = sparse["target_family"].astype(str).map({name: i for i, name in enumerate(target_families)})
    sparse_value_columns = ["signal_value", *[f"text_{i}" for i in range(7)]]
    sparse = sparse.groupby(["symbol", "date"], as_index=False).agg(
        target_family=("target_family", "first"), target_id=("target_id", "first"),
        **{column: (column, "mean") for column in sparse_value_columns},
    )

    # Standardize numeric rate values using the available corpus while
    # retaining NaN for coverage-aware missingness handling.
    rate_columns = {"annual": annual_value_columns, "quarterly": quarterly_value_columns, "daily": daily_value_columns}
    norms: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, columns in rate_columns.items():
        values = pd.concat([annual[columns], quarterly[columns], daily[columns]], ignore_index=True).to_numpy("float64")
        mean = np.nanmean(values, axis=0); scale = np.nanstd(values, axis=0)
        mean = np.nan_to_num(mean, nan=0.0); scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0)
        norms[name] = (mean.astype("float32"), scale.astype("float32"))
        for table in (annual, quarterly, daily):
            table.loc[:, columns] = ((table[columns].to_numpy("float32") - mean) / scale)
    sparse_numeric = sparse["signal_value"].to_numpy("float64")
    sparse["signal_value"] = ((sparse_numeric - np.nanmean(sparse_numeric)) / (np.nanstd(sparse_numeric) or 1.0)).astype("float32")
    sparse["date"] = pd.to_datetime(sparse["date"], errors="coerce").dt.normalize()
    raw_sparse_columns = ["signal_value", *[f"text_{i}" for i in range(7)]]
    sparse_value_columns: list[str] = []
    for family in target_families:
        for column in raw_sparse_columns:
            output_column = f"{family}__{column}"
            sparse[output_column] = sparse[column].where(sparse["target_family"].eq(family), np.nan)
            sparse_value_columns.append(output_column)

    # A document can be anchored by a regular annual observation or by a
    # sparse event.  Using their union preserves early event history even
    # when annual fundamentals begin later for a symbol.
    anchors = pd.concat([
        annual[["symbol", "date"]],
        sparse[["symbol", "date"]],
    ], ignore_index=True).drop_duplicates().sort_values(["symbol", "date"])
    anchors["year"] = pd.to_datetime(anchors["date"]).dt.year
    anchors = anchors.groupby(["symbol", "year"], as_index=False)["date"].max()
    samples: list[dict[str, object]] = []
    for row in anchors.itertuples(index=False):
        symbol = str(row.symbol).upper(); anchor = pd.Timestamp(row.date)
        if symbol not in taxonomy.index:
            continue
        annual_values, annual_padding, _ = _window(annual, symbol, anchor, annual_value_columns, 8)
        quarterly_values, quarterly_padding, _ = _window(quarterly, symbol, anchor, quarterly_value_columns, 20)
        daily_values, daily_padding, _ = _window(daily, symbol, anchor, daily_value_columns, DAILY_WINDOW)
        sparse_values, sparse_padding, sparse_labels, sparse_dates = _sparse_window(sparse, symbol, anchor, sparse_value_columns, 16)
        samples.append({
            "symbol": symbol, "year": int(anchor.year),
            "annual": annual_values, "annual_padding": annual_padding,
            "quarterly": quarterly_values, "quarterly_padding": quarterly_padding,
            "daily": daily_values, "daily_padding": daily_padding,
            "sparse": sparse_values, "sparse_padding": sparse_padding, "sparse_labels": sparse_labels,
            "sector": str(taxonomy.loc[symbol, "sector"]), "subsector": str(taxonomy.loc[symbol, "subsector"]),
            "industry": str(taxonomy.loc[symbol, "industry"]),
        })
    frame = pd.DataFrame([{key: value for key, value in sample.items() if isinstance(value, str) or isinstance(value, int)} for sample in samples])
    label_arrays: dict[str, np.ndarray] = {}
    label_names: dict[str, list[str]] = {}
    for task in ("sector", "subsector", "industry"):
        label_arrays[task], label_names[task], _ = _encode_labels(frame[task])
    years = sorted(frame["year"].unique())
    year_map = {year: i for i, year in enumerate(years)}
    label_arrays["year"] = frame["year"].map(year_map).to_numpy("int64"); label_names["year"] = [str(year) for year in years]
    label_arrays["temporal_year"] = label_arrays["year"]
    label_arrays["cross_sectional_year"] = label_arrays["year"]
    label_names["temporal_year"] = label_names["year"]
    label_names["cross_sectional_year"] = label_names["year"]

    device = torch.device(args.device)
    config = MultiRateTransformerConfig(backbone="encoder_decoder", d_model=128, num_heads=8, layers=2, document_pool="mean", max_position=512)
    tasks = tuple(MultiRateTaskSpec(name, "document", output_dim=len(label_names[name]), source="fused") for name in ("industry", "sector", "subsector", "temporal_year", "cross_sectional_year"))
    prediction_tasks = tuple(
        task
        for rate in ("annual", "quarterly", "daily", "sparse")
        for task in ((MultiRatePredictionTaskSpec(f"next_{rate}_subtoken", "next_token", "subtoken", output_dim=1, source=rate),)
                     + ((MultiRatePredictionTaskSpec("masked_daily_subtoken", "masked_token", "subtoken", output_dim=1, source="daily"),) if rate == "daily" else ()))
    )
    model = MultiRateTransformer(
        {"annual": len(annual_value_columns), "quarterly": len(quarterly_value_columns), "daily": len(daily_value_columns), "sparse": len(sparse_value_columns)},
        config=config,
        feature_families={
            "annual": {family: 1 for family in feature_families},
            "quarterly": {family: 1 for family in feature_families},
            "daily": {family: 1 for family in feature_families},
            "sparse": {family: len(raw_sparse_columns) for family in target_families},
        },
        tasks=tasks, prediction_tasks=prediction_tasks,
        family_classification_dim=len(feature_families) + len(target_families),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    losses: list[float] = []
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    model.train()
    for epoch in range(args.epochs):
        order = np.random.default_rng(epoch).permutation(len(samples)); epoch_loss = 0.0
        for start in range(0, len(order), args.batch_size):
            batch = [samples[index] for index in order[start:start + args.batch_size]]
            def stack(name: str) -> torch.Tensor: return torch.from_numpy(np.stack([item[name] for item in batch])).to(device)
            def context(name: str, padding_name: str) -> tuple[torch.Tensor, torch.Tensor]:
                values = stack(name); padding = stack(padding_name).bool()
                empty = padding.all(dim=1)
                if empty.any():
                    values[empty, -1] = 0.0; padding[empty, -1] = False
                leading = padding[:, 0]
                if leading.any():
                    values[leading, 0] = 0.0; padding[leading, 0] = False
                return values, padding
            daily_batch, daily_mask = context("daily", "daily_padding")
            daily_target_input = stack("daily")
            mask_rng = torch.Generator(device=device).manual_seed(epoch * 100000 + start)
            daily_mask_positions = torch.isfinite(daily_target_input) & (~daily_mask).unsqueeze(-1)
            daily_mask_positions &= torch.rand(daily_mask_positions.shape, device=device, generator=mask_rng).lt(0.15)
            daily_batch = daily_batch.clone()
            daily_batch[daily_mask_positions] = float("nan")
            annual_batch, annual_mask = context("annual", "annual_padding")
            quarterly_batch, quarterly_mask = context("quarterly", "quarterly_padding")
            sparse_batch, sparse_padding_mask = context("sparse", "sparse_padding")
            output = model(
                daily_batch, annual_batch, quarterly_batch, sparse_batch,
                daily_padding_mask=daily_mask, annual_padding_mask=annual_mask,
                quarterly_padding_mask=quarterly_mask, sparse_padding_mask=sparse_padding_mask,
                daily_dates=torch.arange(DAILY_WINDOW, device=device), annual_dates=torch.arange(8, device=device),
                quarterly_dates=torch.arange(20, device=device), sparse_dates=torch.arange(16, device=device),
            )
            document_losses = [nn.functional.cross_entropy(output["document_outputs"][name], torch.tensor(label_arrays[name][order[start:start + args.batch_size]], device=device)) for name in ("industry", "sector", "subsector", "temporal_year", "cross_sectional_year")]
            loss = sum(document_losses)
            for rate in ("annual", "quarterly", "daily", "sparse"):
                raw = stack(rate)
                if rate == "sparse":
                    family_count, width, offset = len(target_families), len(raw_sparse_columns), len(feature_families)
                    observed = torch.isfinite(raw).reshape(raw.shape[0], raw.shape[1], family_count, width).any(dim=-1).any(dim=1)
                else:
                    family_count, width, offset = len(feature_families), 1, 0
                    observed = torch.isfinite(raw).reshape(raw.shape[0], raw.shape[1], family_count, width).any(dim=-1).any(dim=1)
                labels = torch.arange(offset, offset + family_count, device=device).view(1, -1).expand(raw.shape[0], -1)
                logits = output["family_outputs"][rate]
                if observed.any():
                    loss = loss + nn.functional.cross_entropy(logits[observed], labels[observed])
            for rate in ("annual", "quarterly", "daily", "sparse"):
                raw_target = stack(rate)
                if rate == "sparse":
                    # Sparse family adapters consume eight raw fields per
                    # family but expose one learned family subtoken.
                    raw_shape = raw_target.shape
                    target = torch.nan_to_num(raw_target).reshape(*raw_shape[:2], len(target_families), len(raw_sparse_columns)).mean(dim=-1)
                    target_valid = torch.isfinite(raw_target).reshape(*raw_shape[:2], len(target_families), len(raw_sparse_columns)).any(dim=-1)
                else:
                    target = raw_target
                    target_valid = torch.isfinite(target)
                padding = stack(f"{rate}_padding").bool()
                next_prediction = output["prediction_outputs"][f"next_{rate}_subtoken"].squeeze(-1)
                valid_next = ~padding
                valid_next[:, :-1] &= valid_next[:, 1:]
                valid_next[:, -1] = False
                valid_next = valid_next.unsqueeze(-1).expand_as(target).clone()
                valid_next &= target_valid
                valid_next[:, :-1] &= target_valid[:, 1:]
                if valid_next.any():
                    loss = loss + nn.functional.mse_loss(next_prediction[valid_next], target.roll(-1, dims=1)[valid_next])
                if rate == "daily":
                    masked_prediction = output["prediction_outputs"]["masked_daily_subtoken"].squeeze(-1)
                    valid_masked = daily_mask_positions & target_valid
                    if valid_masked.any():
                        loss = loss + nn.functional.mse_loss(masked_prediction[valid_masked], target[valid_masked])
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            epoch_loss += float(loss.detach())
        epoch_mean = epoch_loss / max(1, (len(order) + args.batch_size - 1) // args.batch_size)
        losses.append(epoch_mean)
        if epoch_mean < best_loss - args.min_delta:
            best_loss = epoch_mean
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        print(f"epoch {epoch + 1}/{args.epochs} loss={epoch_mean:.6f}", flush=True)
        if stale_epochs >= max(1, args.patience):
            print(f"early stopping after epoch {epoch + 1}; best_loss={best_loss:.6f}", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval(); predictions: dict[str, list[np.ndarray]] = {name: [] for name in ("industry", "sector", "subsector", "temporal_year", "cross_sectional_year")}; states: list[np.ndarray] = []
    family_correct = 0
    family_total = 0
    with torch.inference_mode():
        for start in range(0, len(samples), args.batch_size):
            batch = samples[start:start + args.batch_size]
            def stack(name: str) -> torch.Tensor: return torch.from_numpy(np.stack([item[name] for item in batch])).to(device)
            def context(name: str, padding_name: str) -> tuple[torch.Tensor, torch.Tensor]:
                values = stack(name); padding = stack(padding_name).bool()
                empty = padding.all(dim=1)
                if empty.any():
                    values[empty, -1] = 0.0; padding[empty, -1] = False
                leading = padding[:, 0]
                if leading.any():
                    values[leading, 0] = 0.0; padding[leading, 0] = False
                return values, padding
            daily_batch, daily_mask = context("daily", "daily_padding")
            annual_batch, annual_mask = context("annual", "annual_padding")
            quarterly_batch, quarterly_mask = context("quarterly", "quarterly_padding")
            sparse_batch, sparse_padding_mask = context("sparse", "sparse_padding")
            output = model(daily_batch, annual_batch, quarterly_batch, sparse_batch, daily_padding_mask=daily_mask, annual_padding_mask=annual_mask, quarterly_padding_mask=quarterly_mask, sparse_padding_mask=sparse_padding_mask, daily_dates=torch.arange(DAILY_WINDOW, device=device), annual_dates=torch.arange(8, device=device), quarterly_dates=torch.arange(20, device=device), sparse_dates=torch.arange(16, device=device))
            states.append(output["document_state"].cpu().numpy())
            for rate in ("annual", "quarterly", "daily", "sparse"):
                raw = stack(rate)
                if rate == "sparse":
                    family_count, width, offset = len(target_families), len(raw_sparse_columns), len(feature_families)
                    observed = torch.isfinite(raw).reshape(raw.shape[0], raw.shape[1], family_count, width).any(dim=-1).any(dim=1)
                else:
                    family_count, width, offset = len(feature_families), 1, 0
                    observed = torch.isfinite(raw).reshape(raw.shape[0], raw.shape[1], family_count, width).any(dim=-1).any(dim=1)
                labels = torch.arange(offset, offset + family_count, device=device).view(1, -1).expand(raw.shape[0], -1)
                logits = output["family_outputs"][rate]
                family_correct += int((logits.argmax(dim=-1)[observed] == labels[observed]).sum())
                family_total += int(observed.sum())
            for name in predictions: predictions[name].append(output["document_outputs"][name].argmax(dim=-1).cpu().numpy())
    task_accuracy = {name: float((np.concatenate(predictions[name]) == label_arrays[name]).mean()) for name in predictions}
    task_accuracy["family_classification"] = family_correct / max(1, family_total)
    metrics = {"device": str(device), "samples": len(samples), "feature_families": feature_families, "target_families": target_families, "tasks": ["family_classification", "industry", "sector", "subsector", "temporal_year", "cross_sectional_year", "next_annual_subtoken", "next_quarterly_subtoken", "next_daily_subtoken", "next_sparse_subtoken", "masked_daily_subtoken"], "losses": losses, "best_loss": best_loss, "epochs_completed": len(losses), "patience": args.patience, "min_delta": args.min_delta, "task_accuracy": task_accuracy, "rates": ["annual", "quarterly", "daily", "sparse"], "backbone": config.backbone}
    (output_dir / "training_summary.json").write_text(json.dumps(metrics, indent=2))
    torch.save({"state_dict": model.state_dict(), "metrics": metrics, "labels": label_names}, output_dir / "multirate_mtl_model.pt")
    embeddings = np.concatenate(states)
    rows: list[dict[str, object]] = []
    for name in predictions:
        for label in label_names[name]:
            indices = np.where(np.asarray(label_names[name])[label_arrays[name]] == label)[0]
            if len(indices): rows.append({"task": name, "label": label, "support": int(len(indices)), "embedding": embeddings[indices].mean(axis=0)})
    coordinates = TSNE(n_components=3, perplexity=min(30.0, len(rows) - 1), init="pca", learning_rate="auto", max_iter=1500, random_state=42).fit_transform(np.stack([row.pop("embedding") for row in rows]).astype("float32"))
    plot = pd.DataFrame(rows); plot[["x", "y", "z"]] = coordinates; plot.to_csv(output_dir / "prototype_coordinates_3d.csv", index=False)


if __name__ == "__main__":
    main()
