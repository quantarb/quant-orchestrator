"""Train the four-rate MultiRateTransformer MTL corpus."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

# Make direct execution use the checkout being trained, even when an older
# globally installed quant-orchestrator is present.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
    parser.add_argument("--grad-accumulation-steps", type=int, default=1)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--min-delta", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.grad_accumulation_steps < 1:
        parser.error("--grad-accumulation-steps must be at least 1")
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
    family_names = [*feature_families, *target_families]
    label_names["family"] = family_names

    device = torch.device(args.device)
    config = MultiRateTransformerConfig(
        backbone="encoder_decoder", d_model=args.d_model, num_heads=args.num_heads,
        layers=args.layers, document_pool="mean", max_position=512,
    )
    tasks = (
        MultiRateTaskSpec("family", "document", output_dim=len(family_names), source="family"),
        *(MultiRateTaskSpec(name, "document", output_dim=len(label_names[name]), source="fused")
          for name in ("industry", "sector", "subsector", "year")),
    )
    prediction_tasks = tuple(
        task
        for rate in ("annual", "quarterly", "daily", "sparse")
        for task in (
            MultiRatePredictionTaskSpec(f"next_{rate}_subtoken", "next_token", "subtoken", output_dim=1, source=rate),
            MultiRatePredictionTaskSpec(f"masked_{rate}_subtoken", "masked_token", "subtoken", output_dim=1, source=rate),
        )
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
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    losses: list[float] = []
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    model.train()
    for epoch in range(args.epochs):
        order = np.random.default_rng(epoch).permutation(len(samples)); epoch_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
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
            annual_batch, annual_mask = context("annual", "annual_padding")
            quarterly_batch, quarterly_mask = context("quarterly", "quarterly_padding")
            sparse_batch, sparse_padding_mask = context("sparse", "sparse_padding")
            masked_positions: dict[str, torch.Tensor] = {}
            masked_batches = {
                "annual": (annual_batch, annual_mask),
                "quarterly": (quarterly_batch, quarterly_mask),
                "daily": (None, None),
                "sparse": (sparse_batch, sparse_padding_mask),
            }
            mask_rng = torch.Generator(device=device).manual_seed(epoch * 100000 + start)
            for rate in ("annual", "quarterly", "daily", "sparse"):
                if rate == "daily":
                    batch_values, padding = context("daily", "daily_padding")
                else:
                    batch_values, padding = masked_batches[rate]
                raw = stack(rate)
                if rate == "sparse":
                    raw_shape = raw.shape
                    target_valid = torch.isfinite(raw).reshape(
                        *raw_shape[:2], len(target_families), len(raw_sparse_columns)
                    ).any(dim=-1)
                    mask_shape = target_valid.shape
                    selected = target_valid & ~padding.unsqueeze(-1)
                    selected &= torch.rand(mask_shape, device=device, generator=mask_rng).lt(0.15)
                    expanded = selected.unsqueeze(-1).expand(*raw_shape[:2], len(target_families), len(raw_sparse_columns)).reshape(raw_shape)
                    batch_values = batch_values.clone()
                    batch_values[expanded] = float("nan")
                else:
                    target_valid = torch.isfinite(raw)
                    selected = target_valid & ~padding.unsqueeze(-1)
                    selected &= torch.rand(selected.shape, device=device, generator=mask_rng).lt(0.15)
                    batch_values = batch_values.clone()
                    batch_values[selected] = float("nan")
                masked_batches[rate] = (batch_values, padding)
                masked_positions[rate] = selected
            annual_batch, annual_mask = masked_batches["annual"]
            quarterly_batch, quarterly_mask = masked_batches["quarterly"]
            daily_batch, daily_mask = masked_batches["daily"]
            sparse_batch, sparse_padding_mask = masked_batches["sparse"]
            output = model(
                daily_batch, annual_batch, quarterly_batch, sparse_batch,
                daily_padding_mask=daily_mask, annual_padding_mask=annual_mask,
                quarterly_padding_mask=quarterly_mask, sparse_padding_mask=sparse_padding_mask,
                daily_dates=torch.arange(DAILY_WINDOW, device=device), annual_dates=torch.arange(8, device=device),
                quarterly_dates=torch.arange(20, device=device), sparse_dates=torch.arange(16, device=device),
            )
            document_losses = [nn.functional.cross_entropy(output["document_outputs"][name], torch.tensor(label_arrays[name][order[start:start + args.batch_size]], device=device)) for name in ("industry", "sector", "subsector", "year")]
            loss = sum(document_losses)
            family_labels = torch.arange(len(family_names), device=device).view(1, -1).expand(len(batch), -1)
            family_valid = torch.zeros((len(batch), len(family_names)), dtype=torch.bool, device=device)
            for rate in ("annual", "quarterly", "daily", "sparse"):
                raw = stack(rate)
                if rate == "sparse":
                    local_count, width, offset = len(target_families), len(raw_sparse_columns), len(feature_families)
                    observed = torch.isfinite(raw).reshape(raw.shape[0], raw.shape[1], local_count, width).any(dim=-1).any(dim=1)
                else:
                    local_count, width, offset = len(feature_families), 1, 0
                    observed = torch.isfinite(raw).reshape(raw.shape[0], raw.shape[1], local_count, width).any(dim=-1).any(dim=1)
                family_valid[:, offset:offset + local_count] |= observed
            family_logits = output["document_outputs"]["family"]
            if family_valid.any():
                loss = loss + nn.functional.cross_entropy(family_logits[family_valid], family_labels[family_valid])
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
                valid_next[:, :-1] = valid_next[:, :-1] & valid_next[:, 1:]
                valid_next[:, -1] = False
                valid_next = valid_next.unsqueeze(-1).expand_as(target).clone()
                valid_next &= target_valid
                valid_next[:, :-1] = valid_next[:, :-1] & target_valid[:, 1:]
                if valid_next.any():
                    loss = loss + nn.functional.mse_loss(next_prediction[valid_next], target.roll(-1, dims=1)[valid_next])
                masked_prediction = output["prediction_outputs"][f"masked_{rate}_subtoken"].squeeze(-1)
                masked_valid = masked_positions[rate]
                if rate == "sparse":
                    masked_target = target
                else:
                    masked_target = target
                if masked_valid.any():
                    loss = loss + nn.functional.mse_loss(masked_prediction[masked_valid], masked_target[masked_valid])
            if not torch.isfinite(loss):
                diagnostics = {
                    "document": {name: bool(torch.isfinite(value).all().item()) for name, value in output["document_outputs"].items()},
                    "prediction": {name: bool(torch.isfinite(value).all().item()) for name, value in output["prediction_outputs"].items()},
                    "family_targets": bool(torch.isfinite(family_labels[family_valid]).all().item()) if family_valid.any() else True,
                }
                raise RuntimeError(f"non-finite MTL loss: {diagnostics}")
            epoch_loss += float(loss.detach())
            (loss / args.grad_accumulation_steps).backward()
            micro_step = start // args.batch_size + 1
            is_last_batch = start + args.batch_size >= len(order)
            if micro_step % args.grad_accumulation_steps == 0 or is_last_batch:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
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

    model.eval(); predictions: dict[str, list[np.ndarray]] = {name: [] for name in ("industry", "sector", "subsector", "year")}; states: list[np.ndarray] = []; family_states: list[np.ndarray] = []; family_valid_rows: list[np.ndarray] = []
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
            family_labels = torch.arange(len(family_names), device=device).view(1, -1).expand(len(batch), -1)
            family_valid = torch.zeros((len(batch), len(family_names)), dtype=torch.bool, device=device)
            for rate in ("annual", "quarterly", "daily", "sparse"):
                raw = stack(rate)
                if rate == "sparse":
                    local_count, width, offset = len(target_families), len(raw_sparse_columns), len(feature_families)
                    observed = torch.isfinite(raw).reshape(raw.shape[0], raw.shape[1], local_count, width).any(dim=-1).any(dim=1)
                else:
                    local_count, width, offset = len(feature_families), 1, 0
                    observed = torch.isfinite(raw).reshape(raw.shape[0], raw.shape[1], local_count, width).any(dim=-1).any(dim=1)
                family_valid[:, offset:offset + local_count] |= observed
            family_states.append(output["family_document_state"].cpu().numpy())
            family_valid_rows.append(family_valid.cpu().numpy())
            family_predictions = output["document_outputs"]["family"].argmax(dim=-1)
            family_correct += int((family_predictions[family_valid] == family_labels[family_valid]).sum())
            family_total += int(family_valid.sum())
            for name in predictions: predictions[name].append(output["document_outputs"][name].argmax(dim=-1).cpu().numpy())
    task_accuracy = {name: float((np.concatenate(predictions[name]) == label_arrays[name]).mean()) for name in predictions}
    task_accuracy["family"] = family_correct / max(1, family_total)
    metrics = {"device": str(device), "samples": len(samples), "feature_families": feature_families, "target_families": target_families, "family_labels": family_names, "tasks": ["family", "industry", "sector", "subsector", "year", "next_annual_subtoken", "next_quarterly_subtoken", "next_daily_subtoken", "next_sparse_subtoken", "masked_annual_subtoken", "masked_quarterly_subtoken", "masked_daily_subtoken", "masked_sparse_subtoken"], "losses": losses, "best_loss": best_loss, "epochs_completed": len(losses), "patience": args.patience, "min_delta": args.min_delta, "task_accuracy": task_accuracy, "rates": ["annual", "quarterly", "daily", "sparse"], "backbone": config.backbone}
    (output_dir / "training_summary.json").write_text(json.dumps(metrics, indent=2))
    torch.save({"state_dict": model.state_dict(), "metrics": metrics, "labels": label_names}, output_dir / "multirate_mtl_model.pt")
    embeddings = np.nan_to_num(np.concatenate(states), nan=0.0, posinf=0.0, neginf=0.0)
    family_embeddings = np.nan_to_num(np.concatenate(family_states), nan=0.0, posinf=0.0, neginf=0.0)
    family_valid_array = np.concatenate(family_valid_rows)
    rows: list[dict[str, object]] = []
    for family_index, label in enumerate(family_names):
        indices = np.where(family_valid_array[:, family_index])[0]
        if len(indices):
            rows.append({"task": "family", "label": label, "support": int(len(indices)), "embedding": family_embeddings[indices, family_index].mean(axis=0)})
    for name in predictions:
        for label in label_names[name]:
            indices = np.where(np.asarray(label_names[name])[label_arrays[name]] == label)[0]
            if len(indices): rows.append({"task": name, "label": label, "support": int(len(indices)), "embedding": embeddings[indices].mean(axis=0)})
    if len(rows) >= 2:
        coordinates = TSNE(n_components=3, perplexity=min(30.0, len(rows) - 1), init="pca", learning_rate="auto", max_iter=1500, random_state=42).fit_transform(np.stack([row.pop("embedding") for row in rows]).astype("float32"))
        plot = pd.DataFrame(rows); plot[["x", "y", "z"]] = coordinates; plot.to_csv(output_dir / "prototype_coordinates_3d.csv", index=False)


if __name__ == "__main__":
    main()
