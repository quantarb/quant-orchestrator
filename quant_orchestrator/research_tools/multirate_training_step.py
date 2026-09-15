"""Shared multi-rate training step for warehouse streams and recorded evaluation."""
import torch
from torch import nn
from time import perf_counter
from .multirate_batch import BatchTensors, supervision_counts
from .multirate_objectives import reconstruction_mask, reconstruction_targets, family_channels
from quant_orchestrator.platforms.ml_frameworks.torch.models.transformers.multirate.temporal_tasks import (
    DOCUMENT_TASK_NAMES, SUPERVISED_TARGET_TASK_NAMES, ORACLE_SUPERVISED_TASK_NAMES,
    FUND_ACTIVITY_SUPERVISED_TASK_NAMES, HOLDER_ACTIVITY_SUPERVISED_TASK_NAMES, TRADE_EVENT_SUPERVISED_TASK_NAMES,
)


def issuer_stream_inputs(batch, stack, *, issuer_context, device):
    if issuer_context == "none":
        return {}
    keys = {}
    ids = []
    for item in batch:
        key = item["issuer_context_key"]
        keys.setdefault(key, len(keys))
        ids.append(keys[key])
    payloads = {}
    for rate in ("daily", "sparse"):
        values = stack(f"issuer_{rate}").clone()
        padding = stack(f"issuer_{rate}_padding").bool().clone()
        # Auxiliary prediction heads use only their local rate states,
        # never this supervised fusion path.
        leading = padding[:, 0]
        values[leading, 0] = 0.
        padding[leading, 0] = False
        payloads[rate] = {"values": values, "padding": padding,
            "dates": stack(f"issuer_{rate}_timestamps"),
            "context_ids": torch.tensor(ids, device=device)}
    return payloads


def make_training_step(*, args, trainer, device, annual_state, reconstruction_widths, feature_family_dimensions, sparse_input_families, raw_sparse_columns, family_names, asset_class_ids, enabled_document_tasks, prediction_names, expected_task_names, mrl_dimensions, task_observations, task_family_observations, task_loss_sums, epoch_clocks, alignment_loss):
    def training_step(module: torch.nn.Module, batch: list[dict[str, object]], active_tasks):
        memory_input = None
        if args.sequence_mode == 'annual_memory':
            annual_state.begin_epoch(trainer.current_epoch)
            memory_input = annual_state.inputs(batch, module)
        if module.training:
            epoch_clocks.setdefault(trainer.current_epoch,perf_counter())
        stack = BatchTensors(batch, device)

        def context(name: str, padding_name: str) -> tuple[torch.Tensor, torch.Tensor]:
            values = stack(name).clone(); padding = stack(padding_name).bool().clone()
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
        family_masked_positions: dict[str, torch.Tensor] = {}
        token_batches = {}
        masked_batches = {"annual": (annual_batch, annual_mask), "quarterly": (quarterly_batch, quarterly_mask), "daily": (None, None), "sparse": (sparse_batch, sparse_padding_mask)}
        mask_rng = torch.Generator(device=device).manual_seed(trainer.current_epoch * 100000 + trainer.current_step)
        for rate in ("annual", "quarterly", "daily", "sparse"):
            if rate == "daily":
                batch_values, padding = context("daily", "daily_padding")
            else:
                batch_values, padding = masked_batches[rate]
            raw = stack(rate)
            selected, family_selected = reconstruction_mask(
                raw, stack(f"{rate}_padding").bool(), widths=reconstruction_widths[rate], generator=mask_rng,
                family_probability=0.15 if args.self_supervision in ("both", "masked") else 0.,
                feature_probability=0.15 if args.self_supervision in ("both", "masked") else 0.,
            )
            token_values = batch_values.clone()
            token_values[family_selected] = float("nan")
            batch_values = batch_values.clone()
            batch_values[selected] = float("nan")
            if rate in {"annual", "quarterly"}:
                representatives = {}
                first = []
                for index, item in enumerate(batch):
                    key = item[f"{rate}_context_key"]
                    representatives.setdefault(key, index)
                    first.append(representatives[key])
                first = torch.tensor(first, device=device)
                batch_values = batch_values.index_select(0, first)
                selected = selected.index_select(0, first)
                family_selected = family_selected.index_select(0, first)
                token_values = token_values.index_select(0, first)
            if module.training:
                task_observations[f"masked_{rate}_whole_family_values"] += int(family_selected.sum())
                task_observations[f"masked_{rate}_individual_values"] += int(selected.sum())
            masked_batches[rate] = (batch_values, padding); masked_positions[rate] = selected
            family_masked_positions[rate] = family_selected
            token_batches[rate] = token_values
        daily_batch, daily_mask = masked_batches["daily"]
        annual_batch, annual_mask = masked_batches["annual"]
        quarterly_batch, quarterly_mask = masked_batches["quarterly"]
        sparse_batch, sparse_padding_mask = masked_batches["sparse"]
        # IDs are deliberately rate-specific. Annual/quarterly issuer streams
        # can be shared by instruments with the same issuer/as-of date; sparse
        # streams retain symbol identity because option-native events differ.
        annual_ids = {key: index for index, key in enumerate(sorted({item["annual_context_key"] for item in batch}))}
        quarterly_ids = {key: index for index, key in enumerate(sorted({item["quarterly_context_key"] for item in batch}))}
        sparse_ids = {key: index for index, key in enumerate(sorted({(str(item["symbol"]), str(item["date"])) for item in batch}))}
        annual_context_ids = torch.tensor([annual_ids[item["annual_context_key"]] for item in batch], device=device)
        quarterly_context_ids = torch.tensor([quarterly_ids[item["quarterly_context_key"]] for item in batch], device=device)
        sparse_context_ids = torch.tensor([sparse_ids[(str(item["symbol"]), str(item["date"]))] for item in batch], device=device)
        rate_context_ids = {
            "annual": annual_context_ids if torch.unique(annual_context_ids).numel() < len(batch) else None,
            "quarterly": quarterly_context_ids if torch.unique(quarterly_context_ids).numel() < len(batch) else None,
            "sparse": sparse_context_ids if torch.unique(sparse_context_ids).numel() < len(batch) else None,
        }
        if memory_input is not None and not module.config.share_recurrent_issuer_context:
            rate_context_ids = {}
        issuer_ids = {issuer: i for i, issuer in enumerate(dict.fromkeys(item['issuer'] for item in batch))} if module.config.cross_instrument_attention else {}
        instrument_groups = torch.tensor([issuer_ids[item['issuer']] for item in batch], device=device) if issuer_ids else None
        output = module(
            daily_batch, annual_batch, quarterly_batch, sparse_batch,
            daily_padding_mask=daily_mask, annual_padding_mask=annual_mask,
            quarterly_padding_mask=quarterly_mask, sparse_padding_mask=sparse_padding_mask,
            daily_dates=stack("daily_timestamps"), annual_dates=stack("annual_timestamps"),
            quarterly_dates=stack("quarterly_timestamps"), sparse_dates=stack("sparse_timestamps"),
            daily_modality_ids=torch.tensor([asset_class_ids[item["asset_class"]] for item in batch], device=device)[:, None].expand(-1, daily_batch.shape[1]),
            rate_context_ids=rate_context_ids, instrument_group_ids=instrument_groups,
            issuer_streams=issuer_stream_inputs(batch, stack, issuer_context=args.issuer_context, device=device),
            **{f"{rate}_family_presence": family_channels(
                torch.isfinite(stack(rate)) & ~stack(f"{rate}_padding").bool().unsqueeze(-1),
                reconstruction_widths[rate],
            ).any(-1) for rate in reconstruction_widths},
            compute_document_outputs=not args.disable_document_tasks, annual_memory=memory_input,
        )
        token_predictions = {}
        if args.self_supervision in ("both", "masked"):
            # Token MTP gets a separate whole-family view, with no feature-level
            # corruption borrowed from the subtoken MTP pass.
            token_predictions = {name: prediction for name, prediction in module(
                token_batches["daily"], token_batches["annual"], token_batches["quarterly"], token_batches["sparse"],
                daily_padding_mask=daily_mask, annual_padding_mask=annual_mask,
                quarterly_padding_mask=quarterly_mask, sparse_padding_mask=sparse_padding_mask,
                daily_dates=stack("daily_timestamps"), annual_dates=stack("annual_timestamps"),
                quarterly_dates=stack("quarterly_timestamps"), sparse_dates=stack("sparse_timestamps"),
                daily_modality_ids=torch.tensor([asset_class_ids[item["asset_class"]] for item in batch], device=device)[:, None].expand(-1, daily_batch.shape[1]),
                rate_context_ids=rate_context_ids, instrument_group_ids=instrument_groups,
                **{f"{rate}_family_presence": family_channels(
                    torch.isfinite(stack(rate)) & ~stack(f"{rate}_padding").bool().unsqueeze(-1),
                    reconstruction_widths[rate],
                ).any(-1) for rate in reconstruction_widths},
                compute_document_outputs=False, annual_memory=memory_input,
            )["prediction_outputs"].items() if name.startswith("masked_") and name.endswith("_token")}
        if tuple(output["document_outputs"]) + tuple(output["token_outputs"]) + tuple(output["prediction_outputs"]) != expected_task_names:
            raise RuntimeError("model task outputs do not match the temporal token+subtoken MTL contract")
        zero_source = next(iter(output["token_outputs"].values()), None)
        if zero_source is None:
            zero_source = next(iter(output["prediction_outputs"].values()))
        zero = zero_source.sum() * 0.0
        active_names = {task.name for task in active_tasks}
        task_losses = {task.name: zero for task in active_tasks}
        if "mrl" in active_names:
            task_losses["mrl"] = alignment_loss(output["document_state"], mrl_dimensions)
        for name in DOCUMENT_TASK_NAMES[1:]:
            if name not in enabled_document_tasks:
                continue
            if name in active_names:
                target = torch.tensor([item[f"{name}_label"] for item in batch], device=device)
                task_losses[name] = nn.functional.cross_entropy(output["document_outputs"][name], target)
        supervised_targets = stack("supervised_targets")
        supervised_valid = stack("supervised_valid").bool()
        for task_index, name in enumerate(SUPERVISED_TARGET_TASK_NAMES):
            if name not in active_names:
                continue
            valid = supervised_valid[:, :, task_index] & ~daily_mask
            if args.sequence_mode not in ('documents', 'annual_memory') and not args.training_sequence_stride:
                valid[:, :-1] = False
            if not valid.any():
                continue
            if module.training:
                total, by_asset = supervision_counts(valid, batch)
                task_observations[name] += total
                for asset, count in by_asset.items():
                    task_observations[f"{asset}:{name}"] += count
            target = supervised_targets[:, :, task_index]
            prediction = output["token_outputs"][name].squeeze(-1)
            if name in ORACLE_SUPERVISED_TASK_NAMES or name in FUND_ACTIVITY_SUPERVISED_TASK_NAMES or name in HOLDER_ACTIVITY_SUPERVISED_TASK_NAMES or name in TRADE_EVENT_SUPERVISED_TASK_NAMES:
                task_losses[name] = nn.functional.binary_cross_entropy_with_logits(prediction[valid], target[valid])
            else:
                task_losses[name] = nn.functional.smooth_l1_loss(prediction[valid], target[valid])
        family_labels = torch.arange(len(family_names), device=device).view(1, -1).expand(len(batch), -1)
        family_valid = torch.zeros((len(batch), len(family_names)), dtype=torch.bool, device=device)
        for rate in ("annual", "quarterly", "daily", "sparse"):
            raw = stack(rate)
            if rate == "sparse":
                local_count, width, family_offset = len(sparse_input_families), len(raw_sparse_columns), len(feature_family_dimensions)
                observed = torch.isfinite(raw).reshape(raw.shape[0], raw.shape[1], local_count, width).any(dim=-1).any(dim=1)
            else:
                family_offset = 0
                offset = 0
                family_observed = []
                for family in feature_family_dimensions:
                    width = feature_family_dimensions[family]
                    family_observed.append(torch.isfinite(raw[:, :, offset:offset + width]).any(dim=-1).any(dim=1))
                    offset += width
                observed = torch.stack(family_observed, dim=1)
                local_count = len(feature_family_dimensions)
            family_valid[:, family_offset:family_offset + local_count] |= observed
        if "family" in active_names and family_valid.any():
            task_losses["family"] = nn.functional.cross_entropy(output["document_outputs"]["family"][family_valid], family_labels[family_valid])
        for rate in reconstruction_widths if prediction_names else ():
            targets = reconstruction_targets(
                stack(rate), stack(f"{rate}_padding").bool(), stack(f"{rate}_timestamps"),
                masked_positions[rate], reconstruction_widths[rate], family_selected=family_masked_positions[rate],
            )
            for objective, (target, valid) in targets.items():
                kind, level = objective.split("_")
                name = f"{kind}_{rate}_{level}"
                prediction = token_predictions[name] if objective == "masked_token" and name in token_predictions else output["prediction_outputs"][name]
                if prediction.shape != target.shape:
                    raise ValueError(f"Reconstruction shape mismatch for {name}: {prediction.shape} != {target.shape}")
                if name in active_names and valid.any():
                    task_losses[name] = nn.functional.mse_loss(prediction[valid], target[valid])
                if module.training and name in active_names:
                    task_observations[name] += int(valid.sum())
                    per_family = valid if level == "subtoken" else family_channels(valid, reconstruction_widths[rate])
                    counts = per_family.sum(dim=(0, 1, 3)).detach().cpu().tolist()
                    names = sparse_input_families if rate == "sparse" else list(feature_family_dimensions)
                    for family, count in zip(names, counts):
                        task_family_observations[f"{name}:{family}"] += count
        if module.training:
            for name, loss in task_losses.items():
                task_loss_sums[name] += float(loss.detach())
            for rate in ("annual", "quarterly", "daily", "sparse"):
                task_observations[f"masked_{rate}"] += int(masked_positions[rate].sum())
        if memory_input is not None:
            annual_state.update(batch, output)
        for item in batch:
            if hasattr(item, "release"):
                item.release()
        return task_losses

    return training_step
