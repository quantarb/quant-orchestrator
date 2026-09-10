"""Measure exact-input encoder reuse with correctness checks before timing."""

import copy
import json
from pathlib import Path
import resource
from time import perf_counter

import torch
from quant_orchestrator.platforms.ml_frameworks.torch.models.transformers.multirate import (
    MultiRateTransformer,
    MultiRateTransformerConfig,
    MultiRateTaskSpec,
)


def benchmark_reuse(output: Path, *, device="cuda", repeats=20):
    torch.manual_seed(0)
    dev = torch.device(device)
    net = MultiRateTransformer(
        {rate: 20 for rate in ("annual", "quarterly", "daily", "sparse")},
        config=MultiRateTransformerConfig(d_model=16, num_heads=2, layers=1, dropout=0.0),
        tasks=[MultiRateTaskSpec("trade", "token", source="daily")],
        modalities=("equity", "option", "note_bond", "preferred"),
    ).to(dev)
    lengths = {"annual": 8, "quarterly": 16, "daily": 64, "sparse": 16}
    data = {
        f"{rate}_values": torch.randn(8, length, 20, device=dev) for rate, length in lengths.items()
    }
    ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], device=dev)
    for rate in ("annual", "quarterly"):
        data[f"{rate}_values"] = data[f"{rate}_values"][[0, 0, 0, 0, 4, 4, 4, 4]].clone()
    grouped = {"annual": ids, "quarterly": ids}

    def sync():
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)

    def measure(fn):
        for _ in range(3):
            fn()
        sync()
        start = perf_counter()
        for _ in range(repeats):
            fn()
        sync()
        return (perf_counter() - start) / repeats

    net.train()
    ref = copy.deepcopy(net)
    ref(**data)["token_outputs"]["trade"].square().mean().backward()
    net(**data, rate_context_ids=grouped)["token_outputs"]["trade"].square().mean().backward()
    maximum_error = 0.0
    for (_, a), (_, b) in zip(net.named_parameters(), ref.named_parameters()):
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, rtol=2e-4, atol=2e-5)
            maximum_error = max(maximum_error, float((a.grad - b.grad).abs().max()))
    del ref

    def train_step(groups):
        net.zero_grad(set_to_none=True)
        net(**data, rate_context_ids=groups)["token_outputs"]["trade"].square().mean().backward()

    plain = measure(lambda: train_step(None))
    shared = measure(lambda: train_step(grouped))
    net.eval()
    with torch.no_grad():
        expected = net(**data)
        cache = net.rate_cache_from_output(expected)
        actual = net(**data, rate_cache=cache)
        torch.testing.assert_close(
            actual["token_outputs"]["trade"], expected["token_outputs"]["trade"]
        )
        uncached = measure(lambda: net(**data))
        cached = measure(lambda: net(**data, rate_cache=cache))
    result = dict(
        device=str(dev),
        batch_size=8,
        distinct_issuer_contexts=2,
        windows=lengths,
        repeats=repeats,
        training_plain_seconds=plain,
        training_shared_seconds=shared,
        training_speedup=plain / shared,
        inference_plain_seconds=uncached,
        inference_cached_seconds=cached,
        inference_speedup=uncached / cached,
        maximum_gradient_error=maximum_error,
        peak_process_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        peak_cuda_mib=torch.cuda.max_memory_allocated(dev) / 2**20 if dev.type == "cuda" else 0,
        scope="Fixed representative tensor shapes, identical inputs, no optimizer changes during inference reuse; not end-to-end throughput",
    )
    output.write_text(json.dumps(result, indent=2))
    return result
