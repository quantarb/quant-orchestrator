from __future__ import annotations

from dataclasses import asdict, dataclass

from quant_orchestrator.platforms.ml_frameworks.torch.runtime import configure_torch_runtime


@dataclass(frozen=True)
class FlairRuntimeInfo:
    torch_device: str
    flair_device: str
    cuda_available: bool
    cuda_device_name: str | None


def configure_flair_runtime(
    *,
    require_cuda: bool = False,
    device: str | None = None,
    allow_tf32: bool = True,
    matmul_precision: str = "high",
) -> FlairRuntimeInfo:
    """Configure Torch and Flair for a notebook or training job."""

    import torch
    import flair

    torch_info = configure_torch_runtime(
        require_cuda=require_cuda,
        device=device,
        allow_tf32=allow_tf32,
        matmul_precision=matmul_precision,
    )
    flair.device = torch.device(torch_info.torch_device)
    return FlairRuntimeInfo(flair_device=str(flair.device), **asdict(torch_info))


configure_flair_device = configure_flair_runtime

