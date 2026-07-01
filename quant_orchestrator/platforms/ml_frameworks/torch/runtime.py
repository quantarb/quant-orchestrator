from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TorchRuntimeInfo:
    torch_device: str
    cuda_available: bool
    cuda_device_name: str | None


def configure_torch_runtime(
    *,
    require_cuda: bool = False,
    device: str | None = None,
    allow_tf32: bool = True,
    matmul_precision: str = "high",
) -> TorchRuntimeInfo:
    """Configure Torch device and numeric runtime knobs for ML framework adapters."""

    import torch

    if device is None:
        if require_cuda and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required, but torch.cuda.is_available() is false.")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    torch_device = torch.device(device)
    if require_cuda and torch_device.type != "cuda":
        raise RuntimeError(f"CUDA is required, got device={torch_device}.")

    if torch_device.type == "cuda":
        torch.cuda.set_device(torch_device)
        torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
        torch.set_float32_matmul_precision(matmul_precision)

    return TorchRuntimeInfo(
        torch_device=str(torch_device),
        cuda_available=bool(torch.cuda.is_available()),
        cuda_device_name=torch.cuda.get_device_name(torch_device) if torch_device.type == "cuda" else None,
    )

