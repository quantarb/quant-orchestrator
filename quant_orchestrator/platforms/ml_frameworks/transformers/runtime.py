from __future__ import annotations

from dataclasses import asdict, dataclass

from quant_orchestrator.platforms.ml_frameworks.torch.runtime import configure_torch_runtime


@dataclass(frozen=True)
class TransformersRuntimeInfo:
    torch_device: str
    cuda_available: bool
    cuda_device_name: str | None


def configure_transformers_runtime(
    *,
    require_cuda: bool = False,
    device: str | None = None,
    allow_tf32: bool = True,
    matmul_precision: str = "high",
) -> TransformersRuntimeInfo:
    """Configure Torch runtime for Hugging Face Transformers workflows."""

    return TransformersRuntimeInfo(
        **asdict(
            configure_torch_runtime(
                require_cuda=require_cuda,
                device=device,
                allow_tf32=allow_tf32,
                matmul_precision=matmul_precision,
            )
        )
    )

