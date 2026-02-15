"""Shared PyTorch/CUDA configuration for STT and TTS workers.

These utilities must be called at specific points in the worker lifecycle:
- configure_cuda_env(): BEFORE any ``import torch`` (module-level)
- configure_torch_inference(): AFTER torch is available (inside serve())
"""

from __future__ import annotations

import os

from macaw.logging import get_logger

logger = get_logger("worker.torch_utils")


def configure_cuda_env() -> None:
    """Set PYTORCH_CUDA_ALLOC_CONF before torch is imported.

    Enables ``expandable_segments`` which saves 10x+ GPU memory with small
    streaming chunks by using dynamically-sized segments instead of fixed
    allocations.

    Reference: NeMo speech_to_text_streaming_infer_rnnt.py

    IMPORTANT: Must be called before ``import torch``.
    """
    alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    if "expandable_segments" not in alloc_conf:
        new_val = (
            f"{alloc_conf},expandable_segments:True" if alloc_conf else "expandable_segments:True"
        )
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = new_val


def configure_torch_inference() -> None:
    """Configure PyTorch for inference-only operation.

    Sets process-wide defaults that eliminate autograd overhead:
    - set_grad_enabled(False): prevents accidental gradient computation
    - set_float32_matmul_precision("high"): enables TF32 on Ampere+ GPUs

    Safe to call even if torch is not installed (e.g., CTranslate2-only engines).
    """
    try:
        import torch

        torch.set_grad_enabled(False)
        torch.set_float32_matmul_precision("high")
        logger.info(
            "torch_inference_configured",
            grad_enabled=False,
            float32_matmul_precision="high",
        )
    except ImportError:
        pass


def resolve_device(device_str: str) -> str:
    """Resolve device string, probing CUDA availability for "auto".

    Args:
        device_str: One of "auto", "cpu", "cuda", or "cuda:N".

    Returns:
        Resolved device string. "auto" becomes "cuda:0" if CUDA is
        available, otherwise "cpu". All other values pass through.
    """
    if device_str == "auto":
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda:0"
        except ImportError:
            pass
        return "cpu"
    return device_str


def release_gpu_memory() -> None:
    """Best-effort GPU memory cleanup via torch.cuda.empty_cache().

    Safe to call even when torch is not installed or CUDA is unavailable.
    """
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
