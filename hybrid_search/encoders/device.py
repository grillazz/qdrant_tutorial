"""Device and dtype selection that works on CUDA, Apple Silicon and plain CPU."""

from __future__ import annotations

import torch

__all__ = ["resolve_device", "resolve_dtype", "describe_runtime"]


def resolve_device(preferred: str | None = None) -> torch.device:
    """Pick a torch device.

    Order: explicit argument, CUDA, Apple MPS, CPU.
    """
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(device: torch.device, half: bool | None = None) -> torch.dtype:
    """Pick a dtype for the given device.

    ``half=None`` means auto. Half precision is only used on CUDA: on MPS it is
    unreliable for the wide vocabulary tensors SPLADE produces, and on CPU it is
    slower than float32.
    """
    if half is False or device.type != "cuda":
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def describe_runtime(device: torch.device, dtype: torch.dtype) -> str:
    """One-line summary for logging."""
    name = device.type
    if device.type == "cuda":
        name = torch.cuda.get_device_name(device)
    return f"{name} / {str(dtype).removeprefix('torch.')}"
