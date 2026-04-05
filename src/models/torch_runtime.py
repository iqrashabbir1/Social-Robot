from __future__ import annotations

from dataclasses import dataclass
from typing import Any


try:  # pragma: no cover - optional runtime dependency
    import torch
    import torch.nn.functional as torch_f
except ImportError:  # pragma: no cover - optional runtime dependency
    torch = None
    torch_f = None


@dataclass(frozen=True)
class TorchRuntime:
    requested_backend: str
    active_backend: str
    device: str
    reason: str


def resolve_torch_runtime(requested_backend: str, requested_device: str = "auto") -> TorchRuntime:
    normalized_backend = str(requested_backend or "cpu").strip().lower()
    normalized_device = str(requested_device or "auto").strip().lower()

    if normalized_backend == "cpu":
        return TorchRuntime("cpu", "cpu", "cpu", "CPU backend explicitly requested.")

    if torch is None:
        if normalized_backend == "gpu":
            raise RuntimeError("GPU backend was requested, but PyTorch is not installed.")
        return TorchRuntime(normalized_backend, "cpu", "cpu", "PyTorch is not installed; falling back to CPU.")

    cuda_available = bool(torch.cuda.is_available())
    if normalized_device == "auto":
        target_device = "cuda" if cuda_available else "cpu"
    else:
        target_device = normalized_device

    if normalized_backend == "gpu":
        if not cuda_available:
            raise RuntimeError("GPU backend was requested, but torch.cuda.is_available() is false.")
        if not target_device.startswith("cuda"):
            raise RuntimeError(f"GPU backend requires a CUDA device, got '{target_device}'.")
        return TorchRuntime("gpu", "gpu", target_device, "CUDA backend enabled.")

    if normalized_backend == "auto":
        if cuda_available and target_device.startswith("cuda"):
            return TorchRuntime("auto", "gpu", target_device, "CUDA detected; using GPU backend.")
        return TorchRuntime("auto", "cpu", "cpu", "CUDA unavailable; using CPU backend.")

    raise ValueError(f"Unsupported runtime backend '{requested_backend}'.")


def require_torch() -> Any:
    if torch is None or torch_f is None:
        raise RuntimeError("PyTorch is required for the GPU-enabled training path.")
    return torch


def encode_labels(labels: list[str], y_values: list[str]) -> list[int]:
    mapping = {label: index for index, label in enumerate(labels)}
    return [mapping[value] for value in y_values]


def synchronize_if_needed(device: str) -> None:
    if torch is not None and str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
