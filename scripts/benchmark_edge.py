from __future__ import annotations

import argparse
import platform
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.io_utils import write_dataframe, write_json
from src.models.torch_runtime import require_torch, synchronize_if_needed
from src.models.vision.image_emotion_classifier import SimpleEmotionCNN, load_model_checkpoint


torch = require_torch()

PLATFORM_CATALOG: dict[str, dict[str, Any]] = {
    "raspberry_pi_4": {
        "platform_label": "Raspberry Pi 4 (4GB)",
        "recommended_device": "cpu",
        "power_estimate_w": 3.8,
        "privacy_penalty": 0.05,
    },
    "jetson_orin": {
        "platform_label": "NVIDIA Jetson Orin",
        "recommended_device": "cuda",
        "power_estimate_w": 7.2,
        "privacy_penalty": 0.08,
    },
    "apple_m2": {
        "platform_label": "Apple M2",
        "recommended_device": "mps",
        "power_estimate_w": 4.5,
        "privacy_penalty": 0.06,
    },
    "cloud_t4": {
        "platform_label": "Cloud T4 GPU",
        "recommended_device": "cuda",
        "power_estimate_w": 70.0,
        "privacy_penalty": 0.28,
    },
}


def _resolve_device(device: str) -> str:
    normalized = str(device).strip().lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if normalized == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for benchmarking, but torch.cuda.is_available() is false.")
    if normalized == "mps":
        if not bool(getattr(torch.backends, "mps", None)) or not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested for benchmarking, but torch.backends.mps.is_available() is false.")
    return normalized


def _load_benchmark_model(checkpoint_path: Path | None, device: str, image_size: int = 128) -> tuple[torch.nn.Module, str]:
    if checkpoint_path and checkpoint_path.exists():
        model_bundle, _, _, _ = load_model_checkpoint(checkpoint_path, device=device)
        model = model_bundle.network.to(device)
        return model.eval(), f"checkpoint:{checkpoint_path.name}"

    model = SimpleEmotionCNN(num_classes=4, image_size=image_size).network.to(device)
    return model.eval(), "cnn_small_fresh_init"


def _power_estimate(platform_id: str | None, device: str) -> float:
    if platform_id and platform_id in PLATFORM_CATALOG:
        return float(PLATFORM_CATALOG[platform_id]["power_estimate_w"])
    if device.startswith("cuda"):
        return 45.0
    if device == "mps":
        return 6.0
    return 12.0


def benchmark_on_hardware(
    model: torch.nn.Module,
    *,
    device: str,
    platform_id: str | None = None,
    batch_size: int = 1,
    image_size: int = 128,
    warmup_runs: int = 10,
    test_runs: int = 100,
) -> dict[str, Any]:
    resolved_device = _resolve_device(device)
    input_tensor = torch.randn(batch_size, 3, image_size, image_size, device=resolved_device)
    process = psutil.Process()
    latencies_ms: list[float] = []
    memory_samples_mb: list[float] = []

    if resolved_device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model.eval()
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(input_tensor)
            synchronize_if_needed(resolved_device)

        if resolved_device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        for _ in range(test_runs):
            rss_before = process.memory_info().rss
            start = time.perf_counter()
            _ = model(input_tensor)
            synchronize_if_needed(resolved_device)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            rss_after = process.memory_info().rss
            latencies_ms.append(float(elapsed_ms))
            memory_samples_mb.append(float(max(rss_before, rss_after) / (1024.0 * 1024.0)))

    if resolved_device.startswith("cuda") and torch.cuda.is_available():
        peak_vram_mb = float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0))
    else:
        peak_vram_mb = 0.0

    mean_latency = float(np.mean(latencies_ms))
    std_latency = float(np.std(latencies_ms, ddof=1)) if len(latencies_ms) > 1 else 0.0
    p99_latency = float(np.quantile(latencies_ms, 0.99))
    mean_ram_mb = float(np.mean(memory_samples_mb))
    peak_ram_mb = float(max(memory_samples_mb))
    fps = float((1000.0 / mean_latency) * batch_size) if mean_latency > 0 else 0.0
    power_w = _power_estimate(platform_id, resolved_device)

    return {
        "platform_id": platform_id or "generic_local",
        "device": resolved_device,
        "batch_size": int(batch_size),
        "warmup_runs": int(warmup_runs),
        "test_runs": int(test_runs),
        "latency_mean_ms": round(mean_latency, 4),
        "latency_std_ms": round(std_latency, 4),
        "latency_p99_ms": round(p99_latency, 4),
        "ram_mean_mb": round(mean_ram_mb, 4),
        "ram_peak_mb": round(peak_ram_mb, 4),
        "vram_peak_mb": round(peak_vram_mb, 4),
        "memory_total_mb": round(mean_ram_mb + peak_vram_mb, 4),
        "fps": round(fps, 4),
        "power_estimate_w": round(power_w, 4),
        "real_time_constraint_ms": 100.0,
        "real_time_met": bool(mean_latency < 100.0),
        "host_platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
    }


def _output_dir(project_root: Path, platform_id: str) -> Path:
    directory = project_root / "outputs" / "benchmarks" / "edge" / platform_id
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark PAEMDT inference on an edge or cloud device.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--platform-id", default="generic_local")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--warmup-runs", type=int, default=10)
    parser.add_argument("--test-runs", type=int, default=100)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    output_dir = _output_dir(project_root, args.platform_id)
    resolved_device = _resolve_device(args.device)
    checkpoint = Path(args.checkpoint).resolve() if args.checkpoint else None
    model, model_label = _load_benchmark_model(checkpoint, resolved_device, image_size=args.image_size)

    benchmark = benchmark_on_hardware(
        model,
        device=resolved_device,
        platform_id=args.platform_id,
        batch_size=args.batch_size,
        image_size=args.image_size,
        warmup_runs=args.warmup_runs,
        test_runs=args.test_runs,
    )
    platform_meta = PLATFORM_CATALOG.get(args.platform_id, {})
    benchmark["platform_label"] = platform_meta.get("platform_label", args.platform_id)
    benchmark["recommended_device"] = platform_meta.get("recommended_device", resolved_device)
    benchmark["privacy_penalty"] = float(platform_meta.get("privacy_penalty", 0.12))
    benchmark["model_label"] = model_label

    benchmark_json = output_dir / "benchmark_summary.json"
    write_json(benchmark_json, benchmark)
    write_dataframe(output_dir / "benchmark_summary.csv", pd.DataFrame([benchmark]))

    print(f"Benchmark summary: {benchmark_json}")
    print(
        f"Latency={benchmark['latency_mean_ms']:.2f}+/-{benchmark['latency_std_ms']:.2f} ms "
        f"(p99={benchmark['latency_p99_ms']:.2f}) | Memory={benchmark['memory_total_mb']:.1f} MB "
        f"| FPS={benchmark['fps']:.2f} | Power={benchmark['power_estimate_w']:.1f} W"
    )


if __name__ == "__main__":
    main()
