from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.io_utils import write_dataframe, write_json
from src.models.torch_runtime import require_torch, synchronize_if_needed
from src.models.vision.image_emotion_classifier import SimpleEmotionCNN, load_model_checkpoint


torch = require_torch()


def _resolve_device(device: str) -> str:
    normalized = str(device).strip().lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if normalized == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for profiling, but torch.cuda.is_available() is false.")
    if normalized == "mps":
        if not bool(getattr(torch.backends, "mps", None)) or not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested for profiling, but torch.backends.mps.is_available() is false.")
    return normalized


def _load_model(checkpoint_path: Path | None, device: str, image_size: int = 128) -> tuple[torch.nn.Module, str]:
    if checkpoint_path and checkpoint_path.exists():
        model_bundle, _, _, _ = load_model_checkpoint(checkpoint_path, device=device)
        return model_bundle.network.to(device).eval(), f"checkpoint:{checkpoint_path.name}"
    return SimpleEmotionCNN(num_classes=4, image_size=image_size).network.to(device).eval(), "cnn_small_fresh_init"


def _activities_for_device(device: str) -> list[torch.profiler.ProfilerActivity]:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.startswith("cuda") and torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    return activities


def _plot_flamegraph(table_df: pd.DataFrame, output_base: Path) -> None:
    top_df = table_df.sort_values("self_time_ms", ascending=False).head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    ax.barh(top_df["name"], top_df["self_time_ms"], color="#4477AA")
    ax.set_title("Model Profiling Flamegraph (Top Operations by Self Time)")
    ax.set_xlabel("Self time (ms)")
    ax.set_ylabel("Operation")
    fig.tight_layout()
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def profile_model_forward(
    model: torch.nn.Module,
    *,
    device: str,
    batch_size: int = 1,
    image_size: int = 128,
    warmup_runs: int = 5,
    active_runs: int = 10,
) -> tuple[pd.DataFrame, dict[str, str]]:
    resolved_device = _resolve_device(device)
    inputs = torch.randn(batch_size, 3, image_size, image_size, device=resolved_device)
    activities = _activities_for_device(resolved_device)

    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(inputs)
            synchronize_if_needed(resolved_device)

        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        ) as profiler:
            for _ in range(active_runs):
                _ = model(inputs)
                synchronize_if_needed(resolved_device)
                profiler.step()

    rows: list[dict[str, object]] = []
    for event in profiler.key_averages():
        rows.append(
            {
                "name": event.key,
                "cpu_time_total_ms": float(event.cpu_time_total / 1000.0),
                "self_time_ms": float(event.self_cpu_time_total / 1000.0),
                "cuda_time_total_ms": float(getattr(event, "cuda_time_total", 0.0) / 1000.0),
                "cpu_memory_mb": float(event.cpu_memory_usage / (1024.0 * 1024.0)),
                "cuda_memory_mb": float(getattr(event, "cuda_memory_usage", 0.0) / (1024.0 * 1024.0)),
                "count": int(event.count),
                "input_shapes": str(event.input_shapes),
            }
        )
    return pd.DataFrame(rows), {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile PAEMDT model inference and export per-layer timing.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--warmup-runs", type=int, default=5)
    parser.add_argument("--active-runs", type=int, default=10)
    parser.add_argument("--output-subdir", default="model_profile")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    output_dir = project_root / "outputs" / "benchmarks" / "profiling" / args.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_device = _resolve_device(args.device)
    checkpoint = Path(args.checkpoint).resolve() if args.checkpoint else None
    model, model_label = _load_model(checkpoint, resolved_device, image_size=args.image_size)
    table_df, _ = profile_model_forward(
        model,
        device=resolved_device,
        batch_size=args.batch_size,
        image_size=args.image_size,
        warmup_runs=args.warmup_runs,
        active_runs=args.active_runs,
    )

    csv_path = output_dir / "profile_layers.csv"
    trace_path = output_dir / "profile_trace.json"
    flamegraph_base = output_dir / "profile_flamegraph"
    write_dataframe(csv_path, table_df)

    # Lightweight trace export for Chrome tracing / TensorBoard viewers.
    with torch.profiler.profile(
        activities=_activities_for_device(resolved_device),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as trace_profiler:
        with torch.no_grad():
            inputs = torch.randn(args.batch_size, 3, args.image_size, args.image_size, device=resolved_device)
            _ = model(inputs)
            synchronize_if_needed(resolved_device)
            trace_profiler.step()
    trace_profiler.export_chrome_trace(str(trace_path))

    _plot_flamegraph(table_df, flamegraph_base)
    write_json(
        output_dir / "profile_manifest.json",
        {
            "device": resolved_device,
            "model_label": model_label,
            "layers_csv": str(csv_path.resolve()),
            "trace_json": str(trace_path.resolve()),
            "flamegraph_png": str(flamegraph_base.with_suffix(".png").resolve()),
            "flamegraph_svg": str(flamegraph_base.with_suffix(".svg").resolve()),
        },
    )

    print(f"Layer profile CSV: {csv_path}")
    print(f"Flamegraph PNG: {flamegraph_base.with_suffix('.png')}")
    print(f"Chrome trace: {trace_path}")


if __name__ == "__main__":
    main()
