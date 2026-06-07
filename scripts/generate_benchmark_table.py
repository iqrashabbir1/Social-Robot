from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.io_utils import write_dataframe, write_json
from src.common.paths import Paper1Paths
from scripts.benchmark_edge import PLATFORM_CATALOG


def _load_benchmark_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _default_run_paths(project_root: Path) -> dict[str, Path]:
    base = project_root / "outputs" / "benchmarks" / "edge"
    return {platform_id: base / platform_id / "benchmark_summary.json" for platform_id in PLATFORM_CATALOG}


def _real_time_label(mean_latency_ms: float | None) -> str:
    if mean_latency_ms is None or pd.isna(mean_latency_ms):
        return "Pending"
    return "Yes" if float(mean_latency_ms) < 100.0 else "No"


def _latency_display(mean_ms: float | None, std_ms: float | None, p99_ms: float | None) -> str:
    if mean_ms is None or std_ms is None or p99_ms is None or any(pd.isna(value) for value in (mean_ms, std_ms, p99_ms)):
        return "Pending device run"
    return f"{float(mean_ms):.1f} +/- {float(std_ms):.1f} (p99={float(p99_ms):.1f})"


def _plot_figure9(df: pd.DataFrame, output_base: Path) -> None:
    measured = df.loc[df["benchmark_status"] == "measured"].copy()
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    if measured.empty:
        ax.text(0.5, 0.5, "No measured platform benchmarks have been aggregated yet.", ha="center", va="center", fontsize=12)
        ax.axis("off")
    else:
        scatter = ax.scatter(
            measured["latency_mean_ms"],
            measured["utility_macro_f1"],
            s=measured["privacy_penalty"].astype(float) * 2400.0 + 160.0,
            c=measured["power_estimate_w"],
            cmap="viridis",
            alpha=0.85,
            edgecolors="black",
            linewidths=0.7,
        )
        for _, row in measured.iterrows():
            ax.annotate(str(row["Platform"]), (row["latency_mean_ms"], row["utility_macro_f1"]), xytext=(6, 4), textcoords="offset points", fontsize=9)
        colorbar = fig.colorbar(scatter, ax=ax)
        colorbar.set_label("Power estimate (W)")
        ax.set_title("Figure 9. Privacy-Utility-Latency Pareto Analysis with Measured Benchmarks")
        ax.set_xlabel("Latency (ms)")
        ax.set_ylabel("Utility (Macro-F1)")
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def generate_benchmark_table(project_root: Path, utility_macro_f1: float = 0.978) -> dict[str, str]:
    paths = Paper1Paths.from_project_root(project_root)
    paths.ensure()
    rows: list[dict[str, object]] = []
    for platform_id, run_path in _default_run_paths(project_root).items():
        meta = PLATFORM_CATALOG[platform_id]
        if run_path.exists():
            payload = _load_benchmark_json(run_path)
            row = {
                "Platform": meta["platform_label"],
                "Device": payload.get("device", meta["recommended_device"]),
                "Latency (ms)": _latency_display(payload.get("latency_mean_ms"), payload.get("latency_std_ms"), payload.get("latency_p99_ms")),
                "Memory (MB)": round(float(payload.get("memory_total_mb", 0.0)), 1),
                "FPS": round(float(payload.get("fps", 0.0)), 1),
                "Power (W)": round(float(payload.get("power_estimate_w", meta["power_estimate_w"])), 1),
                "Real-time?": _real_time_label(payload.get("latency_mean_ms")),
                "benchmark_status": "measured",
                "latency_mean_ms": float(payload.get("latency_mean_ms", float("nan"))),
                "latency_std_ms": float(payload.get("latency_std_ms", float("nan"))),
                "latency_p99_ms": float(payload.get("latency_p99_ms", float("nan"))),
                "memory_total_mb": float(payload.get("memory_total_mb", float("nan"))),
                "fps_numeric": float(payload.get("fps", float("nan"))),
                "power_estimate_w": float(payload.get("power_estimate_w", meta["power_estimate_w"])),
                "privacy_penalty": float(meta["privacy_penalty"]),
                "utility_macro_f1": float(utility_macro_f1),
                "source_json": str(run_path.resolve()),
            }
        else:
            row = {
                "Platform": meta["platform_label"],
                "Device": meta["recommended_device"],
                "Latency (ms)": "Pending device run",
                "Memory (MB)": None,
                "FPS": None,
                "Power (W)": float(meta["power_estimate_w"]),
                "Real-time?": "Pending",
                "benchmark_status": "missing",
                "latency_mean_ms": float("nan"),
                "latency_std_ms": float("nan"),
                "latency_p99_ms": float("nan"),
                "memory_total_mb": float("nan"),
                "fps_numeric": float("nan"),
                "power_estimate_w": float(meta["power_estimate_w"]),
                "privacy_penalty": float(meta["privacy_penalty"]),
                "utility_macro_f1": float(utility_macro_f1),
                "source_json": "",
            }
        rows.append(row)

    full_df = pd.DataFrame(rows)
    table_df = full_df[["Platform", "Device", "Latency (ms)", "Memory (MB)", "FPS", "Power (W)", "Real-time?"]].copy()
    table_path = paths.outputs_tables / "paper1_table_edge_benchmark.csv"
    detailed_path = paths.outputs_csv_paper1 / "edge_benchmark_detailed.csv"
    write_dataframe(table_path, table_df)
    write_dataframe(detailed_path, full_df)

    figure_base = paths.outputs_figures_paper1 / "privacy_utility_latency_pareto_measured"
    _plot_figure9(full_df, figure_base)

    write_json(
        paths.outputs_csv_paper1 / "edge_benchmark_manifest.json",
        {
            "table_csv": str(table_path.resolve()),
            "detailed_csv": str(detailed_path.resolve()),
            "figure_png": str(figure_base.with_suffix(".png").resolve()),
            "figure_svg": str(figure_base.with_suffix(".svg").resolve()),
            "measured_platforms": full_df.loc[full_df["benchmark_status"] == "measured", "Platform"].tolist(),
            "missing_platforms": full_df.loc[full_df["benchmark_status"] != "measured", "Platform"].tolist(),
        },
    )

    return {
        "table_csv": str(table_path.resolve()),
        "detailed_csv": str(detailed_path.resolve()),
        "figure_png": str(figure_base.with_suffix(".png").resolve()),
        "figure_svg": str(figure_base.with_suffix(".svg").resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate measured platform benchmarks into the paper-ready edge benchmark table.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--utility-macro-f1", type=float, default=0.978)
    args = parser.parse_args()

    outputs = generate_benchmark_table(Path(args.project_root).resolve(), utility_macro_f1=args.utility_macro_f1)
    print(f"Benchmark table: {outputs['table_csv']}")
    print(f"Detailed CSV: {outputs['detailed_csv']}")
    print(f"Figure 9 PNG: {outputs['figure_png']}")


if __name__ == "__main__":
    main()
