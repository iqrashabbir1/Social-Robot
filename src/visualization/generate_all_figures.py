from __future__ import annotations

import argparse
from pathlib import Path

from src.evaluation.collect_hybrid_runtime_metrics import collect_hybrid_runtime_metrics
from src.evaluation.extract_rosbag_summary import extract_rosbag_summary
from src.visualization.plot_cs1 import (
    plot_latency_distribution,
    plot_resource_usage,
    plot_simulator_vs_playback,
    plot_sync_error,
    plot_system_architecture,
    plot_task_success,
)
from src.visualization.plot_cs2 import plot_missing_modality, plot_modality_heatmap, plot_sync_quality
from src.visualization.plot_cs3 import generate_cs3_figures
from src.visualization.plot_hybrid_runtime import (
    plot_hybrid_camera_fps,
    plot_hybrid_system_architecture,
    plot_runtime_mode_comparison,
    plot_runtime_verification,
    plot_system_health,
)
from src.visualization.save_camera_sample_frames import save_camera_sample_frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate all Paper 1 figures from CSV inputs.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--rosbag-dir", default="")
    parser.add_argument("--frame-csv", default="")
    parser.add_argument("--frame-dir", default="")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    for experiment_dir in sorted((project_root / "outputs" / "csv" / "cs1").glob("*")):
        if not experiment_dir.is_dir():
            continue
        figure_dir = project_root / "outputs" / "figures" / "cs1" / experiment_dir.name
        if (experiment_dir / "interface_spec.csv").exists():
            plot_system_architecture(experiment_dir / "interface_spec.csv", figure_dir / "system_architecture_diagram")
        if (experiment_dir / "latency_metrics.csv").exists():
            plot_latency_distribution(experiment_dir / "latency_metrics.csv", figure_dir / "latency_distribution")
            plot_task_success(experiment_dir / "latency_metrics.csv", figure_dir / "task_success_comparison")
            plot_resource_usage(experiment_dir / "latency_metrics.csv", figure_dir / "resource_usage")
        if (experiment_dir / "sync_error_timeseries.csv").exists():
            plot_sync_error(experiment_dir / "sync_error_timeseries.csv", figure_dir / "synchronization_error_over_time")
        if (experiment_dir / "simulator_vs_playback_comparison.csv").exists():
            plot_simulator_vs_playback(experiment_dir / "simulator_vs_playback_comparison.csv", figure_dir / "simulator_vs_playback_comparison")

    for experiment_dir in sorted((project_root / "outputs" / "csv" / "cs2").glob("*")):
        if not experiment_dir.is_dir():
            continue
        figure_dir = project_root / "outputs" / "figures" / "cs2" / experiment_dir.name
        if (experiment_dir / "modality_availability.csv").exists():
            plot_modality_heatmap(experiment_dir / "modality_availability.csv", figure_dir / "modality_availability_heatmap")
        if (experiment_dir / "sync_quality_metrics.csv").exists():
            plot_sync_quality(experiment_dir / "sync_quality_metrics.csv", figure_dir / "synchronization_quality_comparison")
            plot_missing_modality(experiment_dir / "sync_quality_metrics.csv", figure_dir / "missing_modality_robustness")

    generate_cs3_figures(project_root)
    extract_rosbag_summary(Path(args.rosbag_dir).resolve() if args.rosbag_dir else None, project_root)
    hybrid_assets = collect_hybrid_runtime_metrics(project_root)
    save_camera_sample_frames(
        project_root=project_root,
        frame_csv=Path(args.frame_csv).resolve() if args.frame_csv else None,
        frame_dir=Path(args.frame_dir).resolve() if args.frame_dir else None,
    )
    paper1_figure_dir = project_root / "outputs" / "figures" / "paper1"
    plot_hybrid_system_architecture(
        Path(hybrid_assets["architecture_nodes_csv"]),
        Path(hybrid_assets["architecture_edges_csv"]),
        paper1_figure_dir / "hybrid_system_architecture",
    )
    plot_runtime_verification(Path(hybrid_assets["verification_csv"]), paper1_figure_dir / "ros2_runtime_verification")
    plot_hybrid_camera_fps(Path(hybrid_assets["fps_csv"]), paper1_figure_dir / "hybrid_camera_fps_over_time")
    plot_system_health(Path(hybrid_assets["health_csv"]), paper1_figure_dir / "system_health_over_time")
    plot_runtime_mode_comparison(
        Path(hybrid_assets["runtime_mode_comparison_csv"]),
        paper1_figure_dir / "runtime_mode_comparison",
    )


if __name__ == "__main__":
    main()
