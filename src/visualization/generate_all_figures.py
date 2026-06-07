from __future__ import annotations

import argparse
from pathlib import Path

from src.evaluation.collect_hybrid_runtime_metrics import collect_hybrid_runtime_metrics
from src.evaluation.extract_rosbag_summary import extract_rosbag_summary
from src.evaluation.run_calibration_analysis import generate_calibration_outputs
from src.evaluation.run_digital_twin_sync_analysis import generate_digital_twin_sync_outputs
from src.evaluation.run_domain_adaptation import generate_domain_adaptation_outputs
from src.evaluation.run_dp_privacy_accounting import generate_dp_privacy_accounting
from src.evaluation.run_evidence_maturity import generate_evidence_maturity_outputs
from src.evaluation.run_missing_modality_robustness import generate_missing_modality_outputs
from src.evaluation.run_privacy_latency_analysis import generate_privacy_latency_outputs
from src.evaluation.run_repeated_cv_statistics import generate_repeated_cv_statistics
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
from src.visualization.plot_ablation_analysis import generate_ablation_outputs
from src.visualization.plot_calibration_analysis import plot_calibration_analysis
from src.visualization.plot_domain_generalization import generate_domain_generalization_outputs
from src.visualization.plot_evidence_maturity import plot_evidence_maturity
from src.visualization.plot_hybrid_runtime import (
    plot_hybrid_camera_fps,
    plot_hybrid_system_architecture,
    plot_runtime_mode_comparison,
    plot_runtime_verification,
    plot_system_health,
)
from src.visualization.plot_missing_modality_robustness import plot_missing_modality_robustness
from src.visualization.plot_privacy_latency_pareto import plot_privacy_latency_pareto
from src.visualization.plot_statistical_significance import plot_repeated_cv_confidence_intervals
from src.visualization.save_camera_sample_frames import save_camera_sample_frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate all Paper 1 figures from CSV inputs.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--rosbag-dir", default="")
    parser.add_argument("--frame-csv", default="")
    parser.add_argument("--frame-dir", default="")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    da_paths = generate_domain_adaptation_outputs(project_root)
    dp_paths = generate_dp_privacy_accounting(project_root)
    calibration_paths = generate_calibration_outputs(project_root)
    missing_paths = generate_missing_modality_outputs(project_root)
    privacy_latency_paths = generate_privacy_latency_outputs(project_root)
    dt_paths = generate_digital_twin_sync_outputs(project_root)
    evidence_paths = generate_evidence_maturity_outputs(project_root)
    print(f"Domain results CSV: {da_paths['domain_results']}")
    print(f"DP privacy CSV: {dp_paths['csv']}")
    print(f"Calibration CSV: {calibration_paths['csv']}")
    print(f"Missing-modality CSV: {missing_paths['csv']}")
    print(f"Privacy-latency CSV: {privacy_latency_paths['csv']}")
    print(f"Digital-twin sync CSV: {dt_paths['csv']}")
    print(f"Evidence maturity CSV: {evidence_paths['csv']}")

    cv_paths = generate_repeated_cv_statistics(project_root)
    figure6_paths = plot_repeated_cv_confidence_intervals(
        cv_paths["summary"],
        project_root / "outputs" / "figures" / "Figure_6_Repeated_CV_Confidence_Intervals",
    )
    print(f"Repeated CV results: {cv_paths['repeated_results']}")
    print(f"Repeated CV summary: {cv_paths['summary']}")
    print(f"Figure 6 PNG: {figure6_paths['png']}")
    print(f"Figure 6 PDF: {figure6_paths['pdf']}")
    print(f"Figure 6 SVG: {figure6_paths['svg']}")
    domain_paths = generate_domain_generalization_outputs(project_root)
    print(f"Table 4 CSV: {domain_paths['table4']}")
    print(f"Table 4b CSV: {domain_paths['table4b']}")
    print(f"Figure 3 PNG: {domain_paths['figure3_png']}")
    print(f"Figure 3 PDF: {domain_paths['figure3_pdf']}")
    print(f"Figure 3 SVG: {domain_paths['figure3_svg']}")
    print(f"Figure 4 PNG: {domain_paths['figure4_png']}")
    print(f"Figure 4 PDF: {domain_paths['figure4_pdf']}")
    print(f"Figure 4 SVG: {domain_paths['figure4_svg']}")
    ablation_paths = generate_ablation_outputs(project_root)
    print(f"Table 5 outputs CSV: {ablation_paths['outputs_table']}")
    print(f"Table 5 paper CSV: {ablation_paths['paper_table']}")
    print(f"Figure 5 PNG: {ablation_paths['png']}")
    print(f"Figure 5 PDF: {ablation_paths['pdf']}")
    print(f"Figure 5 SVG: {ablation_paths['svg']}")
    figure7_paths = plot_calibration_analysis(calibration_paths["csv"], project_root / "outputs" / "figures" / "Figure_7_ECE_Comparison")
    figure8_paths = plot_missing_modality_robustness(missing_paths["csv"], project_root / "outputs" / "figures" / "Figure_8_Missing_Modality_Robustness")
    figure9_paths = plot_privacy_latency_pareto(privacy_latency_paths["csv"], project_root / "outputs" / "figures" / "Figure_9_Privacy_Utility_Latency")
    figure10_paths = plot_evidence_maturity(evidence_paths["csv"], project_root / "outputs" / "figures" / "Figure_10_Evidence_Maturity_Dashboard")
    print(f"Figure 7 PNG: {figure7_paths['png']}")
    print(f"Figure 7 PDF: {figure7_paths['pdf']}")
    print(f"Figure 7 SVG: {figure7_paths['svg']}")
    print(f"Figure 8 PNG: {figure8_paths['png']}")
    print(f"Figure 8 PDF: {figure8_paths['pdf']}")
    print(f"Figure 8 SVG: {figure8_paths['svg']}")
    print(f"Figure 9 PNG: {figure9_paths['png']}")
    print(f"Figure 9 PDF: {figure9_paths['pdf']}")
    print(f"Figure 9 SVG: {figure9_paths['svg']}")
    print(f"Figure 10 PNG: {figure10_paths['png']}")
    print(f"Figure 10 PDF: {figure10_paths['pdf']}")
    print(f"Figure 10 SVG: {figure10_paths['svg']}")

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
