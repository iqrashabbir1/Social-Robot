from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.common.io_utils import write_dataframe
from src.common.paths import Paper1Paths
from src.evaluation.collect_hybrid_runtime_metrics import collect_hybrid_runtime_metrics
from src.evaluation.export_results import export_paper1_tables
from src.evaluation.run_dataset_evaluation import run_dataset_evaluation
from src.visualization.plot_hybrid_runtime import (
    plot_hybrid_system_architecture_paper_ready,
    plot_runtime_mode_heatmap_paper_ready,
    save_pilot_real_anchor_panel,
)


def generate_paper_ready_assets(project_root: Path) -> dict[str, str]:
    paths = Paper1Paths.from_project_root(project_root)
    paths.ensure()
    collect_hybrid_runtime_metrics(project_root)
    export_paper1_tables(project_root)
    dataset_outputs = run_dataset_evaluation(project_root=project_root, output_subdir="dataset_eval")

    paper_ready_dir = paths.outputs_figures_paper1 / "paper_ready"
    paper_ready_dir.mkdir(parents=True, exist_ok=True)

    plot_hybrid_system_architecture_paper_ready(
        paths.outputs_csv_paper1 / "hybrid_system_architecture_nodes.csv",
        paths.outputs_csv_paper1 / "hybrid_system_architecture_edges.csv",
        paper_ready_dir / "hybrid_system_architecture_paper_ready",
    )
    plot_runtime_mode_heatmap_paper_ready(
        paths.outputs_csv_paper1 / "runtime_mode_comparison.csv",
        paper_ready_dir / "runtime_evidence_matrix_paper_ready",
    )

    pilot_manifest_source = project_root / "data" / "pilot" / "sessions" / "paper1_anchor_demo" / "video_frames.csv"
    pilot_manifest_out = paths.outputs_csv_paper1 / "pilot_real_anchor_frame_manifest.csv"
    if pilot_manifest_source.exists():
        pilot_df = pd.read_csv(pilot_manifest_source)
        write_dataframe(pilot_manifest_out, pilot_df)
        save_pilot_real_anchor_panel(pilot_manifest_out, paper_ready_dir / "pilot_real_anchor_camera_panel")

    include_table = pd.DataFrame(
        [
            {
                "asset_type": "figure",
                "path": str(paper_ready_dir / "hybrid_system_architecture_paper_ready.png"),
                "paper_status": "include_main",
                "reason": "Core methods/platform architecture figure backed by current verified runtime design.",
            },
            {
                "asset_type": "figure",
                "path": str(paper_ready_dir / "runtime_evidence_matrix_paper_ready.png"),
                "paper_status": "include_main_or_supplement",
                "reason": "Compact runtime evidence summary derived from the verified mode-comparison matrix.",
            },
            {
                "asset_type": "figure",
                "path": str(paper_ready_dir / "pilot_real_anchor_camera_panel.png"),
                "paper_status": "include_qualitative",
                "reason": "True visual example from the small real-anchor capture already tracked in the repo.",
            },
            {
                "asset_type": "figure",
                "path": str(project_root / "outputs" / "figures" / "paper1" / "dataset_prediction_panel.png"),
                "paper_status": "supplement_only_current_local_data",
                "reason": "Generated correctly, but the current local image set is an unlabeled room-scene pilot capture and should not be used as a main manuscript figure.",
            },
            {
                "asset_type": "figure",
                "path": str(project_root / "outputs" / "figures" / "paper1" / "dataset_replay_sequence.png"),
                "paper_status": "include_main_or_supplement",
                "reason": "Shows that the controlled dataset frames can be replayed through the ROS2 image pipeline.",
            },
            {
                "asset_type": "table",
                "path": str(paths.outputs_tables / "paper1_table_runtime_evidence_summary.csv"),
                "paper_status": "include_main",
                "reason": "Primary runtime evidence summary table for Paper 1.",
            },
            {
                "asset_type": "table",
                "path": str(paths.outputs_tables / "paper1_table_mode_comparison.csv"),
                "paper_status": "include_main_or_supplement",
                "reason": "Supports runtime comparison without relying on placeholder graphics.",
            },
            {
                "asset_type": "table",
                "path": str(paths.outputs_tables / "paper1_table_dataset_summary.csv"),
                "paper_status": "include_main",
                "reason": "Describes the controlled dataset evaluation setup and label availability.",
            },
            {
                "asset_type": "table",
                "path": str(paths.outputs_tables / "paper1_table_dataset_metrics.csv"),
                "paper_status": "supplement_only_current_local_data",
                "reason": "Current local metrics are coverage and confidence summaries because the tracked local dataset does not yet include ground-truth emotion labels.",
            },
            {
                "asset_type": "table",
                "path": str(paths.outputs_tables / "paper1_table_runtime_vs_dataset_evidence.csv"),
                "paper_status": "include_main_or_supplement",
                "reason": "Clarifies why offline dataset evaluation and ROS dataset replay complement the live hybrid runtime.",
            },
        ]
    )
    include_table_path = paths.outputs_tables / "paper1_table_paper_ready_assets.csv"
    write_dataframe(include_table_path, include_table)
    return {
        "architecture_figure": str(paper_ready_dir / "hybrid_system_architecture_paper_ready.png"),
        "runtime_evidence_matrix": str(paper_ready_dir / "runtime_evidence_matrix_paper_ready.png"),
        "pilot_real_anchor_panel": str(paper_ready_dir / "pilot_real_anchor_camera_panel.png"),
        "dataset_predictions_csv": dataset_outputs["predictions_csv"],
        "paper_ready_assets_table": str(include_table_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate only the Paper 1 assets that are safe to include now.")
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    generate_paper_ready_assets(Path(args.project_root).resolve())


if __name__ == "__main__":
    main()
