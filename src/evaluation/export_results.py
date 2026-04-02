from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.common.io_utils import write_dataframe
from src.common.paths import Paper1Paths


def export_paper1_tables(project_root: Path) -> dict[str, str]:
    paths = Paper1Paths.from_project_root(project_root)
    paths.ensure()

    interface_df = pd.read_csv(paths.outputs_csv_cs1 / "interface_spec.csv")
    session_df = pd.read_csv(paths.outputs_csv_cs2 / "session_metadata.csv")
    performance_df = pd.read_csv(paths.outputs_csv_cs3 / "model_performance_summary.csv")
    cs1_summary_df = pd.read_csv(paths.outputs_csv_cs1 / "latency_summary.csv")
    cs2_sync_df = pd.read_csv(paths.outputs_csv_cs2 / "sync_quality_metrics.csv")
    ablation_df = pd.read_csv(paths.outputs_csv_cs3 / "ablation_results.csv")

    system_table = pd.DataFrame(
        [
            {"component_group": "ROS2 interfaces", "count": len(interface_df), "description": "Paper 1 topic and interface specification."},
            {"component_group": "Sessions", "count": len(session_df), "description": "Multimodal synchronization sessions generated for CS2."},
            {"component_group": "Benchmark families", "count": len(performance_df), "description": "B0 through B3 evaluation rows."},
        ]
    )
    metrics_rows: list[dict[str, object]] = []
    for row in cs1_summary_df.to_dict(orient="records"):
        for metric in (
            "mean_latency_ms",
            "p95_latency_ms",
            "max_latency_ms",
            "message_drop_rate",
            "task_success_rate",
            "recovery_rate",
            "cpu_percent_mean",
            "memory_mb_mean",
        ):
            metrics_rows.append(
                {
                    "section": "CS1",
                    "entity": row["mode"],
                    "metric": metric,
                    "value": row[metric],
                    "evidence_level": "simulation_based_evaluation",
                }
            )
    for row in cs2_sync_df.to_dict(orient="records"):
        for metric in (
            "mean_alignment_error_ms",
            "full_modality_window_rate",
            "video_available_rate",
            "audio_available_rate",
            "context_available_rate",
            "physiology_available_rate",
        ):
            metrics_rows.append(
                {
                    "section": "CS2",
                    "entity": row["condition"],
                    "metric": metric,
                    "value": row[metric],
                    "evidence_level": "synthetic_placeholder_benchmark",
                }
            )
    for row in performance_df.to_dict(orient="records"):
        for metric in (
            "accuracy",
            "macro_f1",
            "weighted_f1",
            "uar",
            "inference_latency_ms",
            "robustness_missing_modality_macro_f1",
        ):
            metrics_rows.append(
                {
                    "section": "CS3",
                    "entity": row["model_id"],
                    "metric": metric,
                    "value": row.get(metric),
                    "evidence_level": row["evidence_level"],
                }
            )
    metrics_table = pd.DataFrame(metrics_rows)
    ablation_table = ablation_df.sort_values(["model_id", "ablation_name", "condition"]).reset_index(drop=True)

    write_dataframe(paths.outputs_tables / "paper1_table_system_summary.csv", system_table)
    write_dataframe(paths.outputs_tables / "paper1_table_metrics_summary.csv", metrics_table)
    write_dataframe(paths.outputs_tables / "paper1_table_ablation_summary.csv", ablation_table)

    return {
        "paper1_table_system_summary": str(paths.outputs_tables / "paper1_table_system_summary.csv"),
        "paper1_table_metrics_summary": str(paths.outputs_tables / "paper1_table_metrics_summary.csv"),
        "paper1_table_ablation_summary": str(paths.outputs_tables / "paper1_table_ablation_summary.csv"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Paper 1 summary tables.")
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    export_paper1_tables(Path(args.project_root).resolve())


if __name__ == "__main__":
    main()
