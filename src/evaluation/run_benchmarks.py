from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.evaluation.ablation_plan import default_ablation_plan
from src.pipelines.baseline_mer_pipeline import describe_baseline
from src.visualization.export_plot_data import export_all_csv_artifacts


def _write_benchmark_manifest(output_path: Path) -> None:
    rows = [
        {
            "artifact": "baseline_mer",
            "status": "implemented_real_baseline",
            "notes": "Wraps existing DeepFace plus speech SVM pipeline",
        },
        {
            "artifact": "digital_twin_validation",
            "status": "simulation_based_evaluation",
            "notes": "Scenario package prepared, execution pending runtime and ROS2 details",
        },
        {
            "artifact": "multimodal_transformer",
            "status": "planned_experiment",
            "notes": "Architecture scaffold added, training not executed in this pass",
        },
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_ablation_manifest(output_path: Path) -> None:
    rows = [
        {
            "ablation_id": spec.ablation_id,
            "description": spec.description,
            "comparison_target": spec.comparison_target,
            "evidence_level": spec.evidence_level,
        }
        for spec in default_ablation_plan()
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export benchmark manifests and figure-ready CSV artifacts.",
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Project root containing outputs/, docs/, and src/",
    )
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    outputs_csv = project_root / "outputs" / "csv"
    outputs_tables = project_root / "outputs" / "tables"

    export_all_csv_artifacts(project_root)
    _write_benchmark_manifest(outputs_csv / "benchmark_manifest.csv")
    _write_ablation_manifest(outputs_tables / "ablation_manifest.csv")

    baseline_path = outputs_csv / "baseline_mer_description.csv"
    baseline = describe_baseline()
    with baseline_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["field", "value"])
        for key, value in baseline.items():
            writer.writerow([key, value])

    print(f"Exported benchmark CSV artifacts under {outputs_csv}")


if __name__ == "__main__":
    main()
