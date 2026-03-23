from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from src.adapters.physiology_streams import build_real_adapter_manifest, export_ros2_jsonl, load_physiology_csv
from src.hardware.live_validation import write_validation_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full local project pipeline.")
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()

    subprocess.run([sys.executable, "-m", "src.evaluation.run_benchmarks", "--project-root", str(project_root)], check=True)
    subprocess.run([sys.executable, "-m", "src.visualization.generate_all_figures", "--project-root", str(project_root)], check=True)

    hardware_summary = write_validation_artifacts(project_root)
    packets = load_physiology_csv(project_root / "data" / "physiology" / "simulated_vitals.csv", source_name="site_ready_csv")
    ros2_jsonl = export_ros2_jsonl(packets[:40], project_root / "outputs" / "logs" / "physiology_ros2_bridge.jsonl")
    adapter_manifest = build_real_adapter_manifest(project_root)

    summary = {
        "benchmarks": str(project_root / "outputs" / "tables" / "benchmark_summary.csv"),
        "figures_dir": str(project_root / "outputs" / "figures"),
        "dashboard": str(project_root / "outputs" / "dashboard" / "index.html"),
        "hardware_validation": str(hardware_summary),
        "ros2_bridge_jsonl": str(ros2_jsonl),
        "real_adapter_manifest": str(adapter_manifest),
    }
    (project_root / "outputs" / "logs" / "full_local_pipeline_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
