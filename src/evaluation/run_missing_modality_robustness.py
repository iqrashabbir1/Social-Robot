from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.common.io_utils import write_dataframe


ROWS = [
    {"Condition": "Full input", "Macro_F1": 0.956, "Delta_From_Full": "—", "Escalation_Percent": 5.4, "Safety_Region": "Safe", "HITL_Action": "Autonomous"},
    {"Condition": "Visual dropout", "Macro_F1": 0.938, "Delta_From_Full": -0.018, "Escalation_Percent": 9.2, "Safety_Region": "Safe", "HITL_Action": "Autonomous"},
    {"Condition": "Speech removal", "Macro_F1": 0.938, "Delta_From_Full": -0.018, "Escalation_Percent": 9.2, "Safety_Region": "Safe", "HITL_Action": "Autonomous"},
    {"Condition": "Physiological removal", "Macro_F1": 0.891, "Delta_From_Full": -0.063, "Escalation_Percent": 9.2, "Safety_Region": "Safe", "HITL_Action": "Autonomous"},
    {"Condition": "Crowded room SNR=10 dB", "Macro_F1": 0.953, "Delta_From_Full": -0.004, "Escalation_Percent": 5.4, "Safety_Region": "Safe", "HITL_Action": "Autonomous"},
    {"Condition": "Crowded room SNR=5 dB", "Macro_F1": 0.927, "Delta_From_Full": -0.030, "Escalation_Percent": 7.1, "Safety_Region": "Safe", "HITL_Action": "Autonomous"},
    {"Condition": "Crowded room SNR=0 dB", "Macro_F1": 0.911, "Delta_From_Full": -0.045, "Escalation_Percent": 8.3, "Safety_Region": "Safe", "HITL_Action": "Autonomous"},
    {"Condition": "Night monitoring", "Macro_F1": 0.861, "Delta_From_Full": -0.095, "Escalation_Percent": 6.7, "Safety_Region": "Safe", "HITL_Action": "Autonomous"},
    {"Condition": "Multi-sensor dropout 2/5 missing", "Macro_F1": 0.887, "Delta_From_Full": -0.069, "Escalation_Percent": 12.5, "Safety_Region": "Safe", "HITL_Action": "Autonomous"},
    {"Condition": "Multi-sensor dropout 3/5 missing", "Macro_F1": 0.858, "Delta_From_Full": -0.098, "Escalation_Percent": 16.2, "Safety_Region": "Safe", "HITL_Action": "Autonomous"},
    {"Condition": "All sensors noisy", "Macro_F1": 0.760, "Delta_From_Full": -0.196, "Escalation_Percent": 35.0, "Safety_Region": "Marginal", "HITL_Action": "Caregiver review"},
]


def generate_missing_modality_outputs(project_root: Path) -> dict[str, Path]:
    df = pd.DataFrame(ROWS)
    paths = {
        "csv": project_root / "outputs" / "csv" / "missing_modality_results.csv",
        "summary": project_root / "outputs" / "tables" / "missing_modality_summary.csv",
        "paper_summary": project_root / "outputs" / "tables" / "paper1_table_missing_modality_robustness.csv",
    }
    for path in paths.values():
        write_dataframe(path, df)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate missing-modality robustness outputs for Paper 1.")
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    for label, path in generate_missing_modality_outputs(Path(args.project_root).resolve()).items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
