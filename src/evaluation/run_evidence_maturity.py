from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.common.io_utils import write_dataframe


ROWS = [
    {"Module": "Core multimodal benchmark", "Implementation": "Implemented", "Experimental_Validation": "Implemented", "Translational_Readiness": "Partial", "Evidence_Note": "benchmark-supported technical evidence"},
    {"Module": "Domain adaptation", "Implementation": "Implemented", "Experimental_Validation": "Implemented", "Translational_Readiness": "Partial", "Evidence_Note": "manuscript-facing enhanced benchmark evidence"},
    {"Module": "Differential privacy", "Implementation": "Implemented", "Experimental_Validation": "Partial", "Translational_Readiness": "Partial", "Evidence_Note": "DP-accounted manuscript configuration"},
    {"Module": "Digital-twin predictive replay", "Implementation": "Implemented", "Experimental_Validation": "Implemented", "Translational_Readiness": "Partial", "Evidence_Note": "replay-grounded technical validation"},
    {"Module": "Missing-modality robustness", "Implementation": "Implemented", "Experimental_Validation": "Implemented", "Translational_Readiness": "Partial", "Evidence_Note": "stress-test technical validation"},
    {"Module": "Edge deployment", "Implementation": "Implemented", "Experimental_Validation": "Partial", "Translational_Readiness": "Partial", "Evidence_Note": "Raspberry Pi 4 latency measurement"},
    {"Module": "Physical robot deployment", "Implementation": "Future required", "Experimental_Validation": "Future required", "Translational_Readiness": "Future required", "Evidence_Note": "no real-world deployment yet"},
    {"Module": "Clinical validation", "Implementation": "Future required", "Experimental_Validation": "Future required", "Translational_Readiness": "Future required", "Evidence_Note": "no prospective clinical validation yet"},
]


def generate_evidence_maturity_outputs(project_root: Path) -> dict[str, Path]:
    df = pd.DataFrame(ROWS)
    paths = {
        "csv": project_root / "outputs" / "csv" / "evidence_maturity_matrix.csv",
        "summary": project_root / "outputs" / "tables" / "evidence_maturity_summary.csv",
    }
    write_dataframe(paths["csv"], df)
    write_dataframe(paths["summary"], df)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate evidence-maturity outputs for Paper 1.")
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    for label, path in generate_evidence_maturity_outputs(Path(args.project_root).resolve()).items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
