from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.common.io_utils import write_dataframe


ROWS = [
    {"Module": "Core multimodal benchmark", "Implementation": "Implemented", "Experimental validation": "Implemented", "Translational readiness": "Partial"},
    {"Module": "Domain adaptation", "Implementation": "Implemented", "Experimental validation": "Implemented", "Translational readiness": "Partial"},
    {"Module": "Differential privacy", "Implementation": "Implemented", "Experimental validation": "Partial", "Translational readiness": "Partial"},
    {"Module": "Digital-twin predictive replay", "Implementation": "Implemented", "Experimental validation": "Implemented", "Translational readiness": "Partial"},
    {"Module": "Missing-modality robustness", "Implementation": "Implemented", "Experimental validation": "Implemented", "Translational readiness": "Partial"},
    {"Module": "Edge deployment", "Implementation": "Implemented", "Experimental validation": "Partial", "Translational readiness": "Partial"},
    {"Module": "Physical robot deployment", "Implementation": "Future required", "Experimental validation": "Future required", "Translational readiness": "Future required"},
    {"Module": "Clinical validation", "Implementation": "Future required", "Experimental validation": "Future required", "Translational readiness": "Future required"},
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
