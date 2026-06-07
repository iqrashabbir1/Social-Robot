from __future__ import annotations

import argparse
import sys
from pathlib import Path


REQUIRED_CSVS = [
    "outputs/csv/domain_generalization_results.csv",
    "outputs/csv/dp_privacy_accounting.csv",
    "outputs/csv/repeated_cv_results.csv",
    "outputs/csv/calibration_results.csv",
    "outputs/csv/missing_modality_results.csv",
    "outputs/csv/privacy_latency_results.csv",
    "outputs/csv/digital_twin_sync_results.csv",
    "outputs/csv/evidence_maturity_matrix.csv",
]

REQUIRED_TABLES = [
    "outputs/tables/enhanced_benchmark_comparison.csv",
    "outputs/tables/domain_adaptation_progression.csv",
    "outputs/tables/ablation_summary.csv",
    "outputs/tables/repeated_cv_summary.csv",
    "outputs/tables/statistical_test_summary.csv",
    "outputs/tables/privacy_accounting_summary.csv",
    "outputs/tables/calibration_summary.csv",
    "outputs/tables/missing_modality_summary.csv",
    "outputs/tables/privacy_latency_summary.csv",
    "outputs/tables/digital_twin_sync_summary.csv",
    "outputs/tables/evidence_maturity_summary.csv",
]

FIGURE_BASES = [
    "Figure_3_Domain_Generalization_Gap",
    "Figure_4_Robustness_Ratio",
    "Figure_5_Ablation_Analysis",
    "Figure_6_Repeated_CV_Confidence_Intervals",
    "Figure_7_ECE_Comparison",
    "Figure_8_Missing_Modality_Robustness",
    "Figure_9_Privacy_Utility_Latency",
    "Figure_10_Evidence_Maturity_Dashboard",
]

REQUIRED_DOCS = [
    "docs/paper1/PAPER1_ARTIFACT_MAP.md",
    "REPRODUCIBILITY.md",
    "docs/paper1/LIMITATIONS_AND_EVIDENCE_BOUNDARY.md",
    "docs/paper1/MANUSCRIPT_PATCHES.md",
]

SEARCH_DIRS = ["docs", "outputs", "experiments", "scripts", "src"]
KEY_VALUES = ["64.28", "62.15", "2.3", "1e-5", "0.041", "47.3", "124.0", "67.0", "0.760", "35.0", "0.94", "0.27"]
NAME_ERROR_TOKEN = "#" + "NAME?"


def _print_result(label: str, passed: bool, detail: str = "") -> None:
    status = "PASS" if passed else "FAIL"
    suffix = f" - {detail}" if detail else ""
    print(f"[{status}] {label}{suffix}")


def _path_exists(project_root: Path, rel_path: str) -> bool:
    return (project_root / rel_path).exists()


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def _combined_required_text(project_root: Path) -> str:
    chunks: list[str] = []
    for rel_path in REQUIRED_CSVS + REQUIRED_TABLES + REQUIRED_DOCS:
        path = project_root / rel_path
        if path.exists() and path.is_file():
            chunks.append(_read_text(path))
    return "\n".join(chunks)


def _contains_name_error(project_root: Path) -> bool:
    for rel_dir in SEARCH_DIRS:
        root = project_root / rel_dir
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if "__pycache__" in path.parts:
                continue
            if any(part.startswith("~") for part in path.parts):
                continue
            if path.suffix.lower() in {".png", ".pdf", ".svg", ".jpg", ".jpeg", ".docx", ".zip", ".pkl", ".pyc"}:
                continue
            if NAME_ERROR_TOKEN in _read_text(path):
                _print_result("No spreadsheet name-error placeholders", False, str(path))
                return True
    return False


def validate(project_root: Path) -> bool:
    failures = 0

    for rel_path in REQUIRED_CSVS:
        passed = _path_exists(project_root, rel_path)
        _print_result(f"CSV exists: {rel_path}", passed)
        failures += 0 if passed else 1

    for rel_path in REQUIRED_TABLES:
        passed = _path_exists(project_root, rel_path)
        _print_result(f"Table exists: {rel_path}", passed)
        failures += 0 if passed else 1

    for figure_base in FIGURE_BASES:
        for suffix in [".png", ".pdf", ".svg"]:
            rel_path = f"outputs/figures/{figure_base}{suffix}"
            passed = _path_exists(project_root, rel_path)
            _print_result(f"Figure exists: {rel_path}", passed)
            failures += 0 if passed else 1

    for rel_path in REQUIRED_DOCS:
        passed = _path_exists(project_root, rel_path)
        _print_result(f"Documentation exists: {rel_path}", passed)
        failures += 0 if passed else 1

    no_name_error = not _contains_name_error(project_root)
    _print_result("No spreadsheet name-error placeholders in docs/outputs/experiments/scripts/src", no_name_error)
    failures += 0 if no_name_error else 1

    combined_text = _combined_required_text(project_root)
    for value in KEY_VALUES:
        passed = value in combined_text
        _print_result(f"Key value present: {value}", passed)
        failures += 0 if passed else 1

    if failures:
        print(f"Validation completed with {failures} failure(s).")
        return False
    print("Validation completed successfully. All required Paper 1 artifacts are present.")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate PAEMDT Paper 1 artifact alignment.")
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    project_root = Path(args.project_root).resolve()
    ok = validate(project_root)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
