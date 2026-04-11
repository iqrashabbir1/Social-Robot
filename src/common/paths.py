from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


CASE_STUDY_SLUGS = {
    "CS1": "cs1",
    "CS2": "cs2",
    "CS3": "cs3",
}


def normalize_case_study(case_study: str) -> str:
    cleaned = case_study.strip().upper()
    if cleaned in CASE_STUDY_SLUGS:
        return cleaned
    raise ValueError(f"Unsupported case study '{case_study}'. Expected one of: {', '.join(CASE_STUDY_SLUGS)}.")


@dataclass(frozen=True)
class Paper1Paths:
    project_root: Path
    docs_paper1: Path
    outputs_csv_paper1: Path
    outputs_csv_cs1: Path
    outputs_csv_cs2: Path
    outputs_csv_cs3: Path
    outputs_figures_paper1: Path
    outputs_figures_cs1: Path
    outputs_figures_cs2: Path
    outputs_figures_cs3: Path
    outputs_tables: Path
    outputs_logs: Path
    experiments_cs1: Path
    experiments_cs2: Path
    experiments_cs3: Path

    @classmethod
    def from_project_root(cls, project_root: Path) -> "Paper1Paths":
        root = project_root.resolve()
        return cls(
            project_root=root,
            docs_paper1=root / "docs" / "paper1",
            outputs_csv_paper1=root / "outputs" / "csv" / "paper1",
            outputs_csv_cs1=root / "outputs" / "csv" / "cs1",
            outputs_csv_cs2=root / "outputs" / "csv" / "cs2",
            outputs_csv_cs3=root / "outputs" / "csv" / "cs3",
            outputs_figures_paper1=root / "outputs" / "figures" / "paper1",
            outputs_figures_cs1=root / "outputs" / "figures" / "cs1",
            outputs_figures_cs2=root / "outputs" / "figures" / "cs2",
            outputs_figures_cs3=root / "outputs" / "figures" / "cs3",
            outputs_tables=root / "outputs" / "tables",
            outputs_logs=root / "outputs" / "logs",
            experiments_cs1=root / "experiments" / "cs1_digital_twin",
            experiments_cs2=root / "experiments" / "cs2_multimodal_sync",
            experiments_cs3=root / "experiments" / "cs3_emotion_benchmark",
        )

    def ensure(self) -> None:
        for directory in (
            self.docs_paper1,
            self.outputs_csv_paper1,
            self.outputs_csv_cs1,
            self.outputs_csv_cs2,
            self.outputs_csv_cs3,
            self.outputs_figures_paper1,
            self.outputs_figures_cs1,
            self.outputs_figures_cs2,
            self.outputs_figures_cs3,
            self.outputs_tables,
            self.outputs_logs,
            self.experiments_cs1,
            self.experiments_cs2,
            self.experiments_cs3,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def csv_case_dir(self, case_study: str) -> Path:
        slug = CASE_STUDY_SLUGS[normalize_case_study(case_study)]
        return self.project_root / "outputs" / "csv" / slug

    def figure_case_dir(self, case_study: str) -> Path:
        slug = CASE_STUDY_SLUGS[normalize_case_study(case_study)]
        return self.project_root / "outputs" / "figures" / slug

    def experiment_csv_dir(self, case_study: str, experiment_name: str) -> Path:
        return self.csv_case_dir(case_study) / experiment_name

    def experiment_figure_dir(self, case_study: str, experiment_name: str) -> Path:
        return self.figure_case_dir(case_study) / experiment_name

    def experiment_log_dir(self, case_study: str, experiment_name: str) -> Path:
        slug = CASE_STUDY_SLUGS[normalize_case_study(case_study)]
        return self.outputs_logs / slug / experiment_name

    def ensure_experiment_dirs(self, case_study: str, experiment_name: str) -> tuple[Path, Path, Path]:
        csv_dir = self.experiment_csv_dir(case_study, experiment_name)
        figure_dir = self.experiment_figure_dir(case_study, experiment_name)
        log_dir = self.experiment_log_dir(case_study, experiment_name)
        for directory in (csv_dir, figure_dir, log_dir):
            directory.mkdir(parents=True, exist_ok=True)
        return csv_dir, figure_dir, log_dir
