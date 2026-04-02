from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paper1Paths:
    project_root: Path
    docs_paper1: Path
    outputs_csv_cs1: Path
    outputs_csv_cs2: Path
    outputs_csv_cs3: Path
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
            outputs_csv_cs1=root / "outputs" / "csv" / "cs1",
            outputs_csv_cs2=root / "outputs" / "csv" / "cs2",
            outputs_csv_cs3=root / "outputs" / "csv" / "cs3",
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
            self.outputs_csv_cs1,
            self.outputs_csv_cs2,
            self.outputs_csv_cs3,
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
