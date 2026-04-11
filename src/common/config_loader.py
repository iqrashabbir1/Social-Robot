from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.common.io_utils import read_yaml
from src.common.paths import Paper1Paths, normalize_case_study


REQUIRED_CONFIG_KEYS = (
    "experiment_name",
    "case_study",
    "seed",
    "inputs",
    "outputs",
    "preprocessing",
    "evaluation",
    "plot",
)


@dataclass(frozen=True)
class ExperimentContext:
    project_root: Path
    config_path: Path
    config: dict[str, Any]
    case_study: str
    experiment_name: str
    csv_dir: Path
    figure_dir: Path
    log_dir: Path
    log_path: Path
    config_snapshot_path: Path
    summary_json_path: Path
    metrics_csv_path: Path


def validate_config_schema(config: dict[str, Any], config_path: Path | None = None) -> None:
    missing = [key for key in REQUIRED_CONFIG_KEYS if key not in config]
    if missing:
        location = str(config_path) if config_path is not None else "<in-memory config>"
        raise ValueError(f"Missing required config keys in {location}: {', '.join(missing)}")
    for key in ("inputs", "outputs", "preprocessing", "evaluation", "plot"):
        if not isinstance(config.get(key), dict):
            raise ValueError(f"Config key '{key}' must be a mapping.")


def load_experiment_config(config_path: Path) -> dict[str, Any]:
    config = read_yaml(config_path)
    validate_config_schema(config, config_path)
    config["case_study"] = normalize_case_study(str(config["case_study"]))
    config["experiment_name"] = str(config["experiment_name"]).strip()
    if not config["experiment_name"]:
        raise ValueError(f"Config file {config_path} has an empty experiment_name.")
    return config


def build_experiment_context(project_root: Path, config_path: Path) -> ExperimentContext:
    resolved_root = project_root.resolve()
    resolved_config = config_path.resolve()
    config = load_experiment_config(resolved_config)
    paper1_paths = Paper1Paths.from_project_root(resolved_root)
    paper1_paths.ensure()

    case_study = config["case_study"]
    experiment_name = config["experiment_name"]
    csv_dir, figure_dir, log_dir = paper1_paths.ensure_experiment_dirs(case_study, experiment_name)

    return ExperimentContext(
        project_root=resolved_root,
        config_path=resolved_config,
        config=config,
        case_study=case_study,
        experiment_name=experiment_name,
        csv_dir=csv_dir,
        figure_dir=figure_dir,
        log_dir=log_dir,
        log_path=log_dir / "run.log",
        config_snapshot_path=csv_dir / "config_snapshot.yaml",
        summary_json_path=csv_dir / "summary.json",
        metrics_csv_path=csv_dir / "metrics.csv",
    )
