from __future__ import annotations

from pathlib import Path

from src.common.config_loader import build_experiment_context, load_experiment_config, validate_config_schema
from src.common.io_utils import write_yaml


def _sample_config() -> dict[str, object]:
    return {
        "experiment_name": "unit_test_config",
        "case_study": "CS3",
        "seed": 42,
        "inputs": {"baseline_visual_log": "tests/emotion_log_labeled.csv"},
        "outputs": {"csv_dir": "outputs/csv/cs3/unit_test_config", "figure_dir": "outputs/figures/cs3/unit_test_config"},
        "modalities": {"selected": ["video", "audio"]},
        "preprocessing": {"normalization": "standard_scaler"},
        "dataset": {"name": "synthetic_aligned_multimodal_windows", "n_samples": 64, "test_size": 0.25},
        "model": {"family": "classical", "name": "svm", "hyperparameters": {}},
        "training": {"runtime_backend": "cpu", "epochs": 0},
        "evaluation": {"metrics": ["accuracy"], "evidence_level": "synthetic_placeholder_benchmark"},
        "plot": {"save_png": True, "save_svg": True},
    }


def test_config_loader_builds_experiment_context(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    write_yaml(config_path, _sample_config())

    config = load_experiment_config(config_path)
    validate_config_schema(config, config_path)
    context = build_experiment_context(tmp_path, config_path)

    assert context.case_study == "CS3"
    assert context.experiment_name == "unit_test_config"
    assert context.csv_dir.exists()
    assert context.figure_dir.exists()
    assert context.log_dir.exists()
