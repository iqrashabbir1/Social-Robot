from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from src.common.io_utils import write_yaml
from src.evaluation.benchmark_runner import run_cs3_benchmark


def _write_single_config(target: Path, experiment_name: str, model_name: str, modalities: list[str]) -> None:
    config = {
        "experiment_name": experiment_name,
        "case_study": "CS3",
        "seed": 42,
        "inputs": {"baseline_visual_log": "tests/emotion_log_labeled.csv"},
        "outputs": {"csv_dir": f"outputs/csv/cs3/{experiment_name}", "figure_dir": f"outputs/figures/cs3/{experiment_name}"},
        "modalities": {"selected": modalities},
        "preprocessing": {"normalization": "standard_scaler"},
        "dataset": {"name": "synthetic_aligned_multimodal_windows", "n_samples": 64, "test_size": 0.25},
        "model": {"family": "classical", "name": model_name, "hyperparameters": {}},
        "training": {"runtime_backend": "cpu", "epochs": 0},
        "evaluation": {"metrics": ["accuracy", "macro_f1"], "evidence_level": "synthetic_placeholder_benchmark"},
        "plot": {"save_png": True, "save_svg": True},
    }
    write_yaml(target, config)


def test_benchmark_runner_executes_small_config_list(tmp_path: Path) -> None:
    project_root = tmp_path
    (project_root / "tests").mkdir(parents=True)
    shutil.copy(Path(__file__).resolve().parent / "emotion_log_labeled.csv", project_root / "tests" / "emotion_log_labeled.csv")

    config_dir = project_root / "configs" / "cs3"
    config_dir.mkdir(parents=True)
    _write_single_config(config_dir / "svm_video.yaml", "svm_video", "svm", ["video"])
    _write_single_config(config_dir / "rf_video_audio.yaml", "rf_video_audio", "random_forest", ["video", "audio"])

    benchmark_config = {
        "benchmark_name": "unit_benchmark",
        "case_study": "CS3",
        "config_files": ["configs/cs3/svm_video.yaml", "configs/cs3/rf_video_audio.yaml"],
        "outputs": {"master_summary_csv": "outputs/tables/cs3_master_model_summary.csv"},
        "options": {"continue_on_error": False},
    }
    benchmark_path = config_dir / "benchmark_all.yaml"
    write_yaml(benchmark_path, benchmark_config)

    result = run_cs3_benchmark(project_root, benchmark_path)
    summary_df = pd.read_csv(result["master_summary_csv"])

    assert len(summary_df) == 2
    assert {"experiment_name", "algorithm_name", "macro_f1"}.issubset(summary_df.columns)
