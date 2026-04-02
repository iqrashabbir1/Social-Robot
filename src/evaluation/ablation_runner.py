from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.common.io_utils import read_yaml, write_dataframe, write_json
from src.common.logging_utils import get_logger
from src.common.paths import Paper1Paths
from src.common.reproducibility import set_global_seed
from src.evaluation.metrics_classification import compute_metrics, confusion_dataframe
from src.features.video_features import load_visual_baseline_log
from src.models.classical.train_classical import train_and_evaluate_classical
from src.models.deep.train_deep_fusion import train_and_evaluate_deep_fusion
from src.models.inference_benchmark import (
    LABELS,
    apply_missing_modality,
    build_synthetic_multimodal_dataset,
    split_feature_bundle,
)
from src.models.transformer.train_transformer_fusion import train_and_evaluate_transformer_fusion


ABLATIONS = {
    "video_only": ("video",),
    "audio_only": ("audio",),
    "context_only": ("context",),
    "video_audio": ("video", "audio"),
    "video_audio_context": ("video", "audio", "context"),
}


def _evaluate_baseline_zero(project_root: Path) -> tuple[dict[str, object], pd.DataFrame]:
    df = load_visual_baseline_log(project_root)
    y_true = df["true_emotion"].astype(str).str.lower().tolist()
    y_pred = df["emotion"].astype(str).str.lower().tolist()
    scores = compute_metrics(y_true, y_pred)
    metrics = {
        "model_id": "B0",
        "model_family": "existing_repo_baseline",
        "modality_setting": "video_only",
        "accuracy": round(scores.accuracy, 4),
        "macro_f1": round(scores.macro_f1, 4),
        "weighted_f1": round(scores.weighted_f1, 4),
        "uar": round(scores.unweighted_recall, 4),
        "inference_latency_ms": None,
        "robustness_missing_modality_macro_f1": None,
        "evidence_level": "implemented_real_baseline",
        "data_regime": "real_visual_log",
        "notes": "Preserved DeepFace-style visual baseline evaluated from tests/emotion_log_labeled.csv.",
    }
    confusion = confusion_dataframe(y_true, y_pred, LABELS)
    return metrics, confusion


def _run_ablation_set(
    train_bundle: dict[str, object],
    test_bundle: dict[str, object],
    seed: int,
) -> tuple[pd.DataFrame, list[dict[str, object]], dict[str, pd.DataFrame], pd.DataFrame]:
    performance_rows: list[dict[str, object]] = []
    ablation_rows: list[dict[str, object]] = []
    confusion_outputs: dict[str, pd.DataFrame] = {}
    training_curves: list[pd.DataFrame] = []

    classical = train_and_evaluate_classical(train_bundle, test_bundle, ABLATIONS["video_audio_context"], LABELS, model_name="svm", seed=seed)
    deep = train_and_evaluate_deep_fusion(train_bundle, test_bundle, ABLATIONS["video_audio_context"], LABELS, seed=seed)
    transformer = train_and_evaluate_transformer_fusion(train_bundle, test_bundle, LABELS, seed=seed)

    missing_test_bundle = apply_missing_modality(test_bundle, ("video", "audio", "context"), 0.2, seed + 101)
    classical_missing = train_and_evaluate_classical(train_bundle, missing_test_bundle, ABLATIONS["video_audio_context"], LABELS, model_name="svm", seed=seed)
    deep_missing = train_and_evaluate_deep_fusion(train_bundle, missing_test_bundle, ABLATIONS["video_audio_context"], LABELS, seed=seed)
    transformer_missing = train_and_evaluate_transformer_fusion(train_bundle, missing_test_bundle, LABELS, seed=seed)

    performance_rows.extend(
        [
            {
                "model_id": "B1",
                "model_family": "classical_svm",
                "modality_setting": "video_audio_context",
                **classical.metrics,
                "robustness_missing_modality_macro_f1": classical_missing.metrics["macro_f1"],
                "evidence_level": "synthetic_placeholder_benchmark",
                "data_regime": "synthetic_aligned_multimodal_windows",
                "notes": "Handcrafted multimodal features with SVM.",
            },
            {
                "model_id": "B2",
                "model_family": "deep_late_fusion_mlp",
                "modality_setting": "video_audio_context",
                **deep.metrics,
                "robustness_missing_modality_macro_f1": deep_missing.metrics["macro_f1"],
                "evidence_level": "synthetic_placeholder_benchmark",
                "data_regime": "synthetic_aligned_multimodal_windows",
                "notes": "Late-fusion MLP over concatenated video, audio, and context features.",
            },
            {
                "model_id": "B3",
                "model_family": "lightweight_fusion_transformer",
                "modality_setting": "video_audio_context",
                **transformer.metrics,
                "robustness_missing_modality_macro_f1": transformer_missing.metrics["macro_f1"],
                "evidence_level": "synthetic_placeholder_benchmark",
                "data_regime": "synthetic_aligned_multimodal_windows",
                "notes": "Cross-modal attention surrogate with linear online classifier.",
            },
        ]
    )

    confusion_outputs["confusion_matrix_deep"] = deep.confusion
    confusion_outputs["confusion_matrix_transformer"] = transformer.confusion
    training_curves.extend([deep.training_curve, transformer.training_curve])

    for ablation_name, modalities in ABLATIONS.items():
        for condition_name, condition_bundle in (
            ("nominal", test_bundle),
            ("missing_modality_20pct", apply_missing_modality(test_bundle, modalities, 0.2, seed + len(modalities))),
        ):
            classical_ablation = train_and_evaluate_classical(train_bundle, condition_bundle, modalities, LABELS, model_name="svm", seed=seed)
            deep_ablation = train_and_evaluate_deep_fusion(train_bundle, condition_bundle, modalities, LABELS, seed=seed)
            transformer_ablation = train_and_evaluate_transformer_fusion(train_bundle, condition_bundle, LABELS, modalities=modalities, seed=seed)
            for model_id, result in (
                ("B1", classical_ablation.metrics),
                ("B2", deep_ablation.metrics),
                ("B3", transformer_ablation.metrics),
            ):
                ablation_rows.append(
                    {
                        "model_id": model_id,
                        "ablation_name": ablation_name,
                        "condition": condition_name,
                        "accuracy": result["accuracy"],
                        "macro_f1": result["macro_f1"],
                        "weighted_f1": result["weighted_f1"],
                        "uar": result["uar"],
                        "inference_latency_ms": result["inference_latency_ms"],
                        "evidence_level": "synthetic_placeholder_benchmark",
                    }
                )

    return pd.DataFrame(performance_rows), ablation_rows, confusion_outputs, pd.concat(training_curves, ignore_index=True)


def run_cs3(project_root: Path, config_path: Path) -> dict[str, str]:
    config = read_yaml(config_path)
    seed = int(config.get("seed", 42))
    set_global_seed(seed)
    paper1_paths = Paper1Paths.from_project_root(project_root)
    paper1_paths.ensure()
    logger = get_logger("paper1.cs3", paper1_paths.outputs_logs / "paper1_cs3.log")
    logger.info("Running CS3 emotion recognition benchmark.")

    baseline_metrics, baseline_confusion = _evaluate_baseline_zero(project_root)
    bundle = build_synthetic_multimodal_dataset(
        project_root=project_root,
        seed=seed,
        n_samples=int(config.get("dataset", {}).get("n_samples", 480)),
    )
    split = split_feature_bundle(bundle, test_size=float(config.get("dataset", {}).get("test_size", 0.25)), seed=seed)
    performance_df, ablation_rows, confusion_outputs, training_curve_df = _run_ablation_set(split["train"], split["test"], seed)
    performance_df = pd.concat([pd.DataFrame([baseline_metrics]), performance_df], ignore_index=True)
    ablation_df = pd.DataFrame(ablation_rows)

    write_dataframe(paper1_paths.outputs_csv_cs3 / "model_performance_summary.csv", performance_df)
    write_dataframe(paper1_paths.outputs_csv_cs3 / "confusion_matrix_baseline.csv", baseline_confusion)
    write_dataframe(paper1_paths.outputs_csv_cs3 / "confusion_matrix_deep.csv", confusion_outputs["confusion_matrix_deep"])
    write_dataframe(paper1_paths.outputs_csv_cs3 / "confusion_matrix_transformer.csv", confusion_outputs["confusion_matrix_transformer"])
    write_dataframe(paper1_paths.outputs_csv_cs3 / "ablation_results.csv", ablation_df)
    write_dataframe(paper1_paths.outputs_csv_cs3 / "training_curves.csv", training_curve_df)

    summary = {
        "case_study": "CS3",
        "config_path": str(config_path),
        "model_performance_summary": str(paper1_paths.outputs_csv_cs3 / "model_performance_summary.csv"),
        "ablation_results": str(paper1_paths.outputs_csv_cs3 / "ablation_results.csv"),
        "training_curves": str(paper1_paths.outputs_csv_cs3 / "training_curves.csv"),
    }
    write_json(paper1_paths.outputs_logs / "paper1_cs3_summary.json", summary)
    logger.info("CS3 outputs written to %s", paper1_paths.outputs_csv_cs3)
    return {key: str(value) for key, value in summary.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CS3 multimodal emotion recognition benchmark.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--config", default="configs/cs3/default.yaml")
    args = parser.parse_args()
    run_cs3(Path(args.project_root).resolve(), Path(args.config).resolve())


if __name__ == "__main__":
    main()
