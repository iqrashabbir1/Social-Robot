from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd
import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from src.common.config_loader import build_experiment_context
from src.common.io_utils import write_dataframe, write_json, write_yaml
from src.common.logging_utils import get_logger
from src.common.reproducibility import set_global_seed
from src.evaluation.metrics_classification import compute_metrics, confusion_dataframe
from src.models.deep.train_deep_fusion_gpu import train_and_evaluate_deep_fusion_gpu
from src.models.inference_benchmark import LABELS, assemble_feature_matrix, build_dataset_split, measure_inference_latency
from src.models.torch_runtime import resolve_torch_runtime


@dataclass
class DeepFusionResult:
    model: object
    scaler: StandardScaler
    metrics: dict[str, float]
    confusion: pd.DataFrame
    training_curve: pd.DataFrame
    predictions: list[str]


class DeepFusionWrapper:
    def __init__(self, scaler: StandardScaler, classifier: MLPClassifier) -> None:
        self.scaler = scaler
        self.classifier = classifier

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.classifier.predict(self.scaler.transform(x))


def train_and_evaluate_deep_fusion(
    train_bundle: dict[str, object],
    test_bundle: dict[str, object],
    modalities: tuple[str, ...],
    labels: list[str],
    seed: int = 42,
    epochs: int = 14,
    model_id: str = "B2",
    checkpoint_dir: Path | None = None,
    checkpoint_every: int = 0,
    hidden_layers: tuple[int, ...] = (96, 48),
    learning_rate_init: float = 0.001,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
) -> DeepFusionResult:
    x_train = assemble_feature_matrix(train_bundle, modalities)
    x_test = assemble_feature_matrix(test_bundle, modalities)
    y_train = np.asarray(train_bundle["labels"])
    y_test = np.asarray(test_bundle["labels"])

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    classifier = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation="relu",
        solver="adam",
        random_state=seed,
        learning_rate_init=learning_rate_init,
        max_iter=1,
        warm_start=True,
    )

    curve_rows: list[dict[str, object]] = []
    for epoch in range(1, epochs + 1):
        if epoch == 1:
            classifier.partial_fit(x_train_scaled, y_train, classes=np.asarray(labels))
        else:
            classifier.partial_fit(x_train_scaled, y_train)
        train_pred = classifier.predict(x_train_scaled)
        val_pred = classifier.predict(x_test_scaled)
        train_metrics = compute_metrics(y_train.tolist(), train_pred.tolist())
        val_metrics = compute_metrics(y_test.tolist(), val_pred.tolist())
        epoch_row = {
            "model_id": model_id,
            "epoch": epoch,
            "total_epochs": epochs,
            "train_accuracy": round(train_metrics.accuracy, 4),
            "val_accuracy": round(val_metrics.accuracy, 4),
            "train_macro_f1": round(train_metrics.macro_f1, 4),
            "val_macro_f1": round(val_metrics.macro_f1, 4),
            "loss": round(float(classifier.loss_), 6),
            "evidence_level": "synthetic_placeholder_benchmark",
        }
        curve_rows.append(epoch_row)
        if progress_callback is not None:
            progress_callback(epoch_row)
        if checkpoint_dir and checkpoint_every > 0 and epoch % checkpoint_every == 0:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_payload = {
                "epoch": epoch,
                "modalities": modalities,
                "labels": labels,
                "scaler": scaler,
                "classifier": classifier,
            }
            joblib.dump(checkpoint_payload, checkpoint_dir / f"{model_id.lower()}_epoch_{epoch:04d}.joblib")
            pd.DataFrame(curve_rows).to_csv(checkpoint_dir / f"{model_id.lower()}_curve_until_{epoch:04d}.csv", index=False)

    y_pred = classifier.predict(x_test_scaled).tolist()
    scores = compute_metrics(y_test.tolist(), y_pred)
    wrapper = DeepFusionWrapper(scaler, classifier)
    metrics = {
        "accuracy": round(scores.accuracy, 4),
        "macro_f1": round(scores.macro_f1, 4),
        "weighted_f1": round(scores.weighted_f1, 4),
        "uar": round(scores.unweighted_recall, 4),
        "inference_latency_ms": round(measure_inference_latency(wrapper, x_test), 4),
    }
    confusion = confusion_dataframe(y_test.tolist(), y_pred, labels)
    return DeepFusionResult(
        model=wrapper,
        scaler=scaler,
        metrics=metrics,
        confusion=confusion,
        training_curve=pd.DataFrame(curve_rows),
        predictions=y_pred,
    )


def run_deep_experiment(project_root: Path, config_path: Path) -> dict[str, str]:
    context = build_experiment_context(project_root, config_path)
    config = context.config
    if context.case_study != "CS3":
        raise ValueError(f"Deep trainer only supports CS3 configs, received {context.case_study}.")

    seed = int(config["seed"])
    set_global_seed(seed)
    logger = get_logger(f"paper1.cs3.{context.experiment_name}", context.log_path)
    logger.info("Running deep-fusion CS3 experiment '%s'.", context.experiment_name)

    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})
    modalities = tuple(config.get("modalities", {}).get("selected", ["video", "audio"]))
    split = build_dataset_split(context.project_root, config.get("dataset", {}), seed)
    runtime = resolve_torch_runtime(
        str(training_cfg.get("runtime_backend", "cpu")),
        str(training_cfg.get("torch_device", "auto")),
    )
    checkpoint_every = int(training_cfg.get("checkpoint_every", 0))
    checkpoint_dir = context.log_dir / "checkpoints" if checkpoint_every > 0 else None

    common_kwargs = {
        "train_bundle": split["train"],
        "test_bundle": split["test"],
        "modalities": modalities,
        "labels": LABELS,
        "seed": seed,
        "epochs": int(training_cfg.get("epochs", 40)),
        "model_id": str(model_cfg.get("model_id", context.experiment_name)),
        "checkpoint_dir": checkpoint_dir,
        "checkpoint_every": checkpoint_every,
        "hidden_layers": tuple(model_cfg.get("hyperparameters", {}).get("hidden_layers", [96, 48])),
        "learning_rate_init": float(model_cfg.get("hyperparameters", {}).get("learning_rate_init", 0.001)),
    }
    if runtime.active_backend == "gpu":
        result = train_and_evaluate_deep_fusion_gpu(
            **common_kwargs,
            device=runtime.device,
            batch_size=int(training_cfg.get("batch_size", 128)),
        )
    else:
        result = train_and_evaluate_deep_fusion(**common_kwargs)

    metrics_row = {
        "experiment_name": context.experiment_name,
        "case_study": context.case_study,
        "model_family": str(model_cfg.get("family", "deep")),
        "algorithm_name": str(model_cfg.get("name", "late_fusion_mlp")),
        "modality_setting": "_".join(modalities),
        "seed": seed,
        "accuracy": result.metrics["accuracy"],
        "macro_f1": result.metrics["macro_f1"],
        "weighted_f1": result.metrics["weighted_f1"],
        "uar": result.metrics["uar"],
        "inference_latency_ms": result.metrics["inference_latency_ms"],
        "runtime_backend": runtime.active_backend,
        "device": runtime.device,
        "epochs": int(training_cfg.get("epochs", 40)),
        "data_source_type": str(config.get("evaluation", {}).get("data_source_type", "synthetic")),
        "runtime_type": str(config.get("evaluation", {}).get("runtime_type", "software_only")),
        "model_status": str(config.get("evaluation", {}).get("model_status", "fully_runnable")),
        "evidence_level": str(config.get("evaluation", {}).get("evidence_level", "benchmark_preliminary")),
        "data_regime": str(config.get("dataset", {}).get("name", "synthetic_aligned_multimodal_windows")),
    }

    artifact_path = context.log_dir / f"{context.experiment_name}.joblib"
    joblib.dump({"model": result.model, "modalities": modalities, "labels": LABELS}, artifact_path)

    write_yaml(context.config_snapshot_path, config)
    write_dataframe(context.metrics_csv_path, pd.DataFrame([metrics_row]))
    write_dataframe(context.csv_dir / "model_performance_summary.csv", pd.DataFrame([metrics_row]))
    write_dataframe(context.csv_dir / "confusion_matrix.csv", result.confusion)
    write_dataframe(context.csv_dir / "training_curves.csv", result.training_curve)
    write_dataframe(context.csv_dir / "predictions.csv", pd.DataFrame({"prediction": result.predictions}))

    summary = {
        "experiment_name": context.experiment_name,
        "case_study": context.case_study,
        "config_path": str(context.config_path),
        "summary_json": str(context.summary_json_path),
        "metrics_csv": str(context.metrics_csv_path),
        "training_curves_csv": str(context.csv_dir / "training_curves.csv"),
        "confusion_matrix_csv": str(context.csv_dir / "confusion_matrix.csv"),
        "artifact_path": str(artifact_path),
        "log_path": str(context.log_path),
        "runtime_backend": runtime.active_backend,
        "device": runtime.device,
    }
    write_json(context.summary_json_path, summary)
    logger.info("Finished deep-fusion CS3 experiment '%s'.", context.experiment_name)
    return {key: str(value) for key, value in summary.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one deep-fusion CS3 experiment from a single config.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_deep_experiment(Path(args.project_root).resolve(), Path(args.config).resolve())


if __name__ == "__main__":
    main()
