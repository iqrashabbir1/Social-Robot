from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from src.evaluation.metrics_classification import compute_metrics, confusion_dataframe
from src.models.inference_benchmark import assemble_feature_matrix, measure_inference_latency


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
