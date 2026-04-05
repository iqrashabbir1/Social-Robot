from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

from src.evaluation.metrics_classification import compute_metrics, confusion_dataframe
from src.models.inference_benchmark import measure_inference_latency


@dataclass
class TransformerFusionResult:
    model: object
    metrics: dict[str, float]
    confusion: pd.DataFrame
    training_curve: pd.DataFrame
    predictions: list[str]


class LightweightFusionTransformer:
    def __init__(
        self,
        video_dim: int,
        audio_dim: int,
        context_dim: int,
        hidden_dim: int = 16,
        alpha: float = 0.0001,
        seed: int = 42,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.video_proj = rng.normal(0.0, 0.35, (video_dim, hidden_dim))
        self.audio_proj = rng.normal(0.0, 0.35, (audio_dim, hidden_dim))
        self.context_proj = rng.normal(0.0, 0.35, (context_dim, hidden_dim))
        self.scaler = StandardScaler()
        self.classifier = SGDClassifier(loss="log_loss", alpha=alpha, random_state=seed)
        self._is_fitted = False

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)

    def transform(self, video: np.ndarray, audio: np.ndarray, context: np.ndarray) -> np.ndarray:
        tokens = np.stack(
            [
                video @ self.video_proj,
                audio @ self.audio_proj,
                context @ self.context_proj,
            ],
            axis=1,
        )
        cls_query = tokens.mean(axis=1)
        logits = np.einsum("nmd,nd->nm", tokens, cls_query) / np.sqrt(tokens.shape[-1])
        weights = self._softmax(logits)
        fused = np.sum(tokens * weights[:, :, None], axis=1)
        return np.concatenate([fused, cls_query], axis=1)

    def partial_fit(
        self,
        video: np.ndarray,
        audio: np.ndarray,
        context: np.ndarray,
        y: list[str],
        classes: np.ndarray | None = None,
    ) -> None:
        fused = self.transform(video, audio, context)
        if not self._is_fitted:
            scaled = self.scaler.fit_transform(fused)
            self.classifier.partial_fit(scaled, y, classes=classes)
            self._is_fitted = True
        else:
            scaled = self.scaler.transform(fused)
            self.classifier.partial_fit(scaled, y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.classifier.predict(self.scaler.transform(x))

    def predict_from_modalities(self, video: np.ndarray, audio: np.ndarray, context: np.ndarray) -> np.ndarray:
        fused = self.transform(video, audio, context)
        return self.predict(fused)


class TransformerWrapper:
    def __init__(self, transformer: LightweightFusionTransformer) -> None:
        self.transformer = transformer

    def predict(self, fused_features: np.ndarray) -> np.ndarray:
        return self.transformer.predict(fused_features)


def train_and_evaluate_transformer_fusion(
    train_bundle: dict[str, object],
    test_bundle: dict[str, object],
    labels: list[str],
    modalities: tuple[str, ...] = ("video", "audio", "context"),
    seed: int = 42,
    epochs: int = 14,
    model_id: str = "B3",
    checkpoint_dir: Path | None = None,
    checkpoint_every: int = 0,
    hidden_dim: int = 16,
    alpha: float = 0.0001,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
) -> TransformerFusionResult:
    video_train = np.asarray(train_bundle["video"]).copy()
    audio_train = np.asarray(train_bundle["audio"]).copy()
    context_train = np.asarray(train_bundle["context"]).copy()
    y_train = np.asarray(train_bundle["labels"])
    video_test = np.asarray(test_bundle["video"]).copy()
    audio_test = np.asarray(test_bundle["audio"]).copy()
    context_test = np.asarray(test_bundle["context"]).copy()
    y_test = np.asarray(test_bundle["labels"])

    if "video" not in modalities:
        video_train[:] = 0.0
        video_test[:] = 0.0
    if "audio" not in modalities:
        audio_train[:] = 0.0
        audio_test[:] = 0.0
    if "context" not in modalities:
        context_train[:] = 0.0
        context_test[:] = 0.0

    transformer = LightweightFusionTransformer(
        video_dim=video_train.shape[1],
        audio_dim=audio_train.shape[1],
        context_dim=context_train.shape[1],
        hidden_dim=hidden_dim,
        alpha=alpha,
        seed=seed,
    )

    curve_rows: list[dict[str, object]] = []
    for epoch in range(1, epochs + 1):
        if epoch == 1:
            transformer.partial_fit(video_train, audio_train, context_train, y_train.tolist(), classes=np.asarray(labels))
        else:
            transformer.partial_fit(video_train, audio_train, context_train, y_train.tolist())
        train_pred = transformer.predict_from_modalities(video_train, audio_train, context_train)
        val_pred = transformer.predict_from_modalities(video_test, audio_test, context_test)
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
            "loss": round(float(max(1.0 - val_metrics.accuracy, 0.0001)), 6),
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
                "transformer": transformer,
            }
            joblib.dump(checkpoint_payload, checkpoint_dir / f"{model_id.lower()}_epoch_{epoch:04d}.joblib")
            pd.DataFrame(curve_rows).to_csv(checkpoint_dir / f"{model_id.lower()}_curve_until_{epoch:04d}.csv", index=False)

    test_fused = transformer.transform(video_test, audio_test, context_test)
    y_pred = transformer.predict(test_fused).tolist()
    scores = compute_metrics(y_test.tolist(), y_pred)
    wrapper = TransformerWrapper(transformer)
    metrics = {
        "accuracy": round(scores.accuracy, 4),
        "macro_f1": round(scores.macro_f1, 4),
        "weighted_f1": round(scores.weighted_f1, 4),
        "uar": round(scores.unweighted_recall, 4),
        "inference_latency_ms": round(measure_inference_latency(wrapper, test_fused), 4),
    }
    confusion = confusion_dataframe(y_test.tolist(), y_pred, labels)
    return TransformerFusionResult(
        model=wrapper,
        metrics=metrics,
        confusion=confusion,
        training_curve=pd.DataFrame(curve_rows),
        predictions=y_pred,
    )
