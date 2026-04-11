from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd

from src.evaluation.metrics_classification import compute_metrics, confusion_dataframe
from src.models.inference_benchmark import measure_inference_latency
from src.models.torch_runtime import encode_labels, require_torch, synchronize_if_needed


@dataclass
class TransformerFusionGpuResult:
    model: object
    metrics: dict[str, float | str]
    confusion: pd.DataFrame
    training_curve: pd.DataFrame
    predictions: list[str]


class TorchFusionTransformer:
    def __init__(self, video_dim: int, audio_dim: int, context_dim: int, hidden_dim: int, output_dim: int):
        torch = require_torch()
        self.video_proj = torch.nn.Linear(video_dim, hidden_dim)
        self.audio_proj = torch.nn.Linear(audio_dim, hidden_dim)
        self.context_proj = torch.nn.Linear(context_dim, hidden_dim)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim),
        )

    def parameters(self):
        modules = [self.video_proj, self.audio_proj, self.context_proj, self.classifier]
        for module in modules:
            yield from module.parameters()

    def to(self, device: str):
        self.video_proj.to(device)
        self.audio_proj.to(device)
        self.context_proj.to(device)
        self.classifier.to(device)
        return self

    def train(self):
        self.video_proj.train()
        self.audio_proj.train()
        self.context_proj.train()
        self.classifier.train()

    def eval(self):
        self.video_proj.eval()
        self.audio_proj.eval()
        self.context_proj.eval()
        self.classifier.eval()

    def state_dict(self):
        return {
            "video_proj": self.video_proj.state_dict(),
            "audio_proj": self.audio_proj.state_dict(),
            "context_proj": self.context_proj.state_dict(),
            "classifier": self.classifier.state_dict(),
        }

    def _forward_tokens(self, video, audio, context):
        torch = require_torch()
        video_token = self.video_proj(video)
        audio_token = self.audio_proj(audio)
        context_token = self.context_proj(context)
        tokens = torch.stack([video_token, audio_token, context_token], dim=1)
        query = tokens.mean(dim=1, keepdim=True)
        scores = torch.matmul(tokens, query.transpose(1, 2)).squeeze(-1) / (tokens.shape[-1] ** 0.5)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        fused = (tokens * weights).sum(dim=1)
        return fused, query.squeeze(1)

    def __call__(self, video, audio, context):
        fused, query = self._forward_tokens(video, audio, context)
        return self.classifier(require_torch().cat([fused, query], dim=1))


class TorchTransformerWrapper:
    def __init__(self, model: TorchFusionTransformer, labels: list[str], device: str, video_dim: int, audio_dim: int, context_dim: int) -> None:
        self.model = model
        self.labels = labels
        self.device = device
        self.video_dim = video_dim
        self.audio_dim = audio_dim
        self.context_dim = context_dim
        self.torch = require_torch()

    def predict(self, multimodal_features: np.ndarray) -> np.ndarray:
        fused_tensor = self.torch.tensor(multimodal_features, dtype=self.torch.float32, device=self.device)
        video = fused_tensor[:, : self.video_dim]
        audio = fused_tensor[:, self.video_dim : self.video_dim + self.audio_dim]
        context = fused_tensor[:, self.video_dim + self.audio_dim : self.video_dim + self.audio_dim + self.context_dim]
        self.model.eval()
        with self.torch.no_grad():
            logits = self.model(video, audio, context)
            synchronize_if_needed(self.device)
            pred_indices = logits.argmax(dim=1).detach().cpu().numpy()
        return np.asarray([self.labels[int(index)] for index in pred_indices])


def _prepare_modalities(bundle: dict[str, object], modalities: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    video = np.asarray(bundle["video"]).copy()
    audio = np.asarray(bundle["audio"]).copy()
    context = np.asarray(bundle["context"]).copy()
    if "video" not in modalities:
        video[:] = 0.0
    if "audio" not in modalities:
        audio[:] = 0.0
    if "context" not in modalities:
        context[:] = 0.0
    return video, audio, context


def train_and_evaluate_transformer_fusion_gpu(
    train_bundle: dict[str, object],
    test_bundle: dict[str, object],
    labels: list[str],
    device: str,
    modalities: tuple[str, ...] = ("video", "audio", "context"),
    seed: int = 42,
    epochs: int = 14,
    model_id: str = "B3",
    checkpoint_dir: Path | None = None,
    checkpoint_every: int = 0,
    hidden_dim: int = 16,
    alpha: float = 0.0001,
    batch_size: int = 128,
    shuffle: bool = True,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
) -> TransformerFusionGpuResult:
    torch = require_torch()
    torch.manual_seed(seed)
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    video_train_np, audio_train_np, context_train_np = _prepare_modalities(train_bundle, modalities)
    video_test_np, audio_test_np, context_test_np = _prepare_modalities(test_bundle, modalities)
    y_train = np.asarray(train_bundle["labels"])
    y_test = np.asarray(test_bundle["labels"])

    video_train = torch.tensor(video_train_np, dtype=torch.float32, device=device)
    audio_train = torch.tensor(audio_train_np, dtype=torch.float32, device=device)
    context_train = torch.tensor(context_train_np, dtype=torch.float32, device=device)
    video_test = torch.tensor(video_test_np, dtype=torch.float32, device=device)
    audio_test = torch.tensor(audio_test_np, dtype=torch.float32, device=device)
    context_test = torch.tensor(context_test_np, dtype=torch.float32, device=device)
    train_targets = torch.tensor(encode_labels(labels, y_train.tolist()), dtype=torch.long, device=device)
    _ = torch.tensor(encode_labels(labels, y_test.tolist()), dtype=torch.long, device=device)

    effective_batch_size = max(1, min(int(batch_size), len(video_train_np)))
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(video_train, audio_train, context_train, train_targets),
        batch_size=effective_batch_size,
        shuffle=shuffle,
    )

    model = TorchFusionTransformer(
        video_dim=video_train.shape[1],
        audio_dim=audio_train.shape[1],
        context_dim=context_train.shape[1],
        hidden_dim=hidden_dim,
        output_dim=len(labels),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=alpha)
    criterion = torch.nn.CrossEntropyLoss()

    curve_rows: list[dict[str, object]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        batch_losses: list[float] = []
        for batch_video, batch_audio, batch_context, batch_targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_video, batch_audio, batch_context)
            loss = criterion(logits, batch_targets)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu().item()))

        model.eval()
        with torch.no_grad():
            train_logits = model(video_train, audio_train, context_train)
            val_logits = model(video_test, audio_test, context_test)
            synchronize_if_needed(device)
            train_pred_indices = train_logits.argmax(dim=1).detach().cpu().numpy()
            val_pred_indices = val_logits.argmax(dim=1).detach().cpu().numpy()

        train_pred = [labels[int(index)] for index in train_pred_indices]
        val_pred = [labels[int(index)] for index in val_pred_indices]
        train_metrics = compute_metrics(y_train.tolist(), train_pred)
        val_metrics = compute_metrics(y_test.tolist(), val_pred)
        epoch_row = {
            "model_id": model_id,
            "epoch": epoch,
            "total_epochs": epochs,
            "train_accuracy": round(train_metrics.accuracy, 4),
            "val_accuracy": round(val_metrics.accuracy, 4),
            "train_macro_f1": round(train_metrics.macro_f1, 4),
            "val_macro_f1": round(val_metrics.macro_f1, 4),
            "loss": round(float(sum(batch_losses) / max(len(batch_losses), 1)), 6),
            "runtime_backend": "gpu",
            "device": device,
            "batch_size": effective_batch_size,
            "evidence_level": "synthetic_placeholder_benchmark",
        }
        curve_rows.append(epoch_row)
        if progress_callback is not None:
            progress_callback(epoch_row)
        if checkpoint_dir and checkpoint_every > 0 and epoch % checkpoint_every == 0:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "modalities": modalities,
                    "labels": labels,
                    "hidden_dim": hidden_dim,
                    "state_dict": model.state_dict(),
                },
                checkpoint_dir / f"{model_id.lower()}_epoch_{epoch:04d}.pt",
            )
            metadata = {"hidden_dim": hidden_dim, "alpha": alpha, "device": device}
            joblib.dump(metadata, checkpoint_dir / f"{model_id.lower()}_metadata_epoch_{epoch:04d}.joblib")
            pd.DataFrame(curve_rows).to_csv(checkpoint_dir / f"{model_id.lower()}_curve_until_{epoch:04d}.csv", index=False)

    test_fused = np.concatenate([video_test_np, audio_test_np, context_test_np], axis=1)
    wrapper = TorchTransformerWrapper(
        model,
        labels,
        device,
        video_dim=video_test_np.shape[1],
        audio_dim=audio_test_np.shape[1],
        context_dim=context_test_np.shape[1],
    )
    y_pred = wrapper.predict(test_fused).tolist()
    scores = compute_metrics(y_test.tolist(), y_pred)
    metrics = {
        "accuracy": round(scores.accuracy, 4),
        "macro_f1": round(scores.macro_f1, 4),
        "weighted_f1": round(scores.weighted_f1, 4),
        "uar": round(scores.unweighted_recall, 4),
        "inference_latency_ms": round(measure_inference_latency(wrapper, test_fused), 4),
        "runtime_backend": "gpu",
        "device": device,
    }
    confusion = confusion_dataframe(y_test.tolist(), y_pred, labels)
    return TransformerFusionGpuResult(
        model=wrapper,
        metrics=metrics,
        confusion=confusion,
        training_curve=pd.DataFrame(curve_rows),
        predictions=y_pred,
    )
