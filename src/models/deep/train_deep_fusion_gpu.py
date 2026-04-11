from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.evaluation.metrics_classification import compute_metrics, confusion_dataframe
from src.models.inference_benchmark import assemble_feature_matrix, measure_inference_latency
from src.models.torch_runtime import encode_labels, require_torch, synchronize_if_needed


@dataclass
class DeepFusionGpuResult:
    model: object
    scaler: StandardScaler
    metrics: dict[str, float | str]
    confusion: pd.DataFrame
    training_curve: pd.DataFrame
    predictions: list[str]


class TorchDeepFusionWrapper:
    def __init__(self, scaler: StandardScaler, network, labels: list[str], device: str) -> None:
        self.scaler = scaler
        self.network = network
        self.labels = labels
        self.device = device
        self.torch = require_torch()

    def predict(self, x: np.ndarray) -> np.ndarray:
        x_scaled = self.scaler.transform(x)
        features = self.torch.tensor(x_scaled, dtype=self.torch.float32, device=self.device)
        self.network.eval()
        with self.torch.no_grad():
            logits = self.network(features)
            synchronize_if_needed(self.device)
            predictions = logits.argmax(dim=1).detach().cpu().numpy()
        return np.asarray([self.labels[int(index)] for index in predictions])


def _build_network(input_dim: int, hidden_layers: tuple[int, ...], output_dim: int):
    torch = require_torch()
    layers = []
    current_dim = input_dim
    for hidden_dim in hidden_layers:
        layers.append(torch.nn.Linear(current_dim, hidden_dim))
        layers.append(torch.nn.ReLU())
        current_dim = hidden_dim
    layers.append(torch.nn.Linear(current_dim, output_dim))
    return torch.nn.Sequential(*layers)


def train_and_evaluate_deep_fusion_gpu(
    train_bundle: dict[str, object],
    test_bundle: dict[str, object],
    modalities: tuple[str, ...],
    labels: list[str],
    device: str,
    seed: int = 42,
    epochs: int = 14,
    model_id: str = "B2",
    checkpoint_dir: Path | None = None,
    checkpoint_every: int = 0,
    hidden_layers: tuple[int, ...] = (96, 48),
    learning_rate_init: float = 0.001,
    batch_size: int = 128,
    shuffle: bool = True,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
) -> DeepFusionGpuResult:
    torch = require_torch()
    torch.manual_seed(seed)
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    x_train = assemble_feature_matrix(train_bundle, modalities)
    x_test = assemble_feature_matrix(test_bundle, modalities)
    y_train = np.asarray(train_bundle["labels"])
    y_test = np.asarray(test_bundle["labels"])

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    train_features = torch.tensor(x_train_scaled, dtype=torch.float32, device=device)
    test_features = torch.tensor(x_test_scaled, dtype=torch.float32, device=device)
    train_targets = torch.tensor(encode_labels(labels, y_train.tolist()), dtype=torch.long, device=device)
    _ = torch.tensor(encode_labels(labels, y_test.tolist()), dtype=torch.long, device=device)

    effective_batch_size = max(1, min(int(batch_size), len(x_train_scaled)))
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_features, train_targets),
        batch_size=effective_batch_size,
        shuffle=shuffle,
    )

    network = _build_network(train_features.shape[1], hidden_layers, len(labels)).to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate_init)
    criterion = torch.nn.CrossEntropyLoss()

    curve_rows: list[dict[str, object]] = []
    for epoch in range(1, epochs + 1):
        network.train()
        batch_losses: list[float] = []
        for batch_features, batch_targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = network(batch_features)
            loss = criterion(logits, batch_targets)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu().item()))

        network.eval()
        with torch.no_grad():
            train_logits = network(train_features)
            val_logits = network(test_features)
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
                    "state_dict": network.state_dict(),
                    "hidden_layers": hidden_layers,
                    "input_dim": int(train_features.shape[1]),
                },
                checkpoint_dir / f"{model_id.lower()}_epoch_{epoch:04d}.pt",
            )
            joblib.dump(scaler, checkpoint_dir / f"{model_id.lower()}_scaler_epoch_{epoch:04d}.joblib")
            pd.DataFrame(curve_rows).to_csv(checkpoint_dir / f"{model_id.lower()}_curve_until_{epoch:04d}.csv", index=False)

    wrapper = TorchDeepFusionWrapper(scaler, network, labels, device)
    y_pred = wrapper.predict(x_test).tolist()
    scores = compute_metrics(y_test.tolist(), y_pred)
    metrics = {
        "accuracy": round(scores.accuracy, 4),
        "macro_f1": round(scores.macro_f1, 4),
        "weighted_f1": round(scores.weighted_f1, 4),
        "uar": round(scores.unweighted_recall, 4),
        "inference_latency_ms": round(measure_inference_latency(wrapper, x_test), 4),
        "runtime_backend": "gpu",
        "device": device,
    }
    confusion = confusion_dataframe(y_test.tolist(), y_pred, labels)
    return DeepFusionGpuResult(
        model=wrapper,
        scaler=scaler,
        metrics=metrics,
        confusion=confusion,
        training_curve=pd.DataFrame(curve_rows),
        predictions=y_pred,
    )
