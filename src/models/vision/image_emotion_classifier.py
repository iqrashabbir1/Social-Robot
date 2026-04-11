from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.evaluation.metrics_classification import compute_metrics, confusion_dataframe
from src.models.torch_runtime import require_torch


class SimpleEmotionCNN:
    def __init__(self, num_classes: int, image_size: int = 128) -> None:
        torch = require_torch()
        self.image_size = image_size
        self.network = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((4, 4)),
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 4 * 4, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.25),
            torch.nn.Linear(128, num_classes),
        )


class FrameEmotionDataset:
    def __init__(self, dataframe: pd.DataFrame, label_to_index: dict[str, int], image_size: int = 128) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.label_to_index = label_to_index
        self.image_size = image_size
        self.torch = require_torch()

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int):
        row = self.dataframe.iloc[index]
        frame = cv2.imread(str(row["frame_path"]))
        if frame is None:
            raise ValueError(f"Could not read frame: {row['frame_path']}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        tensor = self.torch.tensor(frame.transpose(2, 0, 1), dtype=self.torch.float32) / 255.0
        label_tensor = self.torch.tensor(self.label_to_index[str(row["label"])], dtype=self.torch.long)
        return tensor, label_tensor, row.to_dict()


def frame_collate_fn(batch):
    torch = require_torch()
    features = torch.stack([item[0] for item in batch], dim=0)
    targets = torch.stack([item[1] for item in batch], dim=0)
    metadata = [item[2] for item in batch]
    return features, targets, metadata


@dataclass
class TrainingArtifacts:
    model_path: Path
    summary_json_path: Path
    metrics_csv_path: Path
    history_csv_path: Path
    predictions_csv_path: Path
    confusion_csv_path: Path


def save_model_checkpoint(
    path: Path,
    state_dict: dict[str, object],
    class_labels: list[str],
    image_size: int,
    target_label_set: str | None,
) -> None:
    torch = require_torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": state_dict,
            "class_labels": class_labels,
            "image_size": image_size,
            "target_label_set": target_label_set,
        },
        path,
    )


def load_model_checkpoint(path: Path, device: str = "cpu"):
    torch = require_torch()
    checkpoint = torch.load(path, map_location=device)
    labels = list(checkpoint["class_labels"])
    image_size = int(checkpoint.get("image_size", 128))
    model = SimpleEmotionCNN(len(labels), image_size=image_size)
    model.network.load_state_dict(checkpoint["state_dict"])
    model.network.to(device)
    model.network.eval()
    return model, labels, image_size, checkpoint.get("target_label_set")


def evaluate_predictions(y_true: list[str], y_pred: list[str], labels: list[str]) -> tuple[dict[str, float], pd.DataFrame]:
    scores = compute_metrics(y_true, y_pred)
    metrics = {
        "accuracy": round(scores.accuracy, 4),
        "macro_f1": round(scores.macro_f1, 4),
        "weighted_f1": round(scores.weighted_f1, 4),
        "unweighted_recall": round(scores.unweighted_recall, 4),
    }
    confusion = confusion_dataframe(y_true, y_pred, labels)
    return metrics, confusion


def write_training_summary(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
