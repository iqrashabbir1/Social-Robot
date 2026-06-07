from __future__ import annotations
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path
from typing import Any

import cv2
import pandas as pd

from src.common.io_utils import write_dataframe, write_json
from src.evaluation.metrics_classification import ClassificationMetrics, compute_metrics, confusion_dataframe
from src.models.domain_adversarial import DANNMultimodalEmotionModel
from src.models.torch_runtime import require_torch, synchronize_if_needed


torch = require_torch()


def compute_mmd(
    source_features: torch.Tensor,
    target_features: torch.Tensor,
    kernel_mul: float = 2.0,
    kernel_num: int = 5,
    fix_sigma: float | None = None,
) -> torch.Tensor:
    """Compute multi-kernel MMD with RBF kernels."""
    if source_features.size(0) == 0 or target_features.size(0) == 0:
        return source_features.new_tensor(0.0)

    total = torch.cat([source_features, target_features], dim=0)
    total0 = total.unsqueeze(0)
    total1 = total.unsqueeze(1)
    l2_distance = ((total0 - total1) ** 2).sum(2)

    if fix_sigma is not None:
        bandwidth = fix_sigma
    else:
        denominator = max(total.size(0) ** 2 - total.size(0), 1)
        bandwidth = torch.sum(l2_distance.detach()) / denominator
    bandwidth = bandwidth / (kernel_mul ** (kernel_num // 2))
    bandwidth_list = [bandwidth * (kernel_mul**index) for index in range(kernel_num)]
    kernels = [torch.exp(-l2_distance / (bandwidth_item + 1e-8)) for bandwidth_item in bandwidth_list]
    kernel_sum = sum(kernels)

    batch_size = source_features.size(0)
    source_source = kernel_sum[:batch_size, :batch_size]
    target_target = kernel_sum[batch_size:, batch_size:]
    source_target = kernel_sum[:batch_size, batch_size:]
    target_source = kernel_sum[batch_size:, :batch_size]
    return source_source.mean() + target_target.mean() - source_target.mean() - target_source.mean()


class DomainAdaptationFrameDataset(torch.utils.data.Dataset):
    """Image-frame dataset with optional labels for source and target domains."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        label_to_index: dict[str, int],
        image_size: int = 128,
        include_labels: bool = True,
    ) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.label_to_index = label_to_index
        self.image_size = int(image_size)
        self.include_labels = bool(include_labels)

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        row = self.dataframe.iloc[index]
        frame = cv2.imread(str(row["frame_path"]))
        if frame is None:
            raise ValueError(f"Could not read frame: {row['frame_path']}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        tensor = torch.tensor(frame.transpose(2, 0, 1), dtype=torch.float32) / 255.0

        if self.include_labels and pd.notna(row.get("label")):
            label_value = self.label_to_index[str(row["label"])]
        else:
            label_value = -1
        label_tensor = torch.tensor(label_value, dtype=torch.long)
        return tensor, label_tensor, row.to_dict()


def domain_adaptation_collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor, dict[str, Any]]]):
    features = torch.stack([item[0] for item in batch], dim=0)
    targets = torch.stack([item[1] for item in batch], dim=0)
    metadata = [item[2] for item in batch]
    return features, targets, metadata


@dataclass
class TrainerEvaluation:
    metrics: ClassificationMetrics
    predictions: pd.DataFrame
    confusion: pd.DataFrame


class DomainAdversarialTrainer:
    """Trainer implementing DANN + MK-MMD + progressive pseudo-labeling."""

    def __init__(
        self,
        model: DANNMultimodalEmotionModel,
        class_labels: list[str],
        device: str,
        learning_rate: float = 1e-4,
        lambda_domain: float = 0.5,
        lambda_mmd: float = 0.1,
        pseudo_label_weight: float = 0.2,
        warmup_epochs: int = 10,
    ) -> None:
        self.model = model
        self.class_labels = list(class_labels)
        self.device = device
        self.lambda_domain = float(lambda_domain)
        self.lambda_mmd = float(lambda_mmd)
        self.pseudo_label_weight = float(pseudo_label_weight)
        self.warmup_epochs = int(warmup_epochs)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.emotion_criterion = torch.nn.CrossEntropyLoss()
        self.domain_criterion = torch.nn.CrossEntropyLoss()

    def rampup_alpha(self, epoch_index: int) -> float:
        """Linearly ramps from 0 to 1 over ``warmup_epochs``."""
        if self.warmup_epochs <= 0:
            return 1.0
        return float(min(1.0, max(0.0, epoch_index / float(self.warmup_epochs))))

    def generate_pseudo_labels(
        self,
        target_logits: torch.Tensor,
        epoch_index: int,
        total_epochs: int,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Generate progressive pseudo-labels on target batches.

        The confidence threshold starts at 0.70 and increases to 0.95 across
        the training run to reduce confirmation bias in later epochs.
        """

        if total_epochs <= 1:
            threshold = 0.95
        else:
            threshold = 0.70 + (0.95 - 0.70) * (epoch_index / float(max(total_epochs - 1, 1)))
        probabilities = torch.softmax(target_logits.detach(), dim=1)
        confidences, pseudo_labels = probabilities.max(dim=1)
        confident_mask = confidences >= threshold
        return pseudo_labels, confident_mask, float(threshold)

    def train_epoch(
        self,
        source_loader: torch.utils.data.DataLoader,
        target_loader: torch.utils.data.DataLoader,
        epoch_index: int,
        total_epochs: int,
    ) -> dict[str, float]:
        self.model.train()
        alpha = self.rampup_alpha(epoch_index)

        source_iterator = cycle(source_loader)
        target_iterator = cycle(target_loader)
        total_steps = max(len(source_loader), len(target_loader))

        running = {
            "loss": 0.0,
            "emotion_loss": 0.0,
            "domain_loss": 0.0,
            "mmd_loss": 0.0,
            "pseudo_loss": 0.0,
            "pseudo_acceptance_rate": 0.0,
        }
        source_targets_all: list[int] = []
        source_predictions_all: list[int] = []

        for _step in range(total_steps):
            source_images, source_labels, _ = next(source_iterator)
            target_images, _, _ = next(target_iterator)
            source_images = source_images.to(self.device)
            source_labels = source_labels.to(self.device)
            target_images = target_images.to(self.device)

            self.optimizer.zero_grad()

            source_output = self.model(source_images, grl_alpha=alpha)
            target_output = self.model(target_images, grl_alpha=alpha)

            emotion_loss = self.emotion_criterion(source_output.emotion_logits, source_labels)

            source_domain_labels = torch.zeros(source_images.size(0), dtype=torch.long, device=self.device)
            target_domain_labels = torch.ones(target_images.size(0), dtype=torch.long, device=self.device)
            source_domain_loss = self.domain_criterion(source_output.domain_logits, source_domain_labels)
            target_domain_loss = self.domain_criterion(target_output.domain_logits, target_domain_labels)
            domain_loss = 0.5 * (source_domain_loss + target_domain_loss)

            mmd_loss = compute_mmd(source_output.features, target_output.features)

            pseudo_labels, confident_mask, threshold = self.generate_pseudo_labels(
                target_output.emotion_logits,
                epoch_index=epoch_index,
                total_epochs=total_epochs,
            )
            if bool(confident_mask.any()):
                pseudo_loss = self.emotion_criterion(
                    target_output.emotion_logits[confident_mask],
                    pseudo_labels[confident_mask],
                )
                pseudo_acceptance_rate = float(confident_mask.float().mean().item())
            else:
                pseudo_loss = source_output.features.new_tensor(0.0)
                pseudo_acceptance_rate = 0.0

            total_loss = (
                emotion_loss
                + alpha * self.lambda_domain * domain_loss
                + alpha * self.lambda_mmd * mmd_loss
                + alpha * self.pseudo_label_weight * pseudo_loss
            )
            total_loss.backward()
            self.optimizer.step()

            running["loss"] += float(total_loss.item())
            running["emotion_loss"] += float(emotion_loss.item())
            running["domain_loss"] += float(domain_loss.item())
            running["mmd_loss"] += float(mmd_loss.item())
            running["pseudo_loss"] += float(pseudo_loss.item())
            running["pseudo_acceptance_rate"] += pseudo_acceptance_rate

            source_predictions = torch.argmax(source_output.emotion_logits.detach(), dim=1)
            source_targets_all.extend(source_labels.detach().cpu().tolist())
            source_predictions_all.extend(source_predictions.cpu().tolist())
            running["pseudo_threshold"] = threshold

        metric_count = float(max(total_steps, 1))
        y_true = [self.class_labels[index] for index in source_targets_all]
        y_pred = [self.class_labels[index] for index in source_predictions_all]
        source_metrics = compute_metrics(y_true, y_pred)

        return {
            "alpha": alpha,
            "loss": running["loss"] / metric_count,
            "emotion_loss": running["emotion_loss"] / metric_count,
            "domain_loss": running["domain_loss"] / metric_count,
            "mmd_loss": running["mmd_loss"] / metric_count,
            "pseudo_loss": running["pseudo_loss"] / metric_count,
            "pseudo_acceptance_rate": running["pseudo_acceptance_rate"] / metric_count,
            "pseudo_threshold": float(running.get("pseudo_threshold", 0.70)),
            "source_train_accuracy": float(source_metrics.accuracy),
            "source_train_macro_f1": float(source_metrics.macro_f1),
        }

    @torch.no_grad()
    def evaluate(
        self,
        loader: torch.utils.data.DataLoader,
        dataset_name: str,
    ) -> TrainerEvaluation:
        self.model.eval()
        predictions_rows: list[dict[str, Any]] = []
        y_true: list[str] = []
        y_pred: list[str] = []

        for images, labels, metadata in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            output = self.model(images, grl_alpha=0.0)
            probabilities = torch.softmax(output.emotion_logits, dim=1)
            confidences, predicted_indices = probabilities.max(dim=1)
            predicted_labels = [self.class_labels[index] for index in predicted_indices.cpu().tolist()]

            for row_meta, label_index, pred_label, confidence in zip(
                metadata,
                labels.cpu().tolist(),
                predicted_labels,
                confidences.cpu().tolist(),
            ):
                true_label = self.class_labels[label_index] if label_index >= 0 else None
                if true_label is not None:
                    y_true.append(true_label)
                    y_pred.append(pred_label)
                predictions_rows.append(
                    {
                        "dataset_name": dataset_name,
                        "sample_id": row_meta.get("sample_id"),
                        "frame_path": row_meta.get("frame_path"),
                        "true_label": true_label,
                        "predicted_label": pred_label,
                        "confidence": round(float(confidence), 6),
                    }
                )

        metrics = compute_metrics(y_true, y_pred)
        confusion = confusion_dataframe(y_true, y_pred, self.class_labels)
        return TrainerEvaluation(
            metrics=metrics,
            predictions=pd.DataFrame(predictions_rows),
            confusion=confusion,
        )

    def save_checkpoint(
        self,
        checkpoint_path: Path,
        epoch_index: int,
        metadata: dict[str, Any],
    ) -> None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": int(epoch_index),
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "class_labels": self.class_labels,
                "metadata": metadata,
            },
            checkpoint_path,
        )


def write_evaluation_bundle(
    csv_dir: Path,
    stem: str,
    evaluation: TrainerEvaluation,
) -> tuple[Path, Path, Path]:
    metrics_path = csv_dir / f"{stem}_metrics.csv"
    predictions_path = csv_dir / f"{stem}_predictions.csv"
    confusion_path = csv_dir / f"{stem}_confusion.csv"

    metrics_df = pd.DataFrame(
        [
            {
                "split": stem,
                "accuracy": round(float(evaluation.metrics.accuracy), 4),
                "macro_f1": round(float(evaluation.metrics.macro_f1), 4),
                "weighted_f1": round(float(evaluation.metrics.weighted_f1), 4),
                "unweighted_recall": round(float(evaluation.metrics.unweighted_recall), 4),
            }
        ]
    )
    write_dataframe(metrics_path, metrics_df)
    write_dataframe(predictions_path, evaluation.predictions)
    write_dataframe(confusion_path, evaluation.confusion)
    return metrics_path, predictions_path, confusion_path


def write_training_status(status_path: Path, payload: dict[str, Any]) -> None:
    synchronize_if_needed(payload.get("device", "cpu"))
    write_json(status_path, payload)
