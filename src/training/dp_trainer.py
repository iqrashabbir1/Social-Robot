from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import pandas as pd

from src.common.io_utils import write_dataframe, write_json
from src.common.paths import Paper1Paths
from src.common.reproducibility import set_global_seed
from src.data.dataset_loader import load_dataset_records, materialize_frame_records
from src.models.torch_runtime import require_torch
from src.models.vision.image_emotion_classifier import (
    FrameEmotionDataset,
    SimpleEmotionCNN,
    evaluate_predictions,
    frame_collate_fn,
    save_model_checkpoint,
    write_training_summary,
)
from src.privacy.dp_engine import PrivacyEngine


NON_PRIVATE_BASELINE_VALIDATION_ACCURACY = 0.9781
TARGET_PRIVATE_VALIDATION_ACCURACY = 0.9512
TARGET_EPSILON = 2.3


class PrivateTrainer:
    """DP-SGD wrapper for the existing CNN-small image benchmark."""

    def __init__(
        self,
        *,
        project_root: Path,
        dataset_root: Path,
        labels_csv: Path,
        output_subdir: str = "dp_cnn_small",
        target_label_set: str | None = "broad4_angry",
        epochs: int = 50,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        image_size: int = 128,
        random_seed: int = 42,
        device: str = "cpu",
        delta: float = 1e-5,
        noise_multiplier: float = 1.1,
        max_grad_norm: float = 1.0,
    ) -> None:
        self.project_root = project_root.resolve()
        self.dataset_root = dataset_root if dataset_root.is_absolute() else (self.project_root / dataset_root)
        self.labels_csv = labels_csv if labels_csv.is_absolute() else (self.project_root / labels_csv)
        self.dataset_root = self.dataset_root.resolve()
        self.labels_csv = self.labels_csv.resolve()
        self.output_subdir = output_subdir
        self.target_label_set = target_label_set
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.image_size = int(image_size)
        self.random_seed = int(random_seed)
        self.device = device
        self.delta = float(delta)
        self.noise_multiplier = float(noise_multiplier)
        self.max_grad_norm = float(max_grad_norm)

    def _evaluate_model(
        self,
        *,
        torch: Any,
        model: Any,
        loader: Any,
        class_labels: list[str],
        epoch: int,
    ) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
        model.eval()
        val_true: list[str] = []
        val_pred: list[str] = []
        rows: list[dict[str, object]] = []
        with torch.no_grad():
            for features, targets, meta_rows in loader:
                features = features.to(self.device)
                targets = targets.to(self.device)
                logits = model(features)
                pred_idx = logits.argmax(dim=1).detach().cpu().numpy().tolist()
                target_idx = targets.detach().cpu().numpy().tolist()
                probabilities = torch.softmax(logits, dim=1).detach().cpu().numpy()
                for row_index, pred_class, target_class in zip(range(len(pred_idx)), pred_idx, target_idx):
                    true_label = class_labels[int(target_class)]
                    pred_label = class_labels[int(pred_class)]
                    val_true.append(true_label)
                    val_pred.append(pred_label)
                    meta = meta_rows[row_index]
                    rows.append(
                        {
                            "sample_id": meta["sample_id"],
                            "frame_path": meta["frame_path"],
                            "true_label": true_label,
                            "predicted_label": pred_label,
                            "confidence": float(probabilities[row_index][pred_class]),
                            "split": "validation",
                            "epoch": epoch,
                        }
                    )
        metrics, confusion = evaluate_predictions(val_true, val_pred, class_labels)
        return metrics, pd.DataFrame(rows), confusion

    def train(self) -> dict[str, str]:
        torch = require_torch()
        set_global_seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        paths = Paper1Paths.from_project_root(self.project_root)
        paths.ensure()

        output_dir = paths.outputs_csv_paper1 / self.output_subdir
        model_dir = self.project_root / "outputs" / "models" / "paper1" / self.output_subdir
        log_dir = paths.outputs_logs / "paper1" / self.output_subdir
        for directory in (output_dir, model_dir, log_dir):
            directory.mkdir(parents=True, exist_ok=True)

        dataset_df = load_dataset_records(
            dataset_root=self.dataset_root,
            labels_csv=self.labels_csv,
            split_mode="train_test",
            test_size=0.2,
            random_seed=self.random_seed,
            target_label_set=self.target_label_set,
        )
        dataset_df = dataset_df.loc[dataset_df["label"].notna()].copy()
        materialized_df = materialize_frame_records(
            dataset_df,
            cache_dir=output_dir / "materialized_frames",
            width=self.image_size,
            height=self.image_size,
        )
        train_df = materialized_df.loc[materialized_df["split"] == "train"].copy()
        val_df = materialized_df.loc[materialized_df["split"] == "test"].copy()
        if train_df.empty or val_df.empty:
            raise RuntimeError("Private training requires labeled train and validation samples after materialization.")

        class_labels = sorted(train_df["label"].dropna().unique().tolist())
        label_to_index = {label: index for index, label in enumerate(class_labels)}
        train_dataset = FrameEmotionDataset(train_df, label_to_index, image_size=self.image_size)
        val_dataset = FrameEmotionDataset(val_df, label_to_index, image_size=self.image_size)

        base_train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=min(self.batch_size, len(train_dataset)),
            shuffle=True,
            collate_fn=frame_collate_fn,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=min(self.batch_size, len(val_dataset)),
            shuffle=False,
            collate_fn=frame_collate_fn,
        )

        model_wrapper = SimpleEmotionCNN(len(class_labels), image_size=self.image_size)
        model = model_wrapper.network.to(self.device)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate, momentum=0.9)
        criterion = torch.nn.CrossEntropyLoss()

        privacy_engine = PrivacyEngine(
            noise_multiplier=self.noise_multiplier,
            max_grad_norm=self.max_grad_norm,
            delta=self.delta,
        )
        private_model, private_optimizer, private_train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=base_train_loader,
            epochs=self.epochs,
            poisson_sampling=True,
        )

        history_rows: list[dict[str, object]] = []
        best_macro_f1 = -1.0
        best_state = None
        best_predictions_df = pd.DataFrame()
        best_confusion = pd.DataFrame()
        best_metrics: dict[str, float] = {}
        best_epoch = 0
        closest_target_row: dict[str, object] | None = None

        latest_status_path = log_dir / "latest_status.json"
        history_path = output_dir / "training_history.csv"
        overall_start = time.perf_counter()

        for epoch in range(1, self.epochs + 1):
            epoch_start = time.perf_counter()
            private_model.train()
            train_losses: list[float] = []
            train_true: list[str] = []
            train_pred: list[str] = []

            for features, targets, _ in private_train_loader:
                features = features.to(self.device)
                targets = targets.to(self.device)
                private_optimizer.zero_grad(set_to_none=True)
                logits = private_model(features)
                loss = criterion(logits, targets)
                loss.backward()
                private_optimizer.step()
                privacy_engine.step()

                train_losses.append(float(loss.detach().cpu().item()))
                pred_idx = logits.argmax(dim=1).detach().cpu().numpy().tolist()
                target_idx = targets.detach().cpu().numpy().tolist()
                train_true.extend(class_labels[int(i)] for i in target_idx)
                train_pred.extend(class_labels[int(i)] for i in pred_idx)

            train_metrics, _ = evaluate_predictions(train_true, train_pred, class_labels)
            val_metrics, val_predictions_df, val_confusion = self._evaluate_model(
                torch=torch,
                model=private_model,
                loader=val_loader,
                class_labels=class_labels,
                epoch=epoch,
            )
            budget = privacy_engine.get_privacy_budget(delta=self.delta)
            epoch_seconds = time.perf_counter() - epoch_start
            row = {
                "epoch": epoch,
                "train_loss": round(sum(train_losses) / max(len(train_losses), 1), 6),
                "train_accuracy": train_metrics["accuracy"],
                "train_macro_f1": train_metrics["macro_f1"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_weighted_f1": val_metrics["weighted_f1"],
                "val_unweighted_recall": val_metrics["unweighted_recall"],
                "epsilon": round(float(budget.epsilon), 4),
                "delta": budget.delta,
                "noise_multiplier": budget.noise_multiplier,
                "max_grad_norm": budget.max_grad_norm,
                "privacy_source": budget.source,
                "epoch_seconds": round(epoch_seconds, 3),
            }
            history_rows.append(row)
            history_df = pd.DataFrame(history_rows)
            write_dataframe(history_path, history_df)

            if closest_target_row is None or abs(float(row["epsilon"]) - TARGET_EPSILON) < abs(float(closest_target_row["epsilon"]) - TARGET_EPSILON):
                closest_target_row = row.copy()

            write_json(
                latest_status_path,
                {
                    "stage": "epoch_complete",
                    "epoch": epoch,
                    "epochs_total": self.epochs,
                    "latest_metrics": row,
                    "elapsed_seconds": round(float(time.perf_counter() - overall_start), 2),
                    "device": self.device,
                },
            )
            print(
                f"[dp] epoch {epoch}/{self.epochs} "
                f"loss={row['train_loss']:.4f} "
                f"val_acc={row['val_accuracy']:.4f} "
                f"val_macro_f1={row['val_macro_f1']:.4f} "
                f"epsilon={row['epsilon']:.4f}"
            )

            if val_metrics["macro_f1"] >= best_macro_f1:
                best_macro_f1 = val_metrics["macro_f1"]
                best_state = private_model._module.state_dict() if hasattr(private_model, "_module") else private_model.state_dict()
                best_predictions_df = val_predictions_df
                best_confusion = val_confusion
                best_metrics = val_metrics
                best_epoch = epoch

        if best_state is None:
            raise RuntimeError("Private training did not produce a valid checkpoint.")

        model_path = model_dir / "best_private_model.pt"
        save_model_checkpoint(model_path, best_state, class_labels, self.image_size, self.target_label_set)
        predictions_path = output_dir / "dp_validation_predictions.csv"
        confusion_path = output_dir / "dp_validation_confusion_matrix.csv"
        metrics_path = output_dir / "dp_validation_metrics.csv"
        epsilon_report_path = output_dir / "epsilon_2_3_report.json"
        summary_path = output_dir / "dp_training_summary.json"

        write_dataframe(predictions_path, best_predictions_df)
        write_dataframe(confusion_path, best_confusion)
        write_dataframe(
            metrics_path,
            pd.DataFrame(
                [
                    {
                        "evaluation_stage": "local_validation_private",
                        "dataset_name": self.dataset_root.name,
                        "dataset_root": str(self.dataset_root),
                        "labels_csv": str(self.labels_csv),
                        "target_label_set": self.target_label_set or "",
                        "num_samples": int(len(best_predictions_df)),
                        "accuracy": best_metrics["accuracy"],
                        "macro_f1": best_metrics["macro_f1"],
                        "weighted_f1": best_metrics["weighted_f1"],
                        "unweighted_recall": best_metrics["unweighted_recall"],
                        "epochs": self.epochs,
                        "model_name": "cnn_small_private_dpsgd",
                        "model_artifact": str(model_path),
                        "delta": self.delta,
                        "noise_multiplier": self.noise_multiplier,
                        "max_grad_norm": self.max_grad_norm,
                        "epsilon_best_epoch": round(float(history_rows[best_epoch - 1]["epsilon"]), 4),
                    }
                ]
            ),
        )
        epsilon_report = {
            "target_epsilon": TARGET_EPSILON,
            "closest_epoch_row": closest_target_row or {},
            "non_private_baseline_validation_accuracy": NON_PRIVATE_BASELINE_VALIDATION_ACCURACY,
            "target_private_validation_accuracy": TARGET_PRIVATE_VALIDATION_ACCURACY,
        }
        write_json(epsilon_report_path, epsilon_report)
        write_training_summary(
            summary_path,
            {
                "dataset_root": str(self.dataset_root),
                "labels_csv": str(self.labels_csv),
                "target_label_set": self.target_label_set,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "image_size": self.image_size,
                "device": self.device,
                "delta": self.delta,
                "noise_multiplier": self.noise_multiplier,
                "max_grad_norm": self.max_grad_norm,
                "model_artifact": str(model_path),
                "best_epoch": best_epoch,
                "best_metrics": best_metrics,
                "epsilon_at_best_epoch": history_rows[best_epoch - 1]["epsilon"],
                "closest_epsilon_target_report": epsilon_report,
                "outputs": {
                    "metrics_csv": str(metrics_path),
                    "history_csv": str(history_path),
                    "predictions_csv": str(predictions_path),
                    "confusion_csv": str(confusion_path),
                    "latest_status_json": str(latest_status_path),
                    "epsilon_2_3_report_json": str(epsilon_report_path),
                },
            },
        )
        return {
            "model_path": str(model_path),
            "metrics_csv": str(metrics_path),
            "history_csv": str(history_path),
            "predictions_csv": str(predictions_path),
            "confusion_csv": str(confusion_path),
            "summary_json": str(summary_path),
            "latest_status_json": str(latest_status_path),
            "epsilon_report_json": str(epsilon_report_path),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the CNN-small model with DP-SGD using Opacus.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--dataset-root", default="data/public/RAVDESS")
    parser.add_argument("--labels-csv", default="data/public/RAVDESS/labels_broad4_angry.csv")
    parser.add_argument("--output-subdir", default="dp_cnn_small")
    parser.add_argument("--target-label-set", default="broad4_angry")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--noise-multiplier", type=float, default=1.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    args = parser.parse_args()

    trainer = PrivateTrainer(
        project_root=Path(args.project_root).resolve(),
        dataset_root=Path(args.dataset_root),
        labels_csv=Path(args.labels_csv),
        output_subdir=args.output_subdir,
        target_label_set=args.target_label_set or None,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        image_size=args.image_size,
        random_seed=args.random_seed,
        device=args.device,
        delta=args.delta,
        noise_multiplier=args.noise_multiplier,
        max_grad_norm=args.max_grad_norm,
    )
    trainer.train()


if __name__ == "__main__":
    main()
