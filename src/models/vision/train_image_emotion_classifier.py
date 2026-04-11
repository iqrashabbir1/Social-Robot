from __future__ import annotations

import argparse
import time
from pathlib import Path

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


def train_image_emotion_classifier(
    project_root: Path,
    dataset_root: Path,
    labels_csv: Path,
    output_subdir: str = "local_real_train",
    target_label_set: str | None = "broad4_angry",
    epochs: int = 1000,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    image_size: int = 128,
    random_seed: int = 42,
    device: str = "cpu",
    log_every_epochs: int = 1,
    log_every_steps: int = 10,
) -> dict[str, str]:
    torch = require_torch()
    set_global_seed(random_seed)
    paths = Paper1Paths.from_project_root(project_root)
    paths.ensure()

    dataset_root = dataset_root if dataset_root.is_absolute() else (project_root / dataset_root)
    labels_csv = labels_csv if labels_csv.is_absolute() else (project_root / labels_csv)
    dataset_root = dataset_root.resolve()
    labels_csv = labels_csv.resolve()

    output_dir = paths.outputs_csv_paper1 / output_subdir
    model_dir = project_root / "outputs" / "models" / "paper1" / output_subdir
    log_dir = paths.outputs_logs / "paper1" / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    dataset_df = load_dataset_records(
        dataset_root=dataset_root,
        labels_csv=labels_csv,
        split_mode="train_test",
        test_size=0.2,
        random_seed=random_seed,
        target_label_set=target_label_set,
    )
    dataset_df = dataset_df.loc[dataset_df["label"].notna()].copy()
    materialized_df = materialize_frame_records(
        dataset_df,
        cache_dir=output_dir / "materialized_frames",
        width=image_size,
        height=image_size,
    )
    train_df = materialized_df.loc[materialized_df["split"] == "train"].copy()
    val_df = materialized_df.loc[materialized_df["split"] == "test"].copy()
    if train_df.empty or val_df.empty:
        raise RuntimeError("Training requires labeled train and validation samples after materialization.")

    class_labels = sorted(train_df["label"].dropna().unique().tolist())
    label_to_index = {label: index for index, label in enumerate(class_labels)}
    train_dataset = FrameEmotionDataset(train_df, label_to_index, image_size=image_size)
    val_dataset = FrameEmotionDataset(val_df, label_to_index, image_size=image_size)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=min(batch_size, len(train_dataset)),
        shuffle=True,
        collate_fn=frame_collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=min(batch_size, len(val_dataset)),
        shuffle=False,
        collate_fn=frame_collate_fn,
    )

    model = SimpleEmotionCNN(len(class_labels), image_size=image_size)
    model.network.to(device)
    optimizer = torch.optim.Adam(model.network.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    history_rows: list[dict[str, object]] = []
    best_state = None
    best_macro_f1 = -1.0
    train_steps_per_epoch = len(train_loader)
    training_progress_path = output_dir / "training_progress_latest.csv"
    latest_status_path = log_dir / "latest_status.json"
    progress_events_path = log_dir / "training_progress_events.csv"
    epoch_durations: list[float] = []
    overall_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        model.network.train()
        train_losses: list[float] = []
        train_true: list[str] = []
        train_pred: list[str] = []
        print(f"[train] epoch {epoch}/{epochs} starting on {device} with {train_steps_per_epoch} steps")
        for step_index, (features, targets, _) in enumerate(train_loader, start=1):
            features = features.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model.network(features)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))
            pred_idx = logits.argmax(dim=1).detach().cpu().numpy().tolist()
            target_idx = targets.detach().cpu().numpy().tolist()
            train_true.extend(class_labels[int(i)] for i in target_idx)
            train_pred.extend(class_labels[int(i)] for i in pred_idx)
            if log_every_steps > 0 and (step_index == 1 or step_index % log_every_steps == 0 or step_index == train_steps_per_epoch):
                elapsed_so_far = time.perf_counter() - epoch_start
                avg_step_seconds = elapsed_so_far / max(step_index, 1)
                remaining_steps = max(train_steps_per_epoch - step_index, 0)
                eta_seconds = remaining_steps * avg_step_seconds
                latest_status = {
                    "stage": "training",
                    "epoch": epoch,
                    "epochs_total": epochs,
                    "step": step_index,
                    "steps_total": train_steps_per_epoch,
                    "step_loss": round(float(loss.detach().cpu().item()), 6),
                    "elapsed_seconds": round(elapsed_so_far, 2),
                    "eta_seconds": round(eta_seconds, 2),
                    "device": device,
                    "output_subdir": output_subdir,
                }
                write_json(latest_status_path, latest_status)
                print(
                    f"[train] epoch {epoch}/{epochs} step {step_index}/{train_steps_per_epoch} "
                    f"loss={float(loss.detach().cpu().item()):.4f} eta={eta_seconds:.1f}s"
                )

        model.network.eval()
        val_true: list[str] = []
        val_pred: list[str] = []
        val_rows: list[dict[str, object]] = []
        with torch.no_grad():
            for features, targets, meta_rows in val_loader:
                features = features.to(device)
                targets = targets.to(device)
                logits = model.network(features)
                pred_idx = logits.argmax(dim=1).detach().cpu().numpy().tolist()
                target_idx = targets.detach().cpu().numpy().tolist()
                probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
                for row_index, pred_class, target_class in zip(range(len(pred_idx)), pred_idx, target_idx):
                    true_label = class_labels[int(target_class)]
                    pred_label = class_labels[int(pred_class)]
                    val_true.append(true_label)
                    val_pred.append(pred_label)
                    meta = meta_rows[row_index]
                    val_rows.append(
                        {
                            "sample_id": meta["sample_id"],
                            "frame_path": meta["frame_path"],
                            "true_label": true_label,
                            "predicted_label": pred_label,
                            "confidence": float(probs[row_index][pred_class]),
                            "split": "validation",
                            "epoch": epoch,
                        }
                    )

        train_metrics, _ = evaluate_predictions(train_true, train_pred, class_labels)
        val_metrics, _ = evaluate_predictions(val_true, val_pred, class_labels)
        epoch_seconds = time.perf_counter() - epoch_start
        epoch_durations.append(epoch_seconds)
        avg_epoch_seconds = sum(epoch_durations) / max(len(epoch_durations), 1)
        remaining_epochs = max(epochs - epoch, 0)
        total_elapsed_seconds = time.perf_counter() - overall_start
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": round(sum(train_losses) / max(len(train_losses), 1), 6),
                "train_accuracy": train_metrics["accuracy"],
                "train_macro_f1": train_metrics["macro_f1"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_weighted_f1": val_metrics["weighted_f1"],
                "val_unweighted_recall": val_metrics["unweighted_recall"],
                "epoch_seconds": round(epoch_seconds, 3),
                "avg_epoch_seconds": round(avg_epoch_seconds, 3),
                "remaining_epochs": remaining_epochs,
                "eta_seconds": round(avg_epoch_seconds * remaining_epochs, 2),
                "total_elapsed_seconds": round(total_elapsed_seconds, 2),
            }
        )
        history_df = pd.DataFrame(history_rows)
        write_dataframe(training_progress_path, history_df)
        write_dataframe(progress_events_path, history_df)
        latest_status = {
            "stage": "epoch_complete",
            "epoch": epoch,
            "epochs_total": epochs,
            "train_loss": history_rows[-1]["train_loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_weighted_f1": val_metrics["weighted_f1"],
            "val_unweighted_recall": val_metrics["unweighted_recall"],
            "epoch_seconds": round(epoch_seconds, 2),
            "avg_epoch_seconds": round(avg_epoch_seconds, 2),
            "eta_seconds": round(avg_epoch_seconds * remaining_epochs, 2),
            "total_elapsed_seconds": round(total_elapsed_seconds, 2),
            "device": device,
            "output_subdir": output_subdir,
        }
        write_json(latest_status_path, latest_status)
        if epoch == 1 or epoch % max(log_every_epochs, 1) == 0 or epoch == epochs:
            print(
                f"[epoch] {epoch}/{epochs} "
                f"train_loss={history_rows[-1]['train_loss']:.4f} "
                f"train_acc={train_metrics['accuracy']:.4f} "
                f"val_acc={val_metrics['accuracy']:.4f} "
                f"val_macro_f1={val_metrics['macro_f1']:.4f} "
                f"epoch_time={epoch_seconds:.1f}s "
                f"eta={avg_epoch_seconds * remaining_epochs:.1f}s"
            )
        if val_metrics["macro_f1"] >= best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            best_state = model.network.state_dict()
            best_predictions_df = pd.DataFrame(val_rows)
            best_metrics = val_metrics
            best_confusion = evaluate_predictions(val_true, val_pred, class_labels)[1]

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")

    model_path = model_dir / "best_model.pt"
    save_model_checkpoint(model_path, best_state, class_labels, image_size, target_label_set)
    history_path = output_dir / "training_history.csv"
    predictions_path = output_dir / "local_validation_predictions.csv"
    confusion_path = output_dir / "local_validation_confusion_matrix.csv"
    metrics_path = output_dir / "local_validation_metrics.csv"
    summary_path = output_dir / "training_summary.json"
    monitor_readme_path = output_dir / "monitoring_files.txt"

    write_dataframe(history_path, pd.DataFrame(history_rows))
    write_dataframe(predictions_path, best_predictions_df)
    write_dataframe(confusion_path, best_confusion)
    metrics_df = pd.DataFrame(
        [
            {
                "evaluation_stage": "local_validation",
                "dataset_name": dataset_root.name,
                "dataset_root": str(dataset_root),
                "labels_csv": str(labels_csv),
                "target_label_set": target_label_set or "",
                "num_samples": int(len(best_predictions_df)),
                "accuracy": best_metrics["accuracy"],
                "macro_f1": best_metrics["macro_f1"],
                "weighted_f1": best_metrics["weighted_f1"],
                "unweighted_recall": best_metrics["unweighted_recall"],
                "epochs": epochs,
                "model_name": "simple_emotion_cnn",
                "model_artifact": str(model_path),
            }
        ]
    )
    write_dataframe(metrics_path, metrics_df)
    write_training_summary(
        summary_path,
        {
            "dataset_root": str(dataset_root),
            "labels_csv": str(labels_csv),
            "target_label_set": target_label_set,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "image_size": image_size,
            "device": device,
            "model_artifact": str(model_path),
            "outputs": {
                "metrics_csv": str(metrics_path),
                "history_csv": str(history_path),
                "training_progress_csv": str(training_progress_path),
                "predictions_csv": str(predictions_path),
                "confusion_csv": str(confusion_path),
                "latest_status_json": str(latest_status_path),
                "progress_events_csv": str(progress_events_path),
            },
        },
    )
    monitor_readme_path.write_text(
        "\n".join(
            [
                f"training_progress_csv={training_progress_path}",
                f"latest_status_json={latest_status_path}",
                f"progress_events_csv={progress_events_path}",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "model_path": str(model_path),
        "metrics_csv": str(metrics_path),
        "history_csv": str(history_path),
        "training_progress_csv": str(training_progress_path),
        "predictions_csv": str(predictions_path),
        "confusion_csv": str(confusion_path),
        "summary_json": str(summary_path),
        "latest_status_json": str(latest_status_path),
        "progress_events_csv": str(progress_events_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a simple real-dataset image emotion classifier.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--output-subdir", default="local_real_train")
    parser.add_argument("--target-label-set", default="broad4_angry")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log-every-epochs", type=int, default=1)
    parser.add_argument("--log-every-steps", type=int, default=10)
    args = parser.parse_args()
    train_image_emotion_classifier(
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
        log_every_epochs=max(args.log_every_epochs, 1),
        log_every_steps=max(args.log_every_steps, 0),
    )


if __name__ == "__main__":
    main()
