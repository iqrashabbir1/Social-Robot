from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.common.io_utils import write_dataframe
from src.common.paths import Paper1Paths
from src.data.dataset_loader import load_dataset_records, materialize_frame_records
from src.models.torch_runtime import require_torch
from src.models.vision.image_emotion_classifier import (
    FrameEmotionDataset,
    evaluate_predictions,
    frame_collate_fn,
    load_model_checkpoint,
    write_training_summary,
)


def evaluate_image_emotion_classifier(
    project_root: Path,
    model_path: Path,
    dataset_root: Path,
    labels_csv: Path,
    output_subdir: str = "public_eval",
    target_label_set: str | None = "broad4_angry",
    image_size_override: int | None = None,
    batch_size: int = 32,
    device: str = "cpu",
) -> dict[str, str]:
    torch = require_torch()
    paths = Paper1Paths.from_project_root(project_root)
    paths.ensure()

    model_path = model_path if model_path.is_absolute() else (project_root / model_path)
    dataset_root = dataset_root if dataset_root.is_absolute() else (project_root / dataset_root)
    labels_csv = labels_csv if labels_csv.is_absolute() else (project_root / labels_csv)
    model_path = model_path.resolve()
    dataset_root = dataset_root.resolve()
    labels_csv = labels_csv.resolve()

    output_dir = paths.outputs_csv_paper1 / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    model, class_labels, checkpoint_image_size, _ = load_model_checkpoint(model_path, device=device)
    image_size = image_size_override or checkpoint_image_size

    dataset_df = load_dataset_records(
        dataset_root=dataset_root,
        labels_csv=labels_csv,
        split_mode="test_only",
        target_label_set=target_label_set,
    )
    dataset_df = dataset_df.loc[dataset_df["label"].notna()].copy()
    materialized_df = materialize_frame_records(
        dataset_df,
        cache_dir=output_dir / "materialized_frames",
        width=image_size,
        height=image_size,
    )
    eval_df = materialized_df.loc[materialized_df["label"].notna()].copy()
    if eval_df.empty:
        raise RuntimeError("External evaluation requires labeled samples.")

    label_to_index = {label: index for index, label in enumerate(class_labels)}
    eval_dataset = FrameEmotionDataset(eval_df, label_to_index, image_size=image_size)
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=min(batch_size, len(eval_dataset)),
        shuffle=False,
        collate_fn=frame_collate_fn,
    )

    model.network.to(device)
    model.network.eval()

    y_true: list[str] = []
    y_pred: list[str] = []
    rows: list[dict[str, object]] = []
    with torch.no_grad():
        for features, targets, meta_rows in eval_loader:
            features = features.to(device)
            logits = model.network(features)
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
            pred_idx = logits.argmax(dim=1).detach().cpu().numpy().tolist()
            target_idx = targets.detach().cpu().numpy().tolist()
            for row_index, pred_class, target_class in zip(range(len(pred_idx)), pred_idx, target_idx):
                true_label = class_labels[int(target_class)]
                pred_label = class_labels[int(pred_class)]
                y_true.append(true_label)
                y_pred.append(pred_label)
                meta = meta_rows[row_index]
                rows.append(
                    {
                        "sample_id": meta["sample_id"],
                        "frame_path": meta["frame_path"],
                        "true_label": true_label,
                        "predicted_label": pred_label,
                        "confidence": float(probs[row_index][pred_class]),
                        "split": "external_test",
                    }
                )

    metrics, confusion = evaluate_predictions(y_true, y_pred, class_labels)
    predictions_path = output_dir / "public_test_predictions.csv"
    metrics_path = output_dir / "public_test_metrics.csv"
    confusion_path = output_dir / "public_test_confusion_matrix.csv"
    summary_path = output_dir / "public_test_summary.json"

    predictions_df = pd.DataFrame(rows)
    metrics_df = pd.DataFrame(
        [
            {
                "evaluation_stage": "public_test",
                "dataset_name": dataset_root.name,
                "dataset_root": str(dataset_root),
                "labels_csv": str(labels_csv),
                "target_label_set": target_label_set or "",
                "num_samples": len(predictions_df),
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "unweighted_recall": metrics["unweighted_recall"],
                "epochs": None,
                "model_name": "simple_emotion_cnn",
                "model_artifact": str(model_path),
            }
        ]
    )
    write_dataframe(predictions_path, predictions_df)
    write_dataframe(metrics_path, metrics_df)
    write_dataframe(confusion_path, confusion)
    write_training_summary(
        summary_path,
        {
            "dataset_root": str(dataset_root),
            "labels_csv": str(labels_csv),
            "target_label_set": target_label_set,
            "model_artifact": str(model_path),
            "outputs": {
                "metrics_csv": str(metrics_path),
                "predictions_csv": str(predictions_path),
                "confusion_csv": str(confusion_path),
            },
        },
    )
    return {
        "metrics_csv": str(metrics_path),
        "predictions_csv": str(predictions_path),
        "confusion_csv": str(confusion_path),
        "summary_json": str(summary_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained image emotion classifier on a public dataset.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--output-subdir", default="public_eval")
    parser.add_argument("--target-label-set", default="broad4_angry")
    parser.add_argument("--image-size", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    evaluate_image_emotion_classifier(
        project_root=Path(args.project_root).resolve(),
        model_path=Path(args.model_path),
        dataset_root=Path(args.dataset_root),
        labels_csv=Path(args.labels_csv),
        output_subdir=args.output_subdir,
        target_label_set=args.target_label_set or None,
        image_size_override=args.image_size or None,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
