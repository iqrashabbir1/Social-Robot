from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from fusion.fusion_logic import normalize_face_emotion
from perception.face_emotion import detect_face_emotion_from_frame
from src.common.io_utils import write_dataframe, write_json
from src.common.paths import Paper1Paths
from src.data.dataset_loader import load_dataset_records, load_image_frame, materialize_frame_records
from src.data.public_dataset_presets import TARGET_LABEL_SETS
from src.evaluation.metrics_classification import compute_metrics, confusion_dataframe
from src.visualization.plot_dataset_results import generate_dataset_figures


def _default_dataset_root(project_root: Path) -> Path:
    return project_root / "data" / "pilot" / "sessions" / "paper1_anchor_demo" / "frames"


def run_dataset_evaluation(
    project_root: Path,
    dataset_root: Path | None = None,
    labels_csv: Path | None = None,
    split_mode: str = "test_only",
    test_size: float = 0.2,
    random_seed: int = 42,
    output_subdir: str = "dataset_eval",
    width: int = 640,
    height: int = 480,
    target_label_set: str | None = "broad4_angry",
) -> dict[str, str]:
    paths = Paper1Paths.from_project_root(project_root)
    paths.ensure()
    dataset_root = dataset_root or _default_dataset_root(project_root)
    dataset_root = dataset_root if dataset_root.is_absolute() else (project_root / dataset_root)
    dataset_root = dataset_root.resolve()
    if labels_csv is not None and not labels_csv.is_absolute():
        labels_csv = (project_root / labels_csv).resolve()
    dataset_csv_dir = paths.outputs_csv_paper1 / output_subdir
    dataset_fig_dir = paths.outputs_figures_paper1 / output_subdir
    dataset_csv_dir.mkdir(parents=True, exist_ok=True)
    dataset_fig_dir.mkdir(parents=True, exist_ok=True)

    dataset_df = load_dataset_records(
        dataset_root=dataset_root,
        labels_csv=labels_csv.resolve() if labels_csv is not None else None,
        split_mode=split_mode,
        test_size=test_size,
        random_seed=random_seed,
        target_label_set=target_label_set,
    )
    materialized_df = materialize_frame_records(
        dataset_df,
        cache_dir=dataset_csv_dir / "materialized_frames",
        width=width,
        height=height,
    )
    test_df = materialized_df.loc[materialized_df["split"] == "test"].copy()
    if test_df.empty:
        test_df = materialized_df.copy()
        test_df["split"] = "test"

    prediction_rows: list[dict[str, object]] = []
    for row in test_df.to_dict(orient="records"):
        media_path = Path(row.get("frame_path") or row["media_path"])
        frame = load_image_frame(media_path, width=width, height=height)
        raw_pred, confidence = detect_face_emotion_from_frame(frame, enforce_detection=False)
        pred = normalize_face_emotion(raw_pred) or "unknown"
        prediction_rows.append(
            {
                "sample_id": row["sample_id"],
                "media_path": str(media_path),
                "true_label": row.get("label"),
                "predicted_label": pred,
                "confidence": confidence,
                "split": row["split"],
                "media_type": row["media_type"],
                "timestamp_ms": row.get("timestamp_ms"),
                "frame_index": row.get("frame_index"),
                "runtime_type": "offline_dataset_evaluation",
                "data_source_type": "pilot_real_anchor" if "pilot" in str(media_path).lower() else "offline_dataset_evaluation",
                "evidence_level": "benchmark_preliminary",
            }
        )

    predictions_df = pd.DataFrame(prediction_rows)
    if predictions_df.empty:
        raise RuntimeError(f"No image samples were evaluated from {dataset_root}")

    labels_available = predictions_df["true_label"].notna().any()
    metrics_rows: list[dict[str, object]] = [
        {
            "metric": "num_samples",
            "value": float(len(predictions_df)),
            "runtime_type": "offline_dataset_evaluation",
            "data_source_type": "pilot_real_anchor" if "pilot" in str(dataset_root).lower() else "offline_dataset_evaluation",
            "evidence_level": "benchmark_preliminary",
        },
        {
            "metric": "prediction_coverage_rate",
            "value": float((predictions_df["predicted_label"] != "unknown").mean()),
            "runtime_type": "offline_dataset_evaluation",
            "data_source_type": "pilot_real_anchor" if "pilot" in str(dataset_root).lower() else "offline_dataset_evaluation",
            "evidence_level": "benchmark_preliminary",
        },
        {
            "metric": "mean_confidence",
            "value": float(pd.to_numeric(predictions_df["confidence"], errors="coerce").dropna().mean()) if pd.to_numeric(predictions_df["confidence"], errors="coerce").dropna().size else None,
            "runtime_type": "offline_dataset_evaluation",
            "data_source_type": "pilot_real_anchor" if "pilot" in str(dataset_root).lower() else "offline_dataset_evaluation",
            "evidence_level": "benchmark_preliminary",
        },
        {
            "metric": "has_ground_truth_labels",
            "value": 1.0 if labels_available else 0.0,
            "runtime_type": "offline_dataset_evaluation",
            "data_source_type": "pilot_real_anchor" if "pilot" in str(dataset_root).lower() else "offline_dataset_evaluation",
            "evidence_level": "benchmark_preliminary",
        },
    ]
    if labels_available:
        labeled = predictions_df.loc[predictions_df["true_label"].notna()].copy()
        metric_obj = compute_metrics(labeled["true_label"].tolist(), labeled["predicted_label"].tolist())
        metrics_rows.extend(
            [
                {"metric": "accuracy", "value": metric_obj.accuracy, "runtime_type": "offline_dataset_evaluation", "data_source_type": "offline_dataset_evaluation", "evidence_level": "benchmark_preliminary"},
                {"metric": "macro_f1", "value": metric_obj.macro_f1, "runtime_type": "offline_dataset_evaluation", "data_source_type": "offline_dataset_evaluation", "evidence_level": "benchmark_preliminary"},
                {"metric": "weighted_f1", "value": metric_obj.weighted_f1, "runtime_type": "offline_dataset_evaluation", "data_source_type": "offline_dataset_evaluation", "evidence_level": "benchmark_preliminary"},
                {"metric": "unweighted_recall", "value": metric_obj.unweighted_recall, "runtime_type": "offline_dataset_evaluation", "data_source_type": "offline_dataset_evaluation", "evidence_level": "benchmark_preliminary"},
            ]
        )
        labels = sorted({*labeled["true_label"].dropna().unique().tolist(), *labeled["predicted_label"].unique().tolist()})
        confusion_df = confusion_dataframe(labeled["true_label"].tolist(), labeled["predicted_label"].tolist(), labels)
    else:
        confusion_df = pd.DataFrame(
            [
                {
                    "true_label": "labels_unavailable",
                    "labels_unavailable": len(predictions_df),
                    "note": "This local dataset demo runs in test-only mode without ground-truth labels.",
                }
            ]
        )

    label_distribution = predictions_df.groupby("predicted_label", as_index=False).size().rename(columns={"size": "count"})
    dataset_summary = pd.DataFrame(
        [
            {
                "dataset_root": str(dataset_root),
                "labels_csv": str(labels_csv.resolve()) if labels_csv else "",
                "split_mode": split_mode,
                "test_size": test_size,
                "random_seed": random_seed,
                "target_label_set": target_label_set or "",
                "target_labels": ",".join(sorted(TARGET_LABEL_SETS.get(target_label_set, set()))) if target_label_set else "",
                "num_test_samples": len(predictions_df),
                "num_labeled_samples": int(predictions_df["true_label"].notna().sum()),
                "media_types": ",".join(sorted(predictions_df["media_type"].unique())),
                "runtime_type": "offline_dataset_evaluation",
                "evidence_level": "benchmark_preliminary",
            }
        ]
    )
    sequence_manifest = predictions_df.sort_values(["timestamp_ms", "frame_index"], na_position="last").reset_index(drop=True)

    predictions_path = dataset_csv_dir / "dataset_predictions.csv"
    metrics_path = dataset_csv_dir / "dataset_metrics_summary.csv"
    confusion_path = dataset_csv_dir / "dataset_confusion_matrix.csv"
    summary_path = dataset_csv_dir / "dataset_summary.csv"
    label_dist_path = dataset_csv_dir / "dataset_label_distribution.csv"
    sequence_path = dataset_csv_dir / "dataset_sequence_manifest.csv"
    summary_json_path = dataset_csv_dir / "dataset_evaluation_summary.json"

    write_dataframe(predictions_path, predictions_df)
    write_dataframe(metrics_path, pd.DataFrame(metrics_rows))
    write_dataframe(confusion_path, confusion_df)
    write_dataframe(summary_path, dataset_summary)
    write_dataframe(label_dist_path, label_distribution)
    write_dataframe(sequence_path, sequence_manifest)
    write_json(
        summary_json_path,
        {
            "dataset_root": str(dataset_root),
            "labels_available": bool(labels_available),
            "num_samples": len(predictions_df),
            "target_label_set": target_label_set,
            "outputs": {
                "predictions_csv": str(predictions_path),
                "metrics_csv": str(metrics_path),
                "confusion_csv": str(confusion_path),
            },
        },
    )

    write_dataframe(project_root / "outputs" / "tables" / "paper1_table_dataset_summary.csv", dataset_summary)
    write_dataframe(project_root / "outputs" / "tables" / "paper1_table_dataset_metrics.csv", pd.DataFrame(metrics_rows))
    runtime_vs_dataset = pd.DataFrame(
        [
            {
                "evidence_type": "offline_dataset_evaluation",
                "runtime_type": "offline_dataset_evaluation",
                "camera_source": "curated dataset images or sampled video frames",
                "strength": "controlled perception testing",
                "limitation": "does not validate live runtime transport",
                "paper_role": "main controlled perception evidence",
            },
            {
                "evidence_type": "ros2_dataset_replay",
                "runtime_type": "ros2_dataset_replay",
                "camera_source": "dataset frames replayed through /camera/image_raw",
                "strength": "tests the ROS pipeline with controlled inputs",
                "limitation": "still not a live sensor run",
                "paper_role": "bridge between offline evaluation and live runtime",
            },
            {
                "evidence_type": "ros2_live_windows_stream_wsl_core",
                "runtime_type": "ros2_live_windows_stream_wsl_core",
                "camera_source": "Windows webcam streamer",
                "strength": "validates live transport and runtime integration",
                "limitation": "less controlled as a perception benchmark",
                "paper_role": "runtime integration evidence",
            },
            {
                "evidence_type": "ros2_playback_grounded",
                "runtime_type": "ros2_playback_grounded",
                "camera_source": "recorded or emulated playback",
                "strength": "repeatable pipeline validation",
                "limitation": "not live sensing",
                "paper_role": "controlled systems baseline",
            },
        ]
    )
    write_dataframe(project_root / "outputs" / "tables" / "paper1_table_runtime_vs_dataset_evidence.csv", runtime_vs_dataset)

    generate_dataset_figures(project_root=project_root, dataset_csv_dir=dataset_csv_dir, dataset_fig_dir=dataset_fig_dir)
    return {
        "predictions_csv": str(predictions_path),
        "metrics_csv": str(metrics_path),
        "confusion_csv": str(confusion_path),
        "summary_csv": str(summary_path),
        "figures_dir": str(dataset_fig_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Paper 1 offline dataset evaluation.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--labels-csv", default="")
    parser.add_argument("--split-mode", default="test_only", choices=["test_only", "train_test"])
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--output-subdir", default="dataset_eval")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--target-label-set", default="broad4_angry", choices=["broad4_angry", "paper1_legacy4", ""])
    args = parser.parse_args()
    run_dataset_evaluation(
        project_root=Path(args.project_root).resolve(),
        dataset_root=Path(args.dataset_root).resolve() if args.dataset_root else None,
        labels_csv=Path(args.labels_csv).resolve() if args.labels_csv else None,
        split_mode=args.split_mode,
        test_size=args.test_size,
        random_seed=args.random_seed,
        output_subdir=args.output_subdir,
        width=args.width,
        height=args.height,
        target_label_set=args.target_label_set or None,
    )


if __name__ == "__main__":
    main()
