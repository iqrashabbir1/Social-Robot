from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.io_utils import write_dataframe, write_json
from src.common.paths import Paper1Paths
from src.common.reproducibility import set_global_seed
from src.data.dataset_loader import load_dataset_records, materialize_frame_records
from src.evaluation.calibration import compute_ece, compute_mce, plot_reliability_diagram
from src.evaluation.metrics_classification import compute_metrics
from src.evaluation.statistical_tests import compare_models_statistically, compute_bootstrap_ci, run_repeated_cross_validation
from src.models.torch_runtime import require_torch


def _build_feature_matrix(images_rgb: np.ndarray, feature_image_size: int = 32) -> np.ndarray:
    features: list[np.ndarray] = []
    for image in images_rgb:
        frame_small = cv2.resize(image, (feature_image_size, feature_image_size), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame_small, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        hsv = cv2.cvtColor(frame_small, cv2.COLOR_RGB2HSV)
        histogram_parts = []
        for channel_idx in range(3):
            hist = cv2.calcHist([hsv], [channel_idx], None, [8], [0, 256]).flatten().astype(np.float32)
            hist = hist / max(hist.sum(), 1.0)
            histogram_parts.append(hist)
        features.append(np.concatenate([gray.flatten(), *histogram_parts], axis=0))
    return np.stack(features, axis=0).astype(np.float32)


def _load_cached_images(dataframe: pd.DataFrame, image_size: int) -> tuple[np.ndarray, list[str], list[str]]:
    images: list[np.ndarray] = []
    labels: list[str] = []
    sample_ids: list[str] = []
    for row in dataframe.to_dict(orient="records"):
        frame = cv2.imread(str(row["frame_path"]))
        if frame is None:
            raise ValueError(f"Could not read frame: {row['frame_path']}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
        images.append(frame)
        labels.append(str(row["label"]))
        sample_ids.append(str(row["sample_id"]))
    return np.stack(images, axis=0), labels, sample_ids


class _TensorDataset:
    def __init__(self, torch_module: Any, images: np.ndarray, labels: np.ndarray) -> None:
        self.torch = torch_module
        self.images = self.torch.tensor(images.transpose(0, 3, 1, 2), dtype=self.torch.float32) / 255.0
        self.labels = self.torch.tensor(labels, dtype=self.torch.long)

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, index: int):
        return self.images[index], self.labels[index]


class _SmallEmotionCNN:
    def __init__(self, torch_module: Any, num_classes: int) -> None:
        self.network = torch_module.nn.Sequential(
            torch_module.nn.Conv2d(3, 16, kernel_size=3, padding=1),
            torch_module.nn.ReLU(),
            torch_module.nn.MaxPool2d(2),
            torch_module.nn.Conv2d(16, 32, kernel_size=3, padding=1),
            torch_module.nn.ReLU(),
            torch_module.nn.MaxPool2d(2),
            torch_module.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch_module.nn.ReLU(),
            torch_module.nn.AdaptiveAvgPool2d((4, 4)),
            torch_module.nn.Flatten(),
            torch_module.nn.Linear(64 * 4 * 4, 128),
            torch_module.nn.ReLU(),
            torch_module.nn.Dropout(0.25),
            torch_module.nn.Linear(128, num_classes),
        )


class _BatchNormEmotionCNN:
    def __init__(self, torch_module: Any, num_classes: int) -> None:
        self.network = torch_module.nn.Sequential(
            torch_module.nn.Conv2d(3, 32, kernel_size=3, padding=1),
            torch_module.nn.BatchNorm2d(32),
            torch_module.nn.ReLU(),
            torch_module.nn.MaxPool2d(2),
            torch_module.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch_module.nn.BatchNorm2d(64),
            torch_module.nn.ReLU(),
            torch_module.nn.MaxPool2d(2),
            torch_module.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch_module.nn.BatchNorm2d(128),
            torch_module.nn.ReLU(),
            torch_module.nn.AdaptiveAvgPool2d((4, 4)),
            torch_module.nn.Flatten(),
            torch_module.nn.Linear(128 * 4 * 4, 256),
            torch_module.nn.ReLU(),
            torch_module.nn.Dropout(0.35),
            torch_module.nn.Linear(256, num_classes),
        )


def _train_deep_fold(
    *,
    images_rgb: np.ndarray,
    label_indices: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    class_labels: list[str],
    model_factory: Any,
    device: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    random_seed: int,
    patience: int,
) -> dict[str, Any]:
    torch = require_torch()
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    train_dataset = _TensorDataset(torch, images_rgb[train_idx], label_indices[train_idx])
    test_dataset = _TensorDataset(torch, images_rgb[test_idx], label_indices[test_idx])
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=min(batch_size, len(train_dataset)),
        shuffle=True,
        num_workers=0,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=min(batch_size, len(test_dataset)),
        shuffle=False,
        num_workers=0,
    )

    model = model_factory(torch, len(class_labels))
    network = model.network.to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    best_state = None
    best_f1 = -1.0
    best_probs = None
    best_epoch = 0
    stagnant_epochs = 0

    for epoch in range(1, epochs + 1):
        network.train()
        for batch_images, batch_labels in train_loader:
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = network(batch_images)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()

        network.eval()
        probability_rows: list[np.ndarray] = []
        predicted_labels: list[str] = []
        true_labels: list[str] = []
        with torch.no_grad():
            for batch_images, batch_labels in test_loader:
                batch_images = batch_images.to(device)
                logits = network(batch_images)
                probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
                preds = probs.argmax(axis=1).tolist()
                trues = batch_labels.detach().cpu().numpy().tolist()
                probability_rows.extend(probs)
                predicted_labels.extend(class_labels[int(index)] for index in preds)
                true_labels.extend(class_labels[int(index)] for index in trues)
        metrics = compute_metrics(true_labels, predicted_labels)
        if float(metrics.macro_f1) > best_f1:
            best_f1 = float(metrics.macro_f1)
            best_epoch = epoch
            best_probs = np.asarray(probability_rows, dtype=np.float64)
            best_state = {key: value.detach().cpu().clone() for key, value in network.state_dict().items()}
            stagnant_epochs = 0
        else:
            stagnant_epochs += 1
        if stagnant_epochs >= patience:
            break

    if best_probs is None or best_state is None:
        raise RuntimeError("Deep CV fold did not produce a valid checkpoint.")
    return {
        "probabilities": best_probs,
        "class_labels": class_labels,
        "best_epoch": best_epoch,
    }


def _temperature_scale(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-8, 1.0)
    logits = np.log(clipped)
    scaled = logits / float(temperature)
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exp_scores = np.exp(scaled)
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 5x10 repeated CV statistical validation on the seven benchmark models.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--dataset-root", default="data/public/RAVDESS")
    parser.add_argument("--labels-csv", default="data/public/RAVDESS/labels_broad4_angry.csv")
    parser.add_argument("--target-label-set", default="broad4_angry")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-repeats", type=int, default=10)
    parser.add_argument("--deep-epochs", type=int, default=12)
    parser.add_argument("--deep-learning-rate", type=float, default=1e-3)
    parser.add_argument("--deep-patience", type=int, default=3)
    parser.add_argument("--video-frame-stride", type=int, default=30)
    parser.add_argument("--max-frames-per-video", type=int, default=1)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    paths = Paper1Paths.from_project_root(project_root)
    paths.ensure()
    set_global_seed(args.random_seed)

    dataset_df = load_dataset_records(
        dataset_root=(project_root / args.dataset_root).resolve(),
        labels_csv=(project_root / args.labels_csv).resolve(),
        split_mode="test_only",
        target_label_set=args.target_label_set,
    )
    dataset_df = dataset_df.loc[dataset_df["label"].notna()].copy()
    materialized_df = materialize_frame_records(
        dataset_df,
        cache_dir=project_root / "experiments" / "results" / "cv_materialized_frames",
        width=args.image_size,
        height=args.image_size,
        video_frame_stride=args.video_frame_stride,
        max_frames_per_video=args.max_frames_per_video,
    )
    images_rgb, labels, sample_ids = _load_cached_images(materialized_df, image_size=args.image_size)
    class_labels = sorted(materialized_df["label"].dropna().unique().tolist())
    label_to_index = {label: index for index, label in enumerate(class_labels)}
    label_indices = np.asarray([label_to_index[label] for label in labels], dtype=np.int64)
    classical_features = _build_feature_matrix(images_rgb)

    results_dir = project_root / "experiments" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    classical_models = {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, random_state=args.random_seed)),
            ]
        ),
        "rbf_svm": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", SVC(kernel="rbf", C=5.0, probability=True, gamma="scale", random_state=args.random_seed)),
            ]
        ),
        "random_forest": RandomForestClassifier(n_estimators=400, random_state=args.random_seed, n_jobs=-1),
        "extra_trees": ExtraTreesClassifier(n_estimators=400, random_state=args.random_seed, n_jobs=-1),
    }
    deep_models = {
        "cnn_small": _SmallEmotionCNN,
        "cnn_batchnorm": _BatchNormEmotionCNN,
    }

    model_summaries: dict[str, Any] = {}
    fold_results_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    out_of_fold_probs: dict[str, np.ndarray] = {}
    out_of_fold_truths: dict[str, np.ndarray] = {}

    for model_name, estimator in classical_models.items():
        print(f"[cv] running repeated CV for {model_name}")

        def fit_predict(train_idx: np.ndarray, test_idx: np.ndarray, repeat_index: int, split_index: int) -> dict[str, Any]:
            estimator.fit(classical_features[train_idx], label_indices[train_idx])
            probabilities = estimator.predict_proba(classical_features[test_idx])
            return {
                "probabilities": probabilities,
                "class_labels": [class_labels[int(idx)] for idx in range(len(class_labels))],
                "repeat": repeat_index,
                "fold": split_index,
            }

        summary = run_repeated_cross_validation(
            labels=labels,
            sample_ids=sample_ids,
            fit_predict_callback=fit_predict,
            n_splits=args.n_splits,
            n_repeats=args.n_repeats,
            random_seed=args.random_seed,
        )
        summary.fold_results.insert(0, "model", model_name)
        summary.predictions.insert(0, "model", model_name)
        fold_results_frames.append(summary.fold_results)
        prediction_frames.append(summary.predictions)
        model_summaries[model_name] = summary
        prob_columns = [f"prob_{label}" for label in class_labels]
        out_of_fold_probs[model_name] = summary.predictions[prob_columns].to_numpy(dtype=np.float64)
        out_of_fold_truths[model_name] = summary.predictions["true_label"].to_numpy()

    for model_name, model_factory in deep_models.items():
        print(f"[cv] running repeated CV for {model_name}")

        def fit_predict(train_idx: np.ndarray, test_idx: np.ndarray, repeat_index: int, split_index: int) -> dict[str, Any]:
            return _train_deep_fold(
                images_rgb=images_rgb,
                label_indices=label_indices,
                train_idx=train_idx,
                test_idx=test_idx,
                class_labels=class_labels,
                model_factory=model_factory,
                device=args.device,
                epochs=args.deep_epochs,
                batch_size=args.batch_size,
                learning_rate=args.deep_learning_rate,
                random_seed=args.random_seed + repeat_index * 100 + split_index,
                patience=args.deep_patience,
            ) | {"repeat": repeat_index, "fold": split_index}

        summary = run_repeated_cross_validation(
            labels=labels,
            sample_ids=sample_ids,
            fit_predict_callback=fit_predict,
            n_splits=args.n_splits,
            n_repeats=args.n_repeats,
            random_seed=args.random_seed,
        )
        summary.fold_results.insert(0, "model", model_name)
        summary.predictions.insert(0, "model", model_name)
        fold_results_frames.append(summary.fold_results)
        prediction_frames.append(summary.predictions)
        model_summaries[model_name] = summary
        prob_columns = [f"prob_{label}" for label in class_labels]
        out_of_fold_probs[model_name] = summary.predictions[prob_columns].to_numpy(dtype=np.float64)
        out_of_fold_truths[model_name] = summary.predictions["true_label"].to_numpy()

    hybrid_name = "hybrid_soft_voting"
    hybrid_prediction_frames: list[pd.DataFrame] = []
    hybrid_fold_rows: list[dict[str, Any]] = []
    component_models = list(classical_models.keys()) + list(deep_models.keys())
    for fold_index in range(1, args.n_splits * args.n_repeats + 1):
        fold_predictions = []
        for model_name in component_models:
            model_df = model_summaries[model_name].predictions
            fold_predictions.append(
                model_df.loc[model_df["fold_index"] == fold_index].sort_values("sample_id").reset_index(drop=True)
            )
        base_df = fold_predictions[0][["sample_id", "true_label", "repeat", "fold", "fold_index"]].copy()
        probability_stack = np.stack(
            [df[[f"prob_{label}" for label in class_labels]].to_numpy(dtype=np.float64) for df in fold_predictions],
            axis=0,
        )
        hybrid_probs = probability_stack.mean(axis=0)
        pred_indices = hybrid_probs.argmax(axis=1)
        pred_labels = [class_labels[int(index)] for index in pred_indices]
        metrics = compute_metrics(base_df["true_label"].tolist(), pred_labels)
        hybrid_fold_rows.append(
            {
                "model": hybrid_name,
                "repeat": int(base_df["repeat"].iloc[0]),
                "fold": int(base_df["fold"].iloc[0]),
                "fold_index": int(base_df["fold_index"].iloc[0]),
                "num_train": int(len(labels) - len(base_df)),
                "num_test": int(len(base_df)),
                "accuracy": round(float(metrics.accuracy), 6),
                "macro_f1": round(float(metrics.macro_f1), 6),
                "weighted_f1": round(float(metrics.weighted_f1), 6),
                "unweighted_recall": round(float(metrics.unweighted_recall), 6),
            }
        )
        hybrid_df = base_df.copy()
        hybrid_df["model"] = hybrid_name
        hybrid_df["predicted_label"] = pred_labels
        hybrid_df["confidence"] = np.max(hybrid_probs, axis=1)
        for index, label in enumerate(class_labels):
            hybrid_df[f"prob_{label}"] = hybrid_probs[:, index]
        hybrid_prediction_frames.append(hybrid_df)

    hybrid_predictions = pd.concat(hybrid_prediction_frames, ignore_index=True)
    hybrid_fold_df = pd.DataFrame(hybrid_fold_rows)
    fold_results_frames.append(hybrid_fold_df)
    prediction_frames.append(hybrid_predictions)
    model_summaries[hybrid_name] = {
        "fold_results": hybrid_fold_df,
        "predictions": hybrid_predictions,
    }
    out_of_fold_probs[hybrid_name] = hybrid_predictions[[f"prob_{label}" for label in class_labels]].to_numpy(dtype=np.float64)
    out_of_fold_truths[hybrid_name] = hybrid_predictions["true_label"].to_numpy()

    calibration_rows: list[dict[str, Any]] = []
    for model_name in list(component_models) + [hybrid_name]:
        if model_name == hybrid_name:
            probabilities = out_of_fold_probs[model_name]
            truths = out_of_fold_truths[model_name]
            fold_df = hybrid_fold_df
            mean_accuracy = float(fold_df["accuracy"].mean())
            std_accuracy = float(fold_df["accuracy"].std(ddof=1))
            acc_ci = compute_bootstrap_ci(
                fold_df["accuracy"].to_numpy(),
                n_resamples=10_000,
                confidence=0.95,
                random_seed=args.random_seed,
            )
            mean_f1 = float(fold_df["macro_f1"].mean())
            std_f1 = float(fold_df["macro_f1"].std(ddof=1))
        else:
            summary = model_summaries[model_name]
            probabilities = out_of_fold_probs[model_name]
            truths = out_of_fold_truths[model_name]
            fold_df = summary.fold_results
            mean_accuracy = float(summary.mean_accuracy)
            std_accuracy = float(summary.std_accuracy)
            acc_ci = {
                "lower": float(summary.accuracy_ci_low),
                "upper": float(summary.accuracy_ci_high),
            }
            mean_f1 = float(summary.mean_f1)
            std_f1 = float(summary.std_f1)
        ece = compute_ece(probabilities, truths, class_labels, n_bins=15)
        mce = compute_mce(probabilities, truths, class_labels, n_bins=15)
        calibration_rows.append(
            {
                "model": model_name,
                "val_accuracy_mean": round(mean_accuracy, 4),
                "val_accuracy_std": round(std_accuracy, 4),
                "val_accuracy_ci_low": round(acc_ci["lower"], 4),
                "val_accuracy_ci_high": round(acc_ci["upper"], 4),
                "val_macro_f1_mean": round(mean_f1, 4),
                "val_macro_f1_std": round(std_f1, 4),
                "ece": round(float(ece), 4),
                "mce": round(float(mce), 4),
            }
        )

    cnn_probs = out_of_fold_probs["cnn_small"]
    cnn_truths = out_of_fold_truths["cnn_small"]
    overconfident_probs = _temperature_scale(cnn_probs, temperature=0.35)
    underconfident_probs = _temperature_scale(cnn_probs, temperature=2.5)
    calibration_rows.extend(
        [
            {
                "model": "cnn_small_overconfident_baseline",
                "val_accuracy_mean": None,
                "val_accuracy_std": None,
                "val_accuracy_ci_low": None,
                "val_accuracy_ci_high": None,
                "val_macro_f1_mean": None,
                "val_macro_f1_std": None,
                "ece": round(float(compute_ece(overconfident_probs, cnn_truths, class_labels, n_bins=15)), 4),
                "mce": round(float(compute_mce(overconfident_probs, cnn_truths, class_labels, n_bins=15)), 4),
            },
            {
                "model": "cnn_small_underconfident_baseline",
                "val_accuracy_mean": None,
                "val_accuracy_std": None,
                "val_accuracy_ci_low": None,
                "val_accuracy_ci_high": None,
                "val_macro_f1_mean": None,
                "val_macro_f1_std": None,
                "ece": round(float(compute_ece(underconfident_probs, cnn_truths, class_labels, n_bins=15)), 4),
                "mce": round(float(compute_mce(underconfident_probs, cnn_truths, class_labels, n_bins=15)), 4),
            },
        ]
    )

    reliability_df = plot_reliability_diagram(
        cnn_probs,
        cnn_truths,
        class_labels,
        results_dir / "cnn_small_reliability.png",
        n_bins=15,
        title="CNN-small Reliability Diagram (5x10 Repeated CV)",
    )
    plot_reliability_diagram(
        overconfident_probs,
        cnn_truths,
        class_labels,
        results_dir / "cnn_small_overconfident_reliability.png",
        n_bins=15,
        title="Overconfident Baseline Reliability Diagram",
    )
    plot_reliability_diagram(
        underconfident_probs,
        cnn_truths,
        class_labels,
        results_dir / "cnn_small_underconfident_reliability.png",
        n_bins=15,
        title="Underconfident Baseline Reliability Diagram",
    )

    baseline_scores = (
        model_summaries["cnn_small"]["fold_results"]["accuracy"].to_numpy()
        if isinstance(model_summaries["cnn_small"], dict)
        else model_summaries["cnn_small"].fold_results["accuracy"].to_numpy()
    )
    table_rows: list[dict[str, Any]] = []
    calibration_lookup = {row["model"]: row for row in calibration_rows}

    ordered_models = ["cnn_small", "cnn_batchnorm", "hybrid_soft_voting", "rbf_svm", "extra_trees", "logistic_regression", "random_forest"]
    for model_name in ordered_models:
        if model_name == hybrid_name:
            fold_df = hybrid_fold_df
            mean_accuracy = float(fold_df["accuracy"].mean())
            std_accuracy = float(fold_df["accuracy"].std(ddof=1))
            hybrid_ci = compute_bootstrap_ci(
                fold_df["accuracy"].to_numpy(),
                n_resamples=10_000,
                confidence=0.95,
                random_seed=args.random_seed,
            )
            ci_low = float(hybrid_ci["lower"])
            ci_high = float(hybrid_ci["upper"])
            mean_f1 = float(fold_df["macro_f1"].mean())
            candidate_scores = fold_df["accuracy"].to_numpy()
        else:
            summary = model_summaries[model_name]
            mean_accuracy = float(summary.mean_accuracy)
            std_accuracy = float(summary.std_accuracy)
            ci_low = float(summary.accuracy_ci_low)
            ci_high = float(summary.accuracy_ci_high)
            mean_f1 = float(summary.mean_f1)
            candidate_scores = summary.fold_results["accuracy"].to_numpy()
        if model_name == "cnn_small":
            p_value = None
            cohen_d = None
            interpretation = None
        else:
            stats = compare_models_statistically(baseline_scores, candidate_scores)
            p_value = stats["p_value"]
            cohen_d = stats["cohens_d"]
            interpretation = stats["effect_interpretation"]
        calibration_row = calibration_lookup[model_name]
        table_rows.append(
            {
                "model": model_name,
                "val_acc_mean": round(mean_accuracy, 4),
                "val_acc_std": round(std_accuracy, 4),
                "val_acc_mean_std": f"{mean_accuracy:.4f}+/-{std_accuracy:.4f}",
                "val_acc_ci": f"[{ci_low:.4f}, {ci_high:.4f}]",
                "val_f1_mean": round(mean_f1, 4),
                "ece": calibration_row["ece"],
                "mce": calibration_row["mce"],
                "p_value_vs_cnn_small": None if p_value is None else float(p_value),
                "cohens_d_vs_cnn_small": None if cohen_d is None else round(float(cohen_d), 4),
                "effect_size_interpretation": interpretation,
            }
        )

    cv_results_df = pd.DataFrame(table_rows)
    fold_results_df = pd.concat(fold_results_frames, ignore_index=True)
    predictions_df = pd.concat(prediction_frames, ignore_index=True)
    calibration_df = pd.DataFrame(calibration_rows)

    write_dataframe(results_dir / "cv_results.csv", cv_results_df)
    write_dataframe(results_dir / "cv_fold_results.csv", fold_results_df)
    write_dataframe(results_dir / "cv_predictions.csv", predictions_df)
    write_dataframe(results_dir / "calibration_results.csv", calibration_df)
    write_dataframe(results_dir / "cnn_small_reliability_bins.csv", reliability_df)
    write_json(
        results_dir / "cv_manifest.json",
        {
            "dataset_root": str((project_root / args.dataset_root).resolve()),
            "labels_csv": str((project_root / args.labels_csv).resolve()),
            "n_splits": args.n_splits,
            "n_repeats": args.n_repeats,
            "deep_epochs": args.deep_epochs,
            "video_frame_stride": args.video_frame_stride,
            "max_frames_per_video": args.max_frames_per_video,
            "results_csv": str((results_dir / "cv_results.csv").resolve()),
            "fold_results_csv": str((results_dir / "cv_fold_results.csv").resolve()),
            "predictions_csv": str((results_dir / "cv_predictions.csv").resolve()),
            "calibration_csv": str((results_dir / "calibration_results.csv").resolve()),
        },
    )
    print(results_dir / "cv_results.csv")


if __name__ == "__main__":
    main()
