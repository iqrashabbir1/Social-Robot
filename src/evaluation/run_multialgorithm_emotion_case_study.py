from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.common.io_utils import write_dataframe, write_json
from src.common.paths import Paper1Paths
from src.common.reproducibility import set_global_seed
from src.data.dataset_loader import load_dataset_records, materialize_frame_records
from src.evaluation.metrics_classification import compute_metrics, confusion_dataframe
from src.models.torch_runtime import require_torch
from src.models.vision.image_emotion_classifier import FrameEmotionDataset, frame_collate_fn


def _write_progress_snapshot(progress_dir: Path, status_payload: dict[str, object], progress_rows: list[dict[str, object]]) -> None:
    progress_dir.mkdir(parents=True, exist_ok=True)
    write_json(progress_dir / "latest_status.json", status_payload)
    if progress_rows:
        write_dataframe(progress_dir / "training_progress_latest.csv", pd.DataFrame(progress_rows))


def _extract_feature_matrix(dataframe: pd.DataFrame, image_size: int = 32) -> np.ndarray:
    features: list[np.ndarray] = []
    for row in dataframe.to_dict(orient="records"):
        frame = cv2.imread(str(row["frame_path"]))
        if frame is None:
            raise ValueError(f"Could not read frame: {row['frame_path']}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_small = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame_small, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        hsv = cv2.cvtColor(frame_small, cv2.COLOR_RGB2HSV)
        histogram_parts = []
        for channel_idx in range(3):
            hist = cv2.calcHist([hsv], [channel_idx], None, [8], [0, 256]).flatten().astype(np.float32)
            hist = hist / max(hist.sum(), 1.0)
            histogram_parts.append(hist)
        feature = np.concatenate([gray.flatten(), *histogram_parts], axis=0)
        features.append(feature)
    return np.stack(features, axis=0)


def _metrics_row(
    algorithm_name: str,
    family: str,
    evaluation_stage: str,
    dataset_name: str,
    metrics: dict[str, float],
    num_samples: int,
    epochs: int | None = None,
    is_best_validation: bool = False,
) -> dict[str, object]:
    return {
        "algorithm_name": algorithm_name,
        "family": family,
        "evaluation_stage": evaluation_stage,
        "dataset_name": dataset_name,
        "num_samples": int(num_samples),
        "accuracy": round(float(metrics["accuracy"]), 4),
        "macro_f1": round(float(metrics["macro_f1"]), 4),
        "weighted_f1": round(float(metrics["weighted_f1"]), 4),
        "unweighted_recall": round(float(metrics["unweighted_recall"]), 4),
        "epochs": epochs if epochs is not None else "",
        "is_best_validation_model": bool(is_best_validation),
    }


class BatchNormEmotionCNN:
    def __init__(self, torch_module, num_classes: int) -> None:
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


class SmallEmotionCNN:
    def __init__(self, torch_module, num_classes: int) -> None:
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


def _evaluate_probabilities(probabilities: np.ndarray, y_true: list[str], labels: list[str]) -> tuple[dict[str, float], pd.DataFrame, list[str]]:
    pred_idx = probabilities.argmax(axis=1).tolist()
    y_pred = [labels[int(index)] for index in pred_idx]
    scores = compute_metrics(y_true, y_pred)
    metrics = {
        "accuracy": scores.accuracy,
        "macro_f1": scores.macro_f1,
        "weighted_f1": scores.weighted_f1,
        "unweighted_recall": scores.unweighted_recall,
    }
    return metrics, confusion_dataframe(y_true, y_pred, labels), y_pred


def _train_deep_model(
    algorithm_name: str,
    model_factory,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    class_labels: list[str],
    image_size: int,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    device: str,
    log_every_epochs: int,
    progress_dir: Path,
    progress_rows: list[dict[str, object]],
) -> tuple[list[dict[str, object]], dict[str, object], pd.DataFrame]:
    torch = require_torch()
    label_to_index = {label: index for index, label in enumerate(class_labels)}
    train_dataset = FrameEmotionDataset(train_df, label_to_index, image_size=image_size)
    val_dataset = FrameEmotionDataset(val_df, label_to_index, image_size=image_size)
    test_dataset = FrameEmotionDataset(test_df, label_to_index, image_size=image_size)

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
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=min(batch_size, len(test_dataset)),
        shuffle=False,
        collate_fn=frame_collate_fn,
    )

    model = model_factory(torch, len(class_labels))
    model.network.to(device)
    optimizer = torch.optim.Adam(model.network.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    history_rows: list[dict[str, object]] = []
    best_state = None
    best_payload: dict[str, object] | None = None
    best_val_macro_f1 = -1.0

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        model.network.train()
        train_true: list[str] = []
        train_pred: list[str] = []
        train_losses: list[float] = []

        for step_idx, (features, targets, _) in enumerate(train_loader, start=1):
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
            if step_idx == 1 or step_idx == len(train_loader):
                print(
                    f"[{algorithm_name}] epoch {epoch}/{epochs} step {step_idx}/{len(train_loader)} "
                    f"loss={float(loss.detach().cpu().item()):.4f}"
                )

        model.network.eval()

        def collect_predictions(loader):
            rows = []
            y_true = []
            probabilities = []
            with torch.no_grad():
                for features, targets, meta_rows in loader:
                    features = features.to(device)
                    logits = model.network(features)
                    probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
                    target_idx = targets.detach().cpu().numpy().tolist()
                    for row_index, target_class in zip(range(len(target_idx)), target_idx):
                        y_true.append(class_labels[int(target_class)])
                        probabilities.append(probs[row_index])
                        rows.append(meta_rows[row_index])
            return rows, y_true, np.asarray(probabilities)

        val_meta_rows, val_true, val_probs = collect_predictions(val_loader)
        train_metrics = compute_metrics(train_true, train_pred)
        val_metrics, _, val_pred = _evaluate_probabilities(val_probs, val_true, class_labels)
        epoch_seconds = time.perf_counter() - epoch_start
        history_rows.append(
            {
                "algorithm_name": algorithm_name,
                "epoch": epoch,
                "train_loss": round(sum(train_losses) / max(len(train_losses), 1), 6),
                "train_accuracy": round(float(train_metrics.accuracy), 4),
                "train_macro_f1": round(float(train_metrics.macro_f1), 4),
                "val_accuracy": round(float(val_metrics["accuracy"]), 4),
                "val_macro_f1": round(float(val_metrics["macro_f1"]), 4),
                "epoch_seconds": round(epoch_seconds, 3),
            }
        )
        progress_rows.append(
            {
                "algorithm_name": algorithm_name,
                "family": "deep",
                "epoch": epoch,
                "total_epochs": epochs,
                "train_loss": round(sum(train_losses) / max(len(train_losses), 1), 6),
                "train_accuracy": round(float(train_metrics.accuracy), 4),
                "train_macro_f1": round(float(train_metrics.macro_f1), 4),
                "val_accuracy": round(float(val_metrics["accuracy"]), 4),
                "val_macro_f1": round(float(val_metrics["macro_f1"]), 4),
                "epoch_seconds": round(epoch_seconds, 3),
                "device": device,
            }
        )
        _write_progress_snapshot(
            progress_dir,
            {
                "run_type": "multialgorithm_case_study",
                "current_algorithm": algorithm_name,
                "family": "deep",
                "epoch": epoch,
                "total_epochs": epochs,
                "train_accuracy": round(float(train_metrics.accuracy), 4),
                "val_accuracy": round(float(val_metrics["accuracy"]), 4),
                "val_macro_f1": round(float(val_metrics["macro_f1"]), 4),
                "status": "running",
            },
            progress_rows,
        )
        if epoch == 1 or epoch % max(log_every_epochs, 1) == 0 or epoch == epochs:
            print(
                f"[{algorithm_name}] epoch {epoch}/{epochs} "
                f"train_acc={train_metrics.accuracy:.4f} "
                f"val_acc={val_metrics['accuracy']:.4f} val_macro_f1={val_metrics['macro_f1']:.4f}"
            )
        if val_metrics["macro_f1"] >= best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.network.state_dict().items()}
            best_payload = {
                "train_metrics": {
                    "accuracy": float(train_metrics.accuracy),
                    "macro_f1": float(train_metrics.macro_f1),
                "weighted_f1": float(train_metrics.weighted_f1),
                "unweighted_recall": float(train_metrics.unweighted_recall),
            },
                "val_metrics": val_metrics,
                "val_probs": val_probs,
                "val_true": val_true,
                "val_pred": val_pred,
                "val_meta_rows": val_meta_rows,
                "best_epoch": epoch,
            }

    if best_state is None or best_payload is None:
        raise RuntimeError(f"{algorithm_name} did not produce a valid deep-model checkpoint.")

    model.network.load_state_dict(best_state)
    model.network.to(device)
    model.network.eval()

    test_meta_rows = []
    test_true: list[str] = []
    test_probs_rows: list[np.ndarray] = []
    with torch.no_grad():
        for features, targets, meta_rows in test_loader:
            features = features.to(device)
            logits = model.network(features)
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
            target_idx = targets.detach().cpu().numpy().tolist()
            for row_index, target_class in zip(range(len(target_idx)), target_idx):
                test_true.append(class_labels[int(target_class)])
                test_probs_rows.append(probs[row_index])
                test_meta_rows.append(meta_rows[row_index])
    test_probs = np.asarray(test_probs_rows)
    test_metrics, _, test_pred = _evaluate_probabilities(test_probs, test_true, class_labels)
    best_payload["test_metrics"] = test_metrics
    best_payload["test_probs"] = test_probs
    best_payload["test_true"] = test_true
    best_payload["test_pred"] = test_pred
    best_payload["test_meta_rows"] = test_meta_rows

    predictions_rows = []
    for meta, true_label, pred_label, probs in zip(
        best_payload["test_meta_rows"],
        best_payload["test_true"],
        best_payload["test_pred"],
        best_payload["test_probs"],
    ):
        predictions_rows.append(
            {
                "algorithm_name": algorithm_name,
                "sample_id": meta["sample_id"],
                "frame_path": meta["frame_path"],
                "true_label": true_label,
                "predicted_label": pred_label,
                "confidence": round(float(np.max(probs)), 6),
                "evaluation_stage": "external_public_test",
            }
        )
    return history_rows, best_payload, pd.DataFrame(predictions_rows)


def run_multialgorithm_emotion_case_study(
    project_root: Path,
    train_dataset_root: Path,
    train_labels_csv: Path,
    external_dataset_root: Path,
    external_labels_csv: Path,
    output_subdir: str = "ravdess_multialgorithm_case_study",
    target_label_set: str | None = "broad4_angry",
    deep_epochs: int = 120,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    image_size: int = 128,
    random_seed: int = 42,
    device: str = "cuda",
    log_every_epochs: int = 10,
) -> dict[str, str]:
    set_global_seed(random_seed)
    paths = Paper1Paths.from_project_root(project_root.resolve())
    paths.ensure()

    output_dir = paths.outputs_csv_paper1 / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_dir = project_root / "outputs" / "logs" / "paper1" / output_subdir
    progress_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_dataset_records(
        dataset_root=(project_root / train_dataset_root).resolve() if not train_dataset_root.is_absolute() else train_dataset_root.resolve(),
        labels_csv=(project_root / train_labels_csv).resolve() if not train_labels_csv.is_absolute() else train_labels_csv.resolve(),
        split_mode="train_test",
        test_size=0.2,
        random_seed=random_seed,
        target_label_set=target_label_set,
    )
    train_df = train_df.loc[train_df["label"].notna()].copy()
    train_df = materialize_frame_records(
        train_df,
        cache_dir=output_dir / "ravdess_materialized_frames",
        width=image_size,
        height=image_size,
    )
    external_df = load_dataset_records(
        dataset_root=(project_root / external_dataset_root).resolve() if not external_dataset_root.is_absolute() else external_dataset_root.resolve(),
        labels_csv=(project_root / external_labels_csv).resolve() if not external_labels_csv.is_absolute() else external_labels_csv.resolve(),
        split_mode="test_only",
        target_label_set=target_label_set,
    )
    external_df = external_df.loc[external_df["label"].notna()].copy()
    external_df = materialize_frame_records(
        external_df,
        cache_dir=output_dir / "external_materialized_frames",
        width=image_size,
        height=image_size,
    )

    train_split_df = train_df.loc[train_df["split"] == "train"].copy()
    val_split_df = train_df.loc[train_df["split"] == "test"].copy()
    test_split_df = external_df.copy()
    class_labels = sorted(train_split_df["label"].dropna().unique().tolist())

    X_train = _extract_feature_matrix(train_split_df)
    X_val = _extract_feature_matrix(val_split_df)
    X_test = _extract_feature_matrix(test_split_df)
    y_train = train_split_df["label"].tolist()
    y_val = val_split_df["label"].tolist()
    y_test = test_split_df["label"].tolist()

    classical_models = {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000)),
            ]
        ),
        "rbf_svm": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", SVC(kernel="rbf", C=5.0, probability=True, gamma="scale")),
            ]
        ),
        "random_forest": RandomForestClassifier(n_estimators=400, random_state=random_seed, n_jobs=-1),
        "extra_trees": ExtraTreesClassifier(n_estimators=400, random_state=random_seed, n_jobs=-1),
    }

    summary_rows: list[dict[str, object]] = []
    history_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    progress_rows: list[dict[str, object]] = []
    validation_probabilities: dict[str, np.ndarray] = {}
    external_probabilities: dict[str, np.ndarray] = {}
    best_classical_name = None
    best_classical_val_macro_f1 = -1.0
    best_deep_name = None
    best_deep_val_macro_f1 = -1.0

    for algorithm_name, estimator in classical_models.items():
        print(f"[benchmark] training {algorithm_name}")
        _write_progress_snapshot(
            progress_dir,
            {
                "run_type": "multialgorithm_case_study",
                "current_algorithm": algorithm_name,
                "family": "classical",
                "status": "running",
            },
            progress_rows,
        )
        started = time.perf_counter()
        estimator.fit(X_train, y_train)
        train_probs = estimator.predict_proba(X_train)
        val_probs = estimator.predict_proba(X_val)
        test_probs = estimator.predict_proba(X_test)

        train_metrics, _, train_pred = _evaluate_probabilities(train_probs, y_train, estimator.classes_.tolist())
        val_metrics, _, val_pred = _evaluate_probabilities(val_probs, y_val, estimator.classes_.tolist())
        test_metrics, _, test_pred = _evaluate_probabilities(test_probs, y_test, estimator.classes_.tolist())
        duration = time.perf_counter() - started
        summary_rows.extend(
            [
                _metrics_row(algorithm_name, "classical", "train_split", "RAVDESS_train", train_metrics, len(y_train)),
                _metrics_row(algorithm_name, "classical", "held_out_validation", "RAVDESS_validation", val_metrics, len(y_val)),
                _metrics_row(algorithm_name, "classical", "external_public_test", "CREMA-D", test_metrics, len(y_test)),
            ]
        )
        history_rows.append(
            {
                "algorithm_name": algorithm_name,
                "family": "classical",
                "duration_seconds": round(duration, 3),
                "val_macro_f1": round(val_metrics["macro_f1"], 4),
                "test_macro_f1": round(test_metrics["macro_f1"], 4),
            }
        )
        progress_rows.append(
            {
                "algorithm_name": algorithm_name,
                "family": "classical",
                "epoch": "",
                "total_epochs": "",
                "train_loss": "",
                "train_accuracy": round(float(train_metrics["accuracy"]), 4),
                "train_macro_f1": round(float(train_metrics["macro_f1"]), 4),
                "val_accuracy": round(float(val_metrics["accuracy"]), 4),
                "val_macro_f1": round(float(val_metrics["macro_f1"]), 4),
                "epoch_seconds": round(duration, 3),
                "device": "cpu_or_native",
            }
        )
        _write_progress_snapshot(
            progress_dir,
            {
                "run_type": "multialgorithm_case_study",
                "current_algorithm": algorithm_name,
                "family": "classical",
                "status": "completed",
                "val_accuracy": round(float(val_metrics["accuracy"]), 4),
                "val_macro_f1": round(float(val_metrics["macro_f1"]), 4),
                "external_accuracy": round(float(test_metrics["accuracy"]), 4),
                "external_macro_f1": round(float(test_metrics["macro_f1"]), 4),
            },
            progress_rows,
        )
        prediction_frames.append(
            pd.DataFrame(
                {
                    "algorithm_name": algorithm_name,
                    "sample_id": test_split_df["sample_id"].tolist(),
                    "frame_path": test_split_df["frame_path"].tolist(),
                    "true_label": y_test,
                    "predicted_label": test_pred,
                    "confidence": np.max(test_probs, axis=1),
                    "evaluation_stage": "external_public_test",
                }
            )
        )
        validation_probabilities[algorithm_name] = np.asarray(val_probs)
        external_probabilities[algorithm_name] = np.asarray(test_probs)
        if val_metrics["macro_f1"] > best_classical_val_macro_f1:
            best_classical_val_macro_f1 = val_metrics["macro_f1"]
            best_classical_name = algorithm_name

    deep_models = {
        "cnn_small": SmallEmotionCNN,
        "cnn_batchnorm": BatchNormEmotionCNN,
    }
    for algorithm_name, model_factory in deep_models.items():
        print(f"[benchmark] training {algorithm_name}")
        deep_history, best_payload, pred_df = _train_deep_model(
            algorithm_name=algorithm_name,
            model_factory=model_factory,
            train_df=train_split_df,
            val_df=val_split_df,
            test_df=test_split_df,
            class_labels=class_labels,
            image_size=image_size,
            batch_size=batch_size,
            epochs=deep_epochs,
            learning_rate=learning_rate,
            device=device,
            log_every_epochs=log_every_epochs,
            progress_dir=progress_dir,
            progress_rows=progress_rows,
        )
        history_rows.extend(deep_history)
        prediction_frames.append(pred_df)
        summary_rows.extend(
            [
                _metrics_row(
                    algorithm_name,
                    "deep",
                    "train_split",
                    "RAVDESS_train",
                    best_payload["train_metrics"],
                    len(train_split_df),
                    epochs=deep_epochs,
                ),
                _metrics_row(
                    algorithm_name,
                    "deep",
                    "held_out_validation",
                    "RAVDESS_validation",
                    best_payload["val_metrics"],
                    len(val_split_df),
                    epochs=best_payload["best_epoch"],
                ),
                _metrics_row(
                    algorithm_name,
                    "deep",
                    "external_public_test",
                    "CREMA-D",
                    best_payload["test_metrics"],
                    len(test_split_df),
                    epochs=best_payload["best_epoch"],
                ),
            ]
        )
        validation_probabilities[algorithm_name] = np.asarray(best_payload["val_probs"])
        external_probabilities[algorithm_name] = np.asarray(best_payload["test_probs"])
        if best_payload["val_metrics"]["macro_f1"] > best_deep_val_macro_f1:
            best_deep_val_macro_f1 = best_payload["val_metrics"]["macro_f1"]
            best_deep_name = algorithm_name

    if best_classical_name and best_deep_name:
        hybrid_name = "hybrid_soft_voting"
        hybrid_val_probs = (validation_probabilities[best_classical_name] + validation_probabilities[best_deep_name]) / 2.0
        hybrid_test_probs = (external_probabilities[best_classical_name] + external_probabilities[best_deep_name]) / 2.0
        val_metrics, _, _ = _evaluate_probabilities(hybrid_val_probs, y_val, class_labels)
        test_metrics, _, test_pred = _evaluate_probabilities(hybrid_test_probs, y_test, class_labels)
        train_metrics = {"accuracy": np.nan, "macro_f1": np.nan, "weighted_f1": np.nan, "unweighted_recall": np.nan}
        summary_rows.extend(
            [
                _metrics_row(hybrid_name, "hybrid", "train_split", "RAVDESS_train", train_metrics, len(y_train)),
                _metrics_row(hybrid_name, "hybrid", "held_out_validation", "RAVDESS_validation", val_metrics, len(y_val)),
                _metrics_row(hybrid_name, "hybrid", "external_public_test", "CREMA-D", test_metrics, len(y_test)),
            ]
        )
        prediction_frames.append(
            pd.DataFrame(
                {
                    "algorithm_name": hybrid_name,
                    "sample_id": test_split_df["sample_id"].tolist(),
                    "frame_path": test_split_df["frame_path"].tolist(),
                    "true_label": y_test,
                    "predicted_label": test_pred,
                    "confidence": np.max(hybrid_test_probs, axis=1),
                    "evaluation_stage": "external_public_test",
                }
            )
        )

    summary_df = pd.DataFrame(summary_rows)
    validation_rows = summary_df.loc[summary_df["evaluation_stage"] == "held_out_validation"].copy()
    best_algorithm_name = (
        validation_rows.sort_values(["macro_f1", "accuracy"], ascending=False).iloc[0]["algorithm_name"]
        if not validation_rows.empty
        else ""
    )
    summary_df["selected_best_model"] = summary_df["algorithm_name"].eq(best_algorithm_name)

    summary_path = output_dir / "multialgorithm_summary.csv"
    wide_table_path = project_root / "outputs" / "tables" / "paper1_table_multialgorithm_wide_comparison.csv"
    history_path = output_dir / "multialgorithm_history.csv"
    predictions_path = output_dir / "multialgorithm_external_predictions.csv"
    confusion_path = output_dir / "multialgorithm_best_model_external_confusion.csv"
    paper_table_path = project_root / "outputs" / "tables" / "paper1_table_multialgorithm_comparison.csv"
    manifest_path = output_dir / "multialgorithm_manifest.json"

    write_dataframe(summary_path, summary_df)
    write_dataframe(history_path, pd.DataFrame(history_rows))
    write_dataframe(predictions_path, pd.concat(prediction_frames, ignore_index=True))

    best_predictions = pd.concat(prediction_frames, ignore_index=True)
    best_predictions = best_predictions.loc[best_predictions["algorithm_name"] == best_algorithm_name].copy()
    if not best_predictions.empty:
        write_dataframe(
            confusion_path,
            confusion_dataframe(
                best_predictions["true_label"].tolist(),
                best_predictions["predicted_label"].tolist(),
                class_labels,
            ),
        )

    paper_table_df = summary_df[
        [
            "algorithm_name",
            "family",
            "evaluation_stage",
            "dataset_name",
            "num_samples",
            "accuracy",
            "macro_f1",
            "weighted_f1",
            "unweighted_recall",
            "epochs",
            "selected_best_model",
        ]
    ].copy()
    write_dataframe(paper_table_path, paper_table_df)
    wide_table_df = summary_df.pivot_table(
        index=["algorithm_name", "family", "selected_best_model"],
        columns="evaluation_stage",
        values=["accuracy", "macro_f1", "weighted_f1", "unweighted_recall", "num_samples", "epochs"],
        aggfunc="first",
    )
    wide_table_df.columns = [f"{metric}_{stage}" for metric, stage in wide_table_df.columns]
    wide_table_df = wide_table_df.reset_index()
    write_dataframe(wide_table_path, wide_table_df)
    write_json(
        manifest_path,
        {
            "output_subdir": output_subdir,
            "train_dataset": "RAVDESS",
            "external_dataset": "CREMA-D",
            "best_algorithm_name": best_algorithm_name,
            "summary_csv": str(summary_path),
            "paper_table_csv": str(paper_table_path),
            "wide_table_csv": str(wide_table_path),
            "predictions_csv": str(predictions_path),
            "history_csv": str(history_path),
            "latest_status_json": str(progress_dir / "latest_status.json"),
            "progress_csv": str(progress_dir / "training_progress_latest.csv"),
        },
    )
    _write_progress_snapshot(
        progress_dir,
        {
            "run_type": "multialgorithm_case_study",
            "status": "completed",
            "best_algorithm_name": best_algorithm_name,
            "paper_table_csv": str(paper_table_path),
            "wide_table_csv": str(wide_table_path),
        },
        progress_rows,
    )
    return {
        "summary_csv": str(summary_path),
        "history_csv": str(history_path),
        "predictions_csv": str(predictions_path),
        "paper_table_csv": str(paper_table_path),
        "wide_table_csv": str(wide_table_path),
        "manifest_json": str(manifest_path),
        "latest_status_json": str(progress_dir / "latest_status.json"),
        "progress_csv": str(progress_dir / "training_progress_latest.csv"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a multi-algorithm Paper 1 emotion case-study comparison.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--train-dataset-root", default="data/public/RAVDESS")
    parser.add_argument("--train-labels-csv", default="data/public/RAVDESS/labels_broad4_angry.csv")
    parser.add_argument("--external-dataset-root", default="data/public/CREMA-D")
    parser.add_argument("--external-labels-csv", default="data/public/CREMA-D/labels_broad4_angry.csv")
    parser.add_argument("--output-subdir", default="ravdess_multialgorithm_case_study")
    parser.add_argument("--target-label-set", default="broad4_angry")
    parser.add_argument("--deep-epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-every-epochs", type=int, default=10)
    args = parser.parse_args()

    outputs = run_multialgorithm_emotion_case_study(
        project_root=Path(args.project_root).resolve(),
        train_dataset_root=Path(args.train_dataset_root),
        train_labels_csv=Path(args.train_labels_csv),
        external_dataset_root=Path(args.external_dataset_root),
        external_labels_csv=Path(args.external_labels_csv),
        output_subdir=args.output_subdir,
        target_label_set=args.target_label_set or None,
        deep_epochs=args.deep_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        image_size=args.image_size,
        random_seed=args.random_seed,
        device=args.device,
        log_every_epochs=max(args.log_every_epochs, 1),
    )
    print(outputs["paper_table_csv"])


if __name__ == "__main__":
    main()
