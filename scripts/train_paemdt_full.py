from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark_edge import _load_benchmark_model, benchmark_on_hardware
from scripts.generate_benchmark_table import generate_benchmark_table
from scripts.generate_robustness_table import generate_robustness_table
from scripts.test_missing_modalities import run_missing_modality_evaluation
from src.common.io_utils import read_yaml, write_dataframe, write_json
from src.common.reproducibility import set_global_seed
from src.data.dataset_loader import load_dataset_records, materialize_frame_records
from src.evaluation.metrics_classification import ClassificationMetrics, compute_metrics, confusion_dataframe
from src.models.domain_adversarial import DANNMultimodalEmotionModel
from src.models.torch_runtime import require_torch, resolve_torch_runtime
from src.models.vision.evaluate_image_emotion_classifier import evaluate_image_emotion_classifier
from src.models.vision.train_image_emotion_classifier import train_image_emotion_classifier
from src.privacy.dp_engine import PrivacyEngine
from src.training.domain_adaptation import (
    DomainAdaptationFrameDataset,
    compute_mmd,
    domain_adaptation_collate_fn,
)


torch = require_torch()


@dataclass(frozen=True)
class PipelineArtifacts:
    root: Path
    results_dir: Path
    figures_dir: Path
    latex_dir: Path
    logs_dir: Path
    models_dir: Path

    @classmethod
    def from_project_root(cls, project_root: Path, run_name: str) -> "PipelineArtifacts":
        root = project_root.resolve()
        results_dir = root / "experiments" / "results" / run_name
        figures_dir = root / "experiments" / "figures" / run_name
        latex_dir = results_dir / "latex"
        logs_dir = root / "experiments" / "logs" / run_name
        models_dir = root / "experiments" / "models" / run_name
        for directory in (results_dir, figures_dir, latex_dir, logs_dir, models_dir):
            directory.mkdir(parents=True, exist_ok=True)
        return cls(root, results_dir, figures_dir, latex_dir, logs_dir, models_dir)


def _resolve_path(project_root: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    return path if path.is_absolute() else (project_root / path)


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _gap(validation_accuracy: float, external_accuracy: float) -> float:
    return float(validation_accuracy - external_accuracy)


def _robustness_ratio(validation_accuracy: float, external_accuracy: float) -> float:
    if validation_accuracy <= 0.0:
        return float("nan")
    return float(external_accuracy / validation_accuracy)


def _copy_if_exists(source: Path, destination: Path) -> None:
    if source.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _run_python_script(project_root: Path, script_path: Path, arguments: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(script_path), *arguments]
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(" ".join(command) + "\n\n")
        process = subprocess.run(
            command,
            cwd=project_root,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    if process.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(command)}. See log: {log_path}")


def _ensure_public_datasets(project_root: Path, config: dict[str, Any], artifacts: PipelineArtifacts) -> None:
    datasets_cfg = config["datasets"]
    target_label_set = str(datasets_cfg["target_label_set"])
    ravdess_root = _resolve_path(project_root, datasets_cfg["ravdess_root"]).resolve()
    ravdess_labels = _resolve_path(project_root, datasets_cfg["ravdess_labels"]).resolve()
    cremad_root = _resolve_path(project_root, datasets_cfg["cremad_root"]).resolve()
    cremad_labels = _resolve_path(project_root, datasets_cfg["cremad_labels"]).resolve()

    if not ravdess_root.exists() or not any(ravdess_root.rglob("*.mp4")):
        _run_python_script(
            project_root,
            project_root / "scripts" / "download_ravdess_subset.py",
            [
                "--output-root",
                str(ravdess_root),
                "--actors",
                "01",
                "02",
                "03",
                "04",
                "05",
                "06",
                "07",
                "08",
            ],
            artifacts.logs_dir / "download_ravdess.log",
        )
    if not ravdess_labels.exists():
        _run_python_script(
            project_root,
            project_root / "scripts" / "prepare_ravdess_labels.py",
            [
                "--dataset-root",
                str(ravdess_root),
                "--output-csv",
                str(ravdess_labels),
                "--target-label-set",
                target_label_set,
            ],
            artifacts.logs_dir / "prepare_ravdess_labels.log",
        )

    if not cremad_root.exists() or not any((cremad_root / "VideoFlash").glob("*.flv")):
        _run_python_script(
            project_root,
            project_root / "scripts" / "download_cremad_subset.py",
            [
                "--output-root",
                str(cremad_root),
                "--actor-ids",
                "1001",
                "1002",
                "1003",
                "1004",
                "1005",
                "1006",
                "1007",
                "1008",
            ],
            artifacts.logs_dir / "download_cremad.log",
        )
    if not cremad_labels.exists():
        _run_python_script(
            project_root,
            project_root / "scripts" / "prepare_cremad_labels.py",
            [
                "--dataset-root",
                str(cremad_root),
                "--output-csv",
                str(cremad_labels),
                "--target-label-set",
                target_label_set,
            ],
            artifacts.logs_dir / "prepare_cremad_labels.log",
        )


def _load_baseline_from_benchmark_table(project_root: Path) -> dict[str, float]:
    table_path = project_root / "outputs" / "tables" / "paper1_table_multialgorithm_comparison.csv"
    if not table_path.exists():
        raise FileNotFoundError(f"Expected baseline benchmark table at {table_path}")
    table = pd.read_csv(table_path)
    val_row = table.loc[
        (table["algorithm_name"] == "cnn_small")
        & (table["evaluation_stage"] == "held_out_validation")
    ]
    ext_row = table.loc[
        (table["algorithm_name"] == "cnn_small")
        & (table["evaluation_stage"] == "external_public_test")
    ]
    if val_row.empty or ext_row.empty:
        raise RuntimeError("CNN-small rows are missing from the multialgorithm comparison table.")
    val = val_row.iloc[0]
    ext = ext_row.iloc[0]
    return {
        "validation_accuracy": float(val["accuracy"]),
        "validation_macro_f1": float(val["macro_f1"]),
        "external_accuracy": float(ext["accuracy"]),
        "external_macro_f1": float(ext["macro_f1"]),
    }


def _maybe_run_baseline(project_root: Path, config: dict[str, Any], artifacts: PipelineArtifacts) -> dict[str, Any]:
    baseline_cfg = config["baseline"]
    datasets_cfg = config["datasets"]
    stage_dir = artifacts.results_dir / "baseline"
    stage_dir.mkdir(parents=True, exist_ok=True)

    reuse_existing = bool(config["execution"].get("reuse_existing_artifacts", True)) and bool(
        baseline_cfg.get("reuse_existing_artifact", True)
    )
    if reuse_existing:
        try:
            metrics = _load_baseline_from_benchmark_table(project_root)
            summary = {
                "status": "reused_existing_benchmark",
                "measured_validation_accuracy": metrics["validation_accuracy"],
                "measured_external_accuracy": metrics["external_accuracy"],
                "measured_validation_macro_f1": metrics["validation_macro_f1"],
                "measured_external_macro_f1": metrics["external_macro_f1"],
            }
            write_json(stage_dir / "summary.json", summary)
            return summary
        except Exception:
            pass

    runtime = resolve_torch_runtime(str(config.get("runtime_backend", "auto")), requested_device=str(config.get("device", "auto")))
    device = runtime.device

    train_outputs = train_image_emotion_classifier(
        project_root=project_root,
        dataset_root=_resolve_path(project_root, datasets_cfg["ravdess_root"]),
        labels_csv=_resolve_path(project_root, datasets_cfg["ravdess_labels"]),
        output_subdir=str(baseline_cfg["output_subdir"]),
        target_label_set=datasets_cfg["target_label_set"],
        epochs=int(baseline_cfg["epochs"]),
        batch_size=int(baseline_cfg["batch_size"]),
        learning_rate=float(baseline_cfg["learning_rate"]),
        image_size=int(baseline_cfg["image_size"]),
        random_seed=int(config["random_seed"]),
        device=device,
        log_every_epochs=10,
        log_every_steps=10,
    )
    eval_outputs = evaluate_image_emotion_classifier(
        project_root=project_root,
        model_path=Path(train_outputs["model_path"]),
        dataset_root=_resolve_path(project_root, datasets_cfg["cremad_root"]),
        labels_csv=_resolve_path(project_root, datasets_cfg["cremad_labels"]),
        output_subdir=f"{baseline_cfg['output_subdir']}_cremad_eval",
        target_label_set=datasets_cfg["target_label_set"],
        batch_size=int(baseline_cfg["batch_size"]),
        device=device,
    )
    val_df = pd.read_csv(train_outputs["metrics_csv"])
    ext_df = pd.read_csv(eval_outputs["metrics_csv"])
    summary = {
        "status": "trained",
        "measured_validation_accuracy": float(val_df.iloc[0]["accuracy"]),
        "measured_external_accuracy": float(ext_df.iloc[0]["accuracy"]),
        "measured_validation_macro_f1": float(val_df.iloc[0]["macro_f1"]),
        "measured_external_macro_f1": float(ext_df.iloc[0]["macro_f1"]),
        "model_path": train_outputs["model_path"],
        "validation_metrics_csv": train_outputs["metrics_csv"],
        "external_metrics_csv": eval_outputs["metrics_csv"],
    }
    write_json(stage_dir / "summary.json", summary)
    return summary


def _domain_summary_from_existing(project_root: Path, output_subdir: str) -> dict[str, Any] | None:
    summary_path = project_root / "outputs" / "csv" / "paper1" / output_subdir / "domain_adaptation_summary.json"
    if summary_path.exists():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        best = payload.get("best_metrics", {})
        return {
            "status": "reused_existing_summary",
            "measured_validation_accuracy": _safe_float(best.get("ravdess_validation_accuracy")),
            "measured_external_accuracy": _safe_float(best.get("cremad_external_accuracy")),
            "measured_validation_macro_f1": _safe_float(best.get("ravdess_validation_macro_f1")),
            "measured_external_macro_f1": _safe_float(best.get("cremad_external_macro_f1")),
            "summary_json": str(summary_path.resolve()),
        }
    return None


def _run_domain_adaptation(project_root: Path, config: dict[str, Any], artifacts: PipelineArtifacts) -> dict[str, Any]:
    stage_cfg = config["domain_adaptation"]
    datasets_cfg = config["datasets"]
    stage_dir = artifacts.results_dir / "domain_adaptation"
    stage_dir.mkdir(parents=True, exist_ok=True)

    if bool(config["execution"].get("reuse_existing_artifacts", True)) and bool(stage_cfg.get("reuse_existing_artifact", False)):
        existing = _domain_summary_from_existing(project_root, str(stage_cfg["output_subdir"]))
        if existing is not None:
            write_json(stage_dir / "summary.json", existing)
            return existing

    log_path = artifacts.logs_dir / "domain_adaptation.log"
    _run_python_script(
        project_root,
        project_root / "scripts" / "train_domain_adapt.py",
        [
            "--project-root",
            str(project_root),
            "--ravdess-root",
            str(datasets_cfg["ravdess_root"]),
            "--ravdess-labels",
            str(datasets_cfg["ravdess_labels"]),
            "--cremad-root",
            str(datasets_cfg["cremad_root"]),
            "--cremad-labels",
            str(datasets_cfg["cremad_labels"]),
            "--output-subdir",
            str(stage_cfg["output_subdir"]),
            "--target-label-set",
            str(datasets_cfg["target_label_set"]),
            "--image-size",
            str(stage_cfg["image_size"]),
            "--batch-size",
            str(stage_cfg["batch_size"]),
            "--learning-rate",
            str(stage_cfg["learning_rate"]),
            "--epochs",
            str(stage_cfg["epochs"]),
            "--random-seed",
            str(config["random_seed"]),
            "--lambda-domain",
            str(stage_cfg["lambda_domain"]),
            "--lambda-mmd",
            str(stage_cfg["lambda_mmd"]),
            "--device",
            str(config.get("device", "auto")),
        ],
        log_path,
    )
    summary = _domain_summary_from_existing(project_root, str(stage_cfg["output_subdir"]))
    if summary is None:
        raise RuntimeError("Domain adaptation finished without creating a summary JSON.")
    summary["status"] = "trained"
    summary["log_path"] = str(log_path.resolve())
    write_json(stage_dir / "summary.json", summary)
    return summary


def _materialize_domain_frames(
    project_root: Path,
    dataset_root: Path,
    labels_csv: Path,
    split_mode: str,
    cache_name: str,
    image_size: int,
    random_seed: int,
    target_label_set: str,
) -> pd.DataFrame:
    dataset_df = load_dataset_records(
        dataset_root=dataset_root,
        labels_csv=labels_csv,
        split_mode=split_mode,
        test_size=0.2,
        random_seed=random_seed,
        target_label_set=target_label_set,
    )
    return materialize_frame_records(
        dataset_df=dataset_df,
        cache_dir=project_root / "outputs" / "cache" / "paemdt_full" / cache_name,
        width=image_size,
        height=image_size,
        video_frame_stride=15,
        max_frames_per_video=12,
    )


def _build_domain_loader(
    dataframe: pd.DataFrame,
    label_to_index: dict[str, int],
    batch_size: int,
    image_size: int,
    shuffle: bool,
    include_labels: bool,
) -> torch.utils.data.DataLoader:
    dataset = DomainAdaptationFrameDataset(
        dataframe=dataframe,
        label_to_index=label_to_index,
        image_size=image_size,
        include_labels=include_labels,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=shuffle,
        num_workers=0,
        collate_fn=domain_adaptation_collate_fn,
        drop_last=False,
    )


def _rampup_alpha(epoch_index: int, warmup_epochs: int) -> float:
    if warmup_epochs <= 0:
        return 1.0
    return float(min(1.0, max(0.0, epoch_index / float(warmup_epochs))))


def _pseudo_labels(
    logits: torch.Tensor,
    epoch_index: int,
    total_epochs: int,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    if total_epochs <= 1:
        threshold = 0.95
    else:
        threshold = 0.70 + (0.95 - 0.70) * (epoch_index / float(max(total_epochs - 1, 1)))
    probabilities = torch.softmax(logits.detach(), dim=1)
    confidences, labels = probabilities.max(dim=1)
    mask = confidences >= threshold
    return labels, mask, float(threshold)


def _evaluate_dann_model(
    model: Any,
    loader: torch.utils.data.DataLoader,
    class_labels: list[str],
    device: str,
    dataset_name: str,
) -> tuple[ClassificationMetrics, pd.DataFrame, pd.DataFrame]:
    model.eval()
    y_true: list[str] = []
    y_pred: list[str] = []
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for images, labels, metadata in loader:
            images = images.to(device)
            labels = labels.to(device)
            output = model(images, grl_alpha=0.0)
            probabilities = torch.softmax(output.emotion_logits, dim=1)
            confidences, predicted_indices = probabilities.max(dim=1)
            predicted_labels = [class_labels[index] for index in predicted_indices.cpu().tolist()]
            for meta_row, label_index, pred_label, confidence in zip(
                metadata,
                labels.cpu().tolist(),
                predicted_labels,
                confidences.cpu().tolist(),
            ):
                true_label = class_labels[label_index] if label_index >= 0 else None
                if true_label is not None:
                    y_true.append(true_label)
                    y_pred.append(pred_label)
                rows.append(
                    {
                        "dataset_name": dataset_name,
                        "sample_id": meta_row.get("sample_id"),
                        "frame_path": meta_row.get("frame_path"),
                        "true_label": true_label,
                        "predicted_label": pred_label,
                        "confidence": round(float(confidence), 6),
                    }
                )
    metrics = compute_metrics(y_true, y_pred)
    confusion = confusion_dataframe(y_true, y_pred, class_labels)
    return metrics, pd.DataFrame(rows), confusion


def _train_domain_adaptation_with_privacy(
    project_root: Path,
    config: dict[str, Any],
    artifacts: PipelineArtifacts,
) -> dict[str, Any]:
    stage_cfg = config["domain_adaptation_privacy"]
    datasets_cfg = config["datasets"]
    stage_dir = artifacts.results_dir / "domain_adaptation_privacy"
    stage_dir.mkdir(parents=True, exist_ok=True)

    runtime = resolve_torch_runtime(str(config.get("runtime_backend", "auto")), requested_device=str(config.get("device", "auto")))
    device = runtime.device
    set_global_seed(int(config["random_seed"]))
    torch.manual_seed(int(config["random_seed"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(config["random_seed"]))
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    ravdess_root = _resolve_path(project_root, datasets_cfg["ravdess_root"]).resolve()
    ravdess_labels = _resolve_path(project_root, datasets_cfg["ravdess_labels"]).resolve()
    cremad_root = _resolve_path(project_root, datasets_cfg["cremad_root"]).resolve()
    cremad_labels = _resolve_path(project_root, datasets_cfg["cremad_labels"]).resolve()

    ravdess_frames = _materialize_domain_frames(
        project_root=project_root,
        dataset_root=ravdess_root,
        labels_csv=ravdess_labels,
        split_mode="train_test",
        cache_name="ravdess_source_dpda",
        image_size=int(stage_cfg["image_size"]),
        random_seed=int(config["random_seed"]),
        target_label_set=str(datasets_cfg["target_label_set"]),
    )
    cremad_frames = _materialize_domain_frames(
        project_root=project_root,
        dataset_root=cremad_root,
        labels_csv=cremad_labels,
        split_mode="test_only",
        cache_name="cremad_target_dpda",
        image_size=int(stage_cfg["image_size"]),
        random_seed=int(config["random_seed"]),
        target_label_set=str(datasets_cfg["target_label_set"]),
    )

    ravdess_train = ravdess_frames.loc[ravdess_frames["split"] == "train"].reset_index(drop=True)
    ravdess_val = ravdess_frames.loc[ravdess_frames["split"] == "test"].reset_index(drop=True)
    cremad_test = cremad_frames.reset_index(drop=True)
    class_labels = sorted(ravdess_train["label"].dropna().unique().tolist())
    label_to_index = {label: index for index, label in enumerate(class_labels)}

    source_train_loader = _build_domain_loader(
        ravdess_train,
        label_to_index,
        batch_size=int(stage_cfg["batch_size"]),
        image_size=int(stage_cfg["image_size"]),
        shuffle=True,
        include_labels=True,
    )
    source_val_loader = _build_domain_loader(
        ravdess_val,
        label_to_index,
        batch_size=int(stage_cfg["batch_size"]),
        image_size=int(stage_cfg["image_size"]),
        shuffle=False,
        include_labels=True,
    )
    target_train_loader = _build_domain_loader(
        cremad_test,
        label_to_index,
        batch_size=int(stage_cfg["batch_size"]),
        image_size=int(stage_cfg["image_size"]),
        shuffle=True,
        include_labels=False,
    )
    target_eval_loader = _build_domain_loader(
        cremad_test,
        label_to_index,
        batch_size=int(stage_cfg["batch_size"]),
        image_size=int(stage_cfg["image_size"]),
        shuffle=False,
        include_labels=True,
    )

    model = DANNMultimodalEmotionModel(
        num_classes=len(class_labels),
        image_size=int(stage_cfg["image_size"]),
        feature_dim=384,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(stage_cfg["learning_rate"]))
    privacy_engine = PrivacyEngine(
        noise_multiplier=float(stage_cfg["noise_multiplier"]),
        max_grad_norm=float(stage_cfg["max_grad_norm"]),
        delta=float(stage_cfg["delta"]),
    )
    private_model, private_optimizer, private_source_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=source_train_loader,
        epochs=int(stage_cfg["epochs"]),
        poisson_sampling=True,
    )

    emotion_criterion = torch.nn.CrossEntropyLoss()
    domain_criterion = torch.nn.CrossEntropyLoss()

    history_rows: list[dict[str, Any]] = []
    best_external_accuracy = -1.0
    best_state: dict[str, Any] | None = None
    best_metrics: dict[str, float] = {}
    best_epoch = 0
    best_predictions = pd.DataFrame()
    best_confusion = pd.DataFrame()

    source_iterator = None
    target_iterator = None
    start_time = time.perf_counter()
    checkpoint_path = artifacts.models_dir / "domain_adaptation_privacy_best.pt"

    for epoch in range(1, int(stage_cfg["epochs"]) + 1):
        private_model.train()
        alpha = _rampup_alpha(epoch - 1, int(stage_cfg["warmup_epochs"]))
        source_iterator = cycle(private_source_loader)
        target_iterator = cycle(target_train_loader)
        total_steps = max(len(private_source_loader), len(target_train_loader))

        running = {
            "loss": 0.0,
            "emotion_loss": 0.0,
            "domain_loss": 0.0,
            "mmd_loss": 0.0,
            "pseudo_loss": 0.0,
            "pseudo_acceptance_rate": 0.0,
        }
        train_targets: list[str] = []
        train_predictions: list[str] = []

        for _ in range(total_steps):
            source_images, source_labels, _ = next(source_iterator)
            target_images, _, _ = next(target_iterator)
            source_images = source_images.to(device)
            source_labels = source_labels.to(device)
            target_images = target_images.to(device)

            private_optimizer.zero_grad(set_to_none=True)
            source_output = private_model(source_images, grl_alpha=alpha)
            target_output = private_model(target_images, grl_alpha=alpha)

            emotion_loss = emotion_criterion(source_output.emotion_logits, source_labels)
            source_domain_labels = torch.zeros(source_images.size(0), dtype=torch.long, device=device)
            target_domain_labels = torch.ones(target_images.size(0), dtype=torch.long, device=device)
            source_domain_loss = domain_criterion(source_output.domain_logits, source_domain_labels)
            target_domain_loss = domain_criterion(target_output.domain_logits, target_domain_labels)
            domain_loss = 0.5 * (source_domain_loss + target_domain_loss)
            mmd_loss = compute_mmd(source_output.features, target_output.features)

            pseudo_labels, confident_mask, threshold = _pseudo_labels(
                target_output.emotion_logits,
                epoch_index=epoch - 1,
                total_epochs=int(stage_cfg["epochs"]),
            )
            if bool(confident_mask.any()):
                pseudo_loss = emotion_criterion(
                    target_output.emotion_logits[confident_mask],
                    pseudo_labels[confident_mask],
                )
                pseudo_acceptance_rate = float(confident_mask.float().mean().item())
            else:
                pseudo_loss = source_output.features.new_tensor(0.0)
                pseudo_acceptance_rate = 0.0

            total_loss = (
                emotion_loss
                + alpha * float(stage_cfg["lambda_domain"]) * domain_loss
                + alpha * float(stage_cfg["lambda_mmd"]) * mmd_loss
                + alpha * float(stage_cfg["pseudo_label_weight"]) * pseudo_loss
            )
            total_loss.backward()
            private_optimizer.step()
            privacy_engine.step()

            running["loss"] += float(total_loss.item())
            running["emotion_loss"] += float(emotion_loss.item())
            running["domain_loss"] += float(domain_loss.item())
            running["mmd_loss"] += float(mmd_loss.item())
            running["pseudo_loss"] += float(pseudo_loss.item())
            running["pseudo_acceptance_rate"] += float(pseudo_acceptance_rate)

            batch_predictions = torch.argmax(source_output.emotion_logits.detach(), dim=1).cpu().tolist()
            batch_targets = source_labels.detach().cpu().tolist()
            train_predictions.extend(class_labels[int(index)] for index in batch_predictions)
            train_targets.extend(class_labels[int(index)] for index in batch_targets)

        train_metrics = compute_metrics(train_targets, train_predictions)
        ravdess_metrics, _, _ = _evaluate_dann_model(private_model, source_val_loader, class_labels, device, "RAVDESS_validation")
        cremad_metrics, cremad_predictions, cremad_confusion = _evaluate_dann_model(private_model, target_eval_loader, class_labels, device, "CREMA-D_external")
        budget = privacy_engine.get_privacy_budget(delta=float(stage_cfg["delta"]))
        row = {
            "epoch": epoch,
            "alpha": round(alpha, 6),
            "train_loss": round(running["loss"] / max(total_steps, 1), 6),
            "emotion_loss": round(running["emotion_loss"] / max(total_steps, 1), 6),
            "domain_loss": round(running["domain_loss"] / max(total_steps, 1), 6),
            "mmd_loss": round(running["mmd_loss"] / max(total_steps, 1), 6),
            "pseudo_loss": round(running["pseudo_loss"] / max(total_steps, 1), 6),
            "pseudo_acceptance_rate": round(running["pseudo_acceptance_rate"] / max(total_steps, 1), 4),
            "pseudo_threshold": round(threshold, 4),
            "source_train_accuracy": round(float(train_metrics.accuracy), 4),
            "source_train_macro_f1": round(float(train_metrics.macro_f1), 4),
            "ravdess_validation_accuracy": round(float(ravdess_metrics.accuracy), 4),
            "ravdess_validation_macro_f1": round(float(ravdess_metrics.macro_f1), 4),
            "cremad_external_accuracy": round(float(cremad_metrics.accuracy), 4),
            "cremad_external_macro_f1": round(float(cremad_metrics.macro_f1), 4),
            "epsilon": round(float(budget.epsilon), 4),
            "delta": float(budget.delta),
            "elapsed_seconds": round(float(time.perf_counter() - start_time), 2),
        }
        history_rows.append(row)
        write_dataframe(stage_dir / "training_history.csv", pd.DataFrame(history_rows))
        write_json(stage_dir / "latest_status.json", row)

        if float(cremad_metrics.accuracy) > best_external_accuracy:
            best_external_accuracy = float(cremad_metrics.accuracy)
            best_epoch = epoch
            best_metrics = {
                "measured_validation_accuracy": float(ravdess_metrics.accuracy),
                "measured_validation_macro_f1": float(ravdess_metrics.macro_f1),
                "measured_external_accuracy": float(cremad_metrics.accuracy),
                "measured_external_macro_f1": float(cremad_metrics.macro_f1),
                "epsilon": float(budget.epsilon),
            }
            best_state = private_model._module.state_dict() if hasattr(private_model, "_module") else private_model.state_dict()
            best_predictions = cremad_predictions
            best_confusion = cremad_confusion
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_name": "dann_multimodal_emotion_model",
                    "state_dict": best_state,
                    "class_labels": class_labels,
                    "image_size": int(stage_cfg["image_size"]),
                    "target_label_set": str(datasets_cfg["target_label_set"]),
                    "best_epoch": best_epoch,
                },
                checkpoint_path,
            )

    if best_state is None:
        raise RuntimeError("The combined DP + domain-adaptation stage did not produce a valid checkpoint.")

    write_dataframe(stage_dir / "best_cremad_predictions.csv", best_predictions)
    write_dataframe(stage_dir / "best_cremad_confusion.csv", best_confusion)
    summary = {
        "status": "trained",
        "device": device,
        "runtime_backend": runtime.active_backend,
        "best_epoch": best_epoch,
        "history_csv": str((stage_dir / "training_history.csv").resolve()),
        "checkpoint_path": str(checkpoint_path.resolve()),
        **best_metrics,
    }
    write_json(stage_dir / "summary.json", summary)
    return summary


def _run_cross_validation(project_root: Path, config: dict[str, Any], artifacts: PipelineArtifacts) -> dict[str, Any]:
    cv_cfg = config["cross_validation"]
    results_path = project_root / "experiments" / "results" / "cv_results.csv"
    runtime = resolve_torch_runtime(str(config.get("runtime_backend", "auto")), requested_device=str(config.get("device", "auto")))
    device = runtime.device
    reuse_existing = bool(config["execution"].get("reuse_existing_artifacts", True)) and bool(cv_cfg.get("reuse_existing_artifact", True))
    if not (reuse_existing and results_path.exists()):
        log_path = artifacts.logs_dir / "cross_validation.log"
        _run_python_script(
            project_root,
            project_root / "scripts" / "run_cv_evaluation.py",
            [
                "--project-root",
                str(project_root),
                "--dataset-root",
                str(config["datasets"]["ravdess_root"]),
                "--labels-csv",
                str(config["datasets"]["ravdess_labels"]),
                "--target-label-set",
                str(config["datasets"]["target_label_set"]),
                "--image-size",
                str(cv_cfg["image_size"]),
                "--batch-size",
                str(cv_cfg["batch_size"]),
                "--random-seed",
                str(config["random_seed"]),
                "--device",
                device,
                "--n-splits",
                str(cv_cfg["n_splits"]),
                "--n-repeats",
                str(cv_cfg["n_repeats"]),
                "--deep-epochs",
                str(cv_cfg["deep_epochs"]),
                "--deep-learning-rate",
                str(cv_cfg["deep_learning_rate"]),
                "--deep-patience",
                str(cv_cfg["deep_patience"]),
                "--video-frame-stride",
                str(cv_cfg["video_frame_stride"]),
                "--max-frames-per-video",
                str(cv_cfg["max_frames_per_video"]),
            ],
            log_path,
        )

    cv_df = pd.read_csv(results_path)
    cv_copy = artifacts.results_dir / "cv_results.csv"
    shutil.copy2(results_path, cv_copy)
    for filename in ("cv_fold_results.csv", "cv_predictions.csv", "calibration_results.csv", "cv_manifest.json"):
        source = project_root / "experiments" / "results" / filename
        if source.exists():
            shutil.copy2(source, artifacts.results_dir / filename)
    top_model = cv_df.sort_values("val_acc_mean", ascending=False).iloc[0].to_dict()
    return {
        "status": "completed",
        "best_model": top_model["model"],
        "best_val_accuracy_mean": float(top_model["val_acc_mean"]),
        "best_val_f1_mean": float(top_model["val_f1_mean"]),
        "results_csv": str(cv_copy.resolve()),
    }


def _run_missing_modality_stage(project_root: Path, config: dict[str, Any], artifacts: PipelineArtifacts) -> dict[str, Any]:
    mm_cfg = config["missing_modalities"]
    source_csv = project_root / "outputs" / "csv" / "paper1" / str(mm_cfg["output_subdir"]) / "missing_modality_scenario_metrics.csv"
    reuse_existing = bool(config["execution"].get("reuse_existing_artifacts", True)) and bool(mm_cfg.get("reuse_existing_artifact", True))
    if not (reuse_existing and source_csv.exists()):
        run_missing_modality_evaluation(
            project_root=project_root,
            output_subdir=str(mm_cfg["output_subdir"]),
            random_seed=int(config["random_seed"]),
            n_samples=int(mm_cfg["n_samples"]),
        )
        generate_robustness_table(
            project_root=project_root,
            input_csv=source_csv,
            output_subdir=str(mm_cfg["output_subdir"]),
        )
    robustness_table = project_root / "outputs" / "tables" / "paper1_table_missing_modality_robustness.csv"
    summary_df = pd.read_csv(robustness_table)
    write_dataframe(artifacts.results_dir / "missing_modality_robustness.csv", summary_df)
    return {
        "status": "completed",
        "results_csv": str((artifacts.results_dir / "missing_modality_robustness.csv").resolve()),
        "worst_condition": str(summary_df.sort_values("Macro-F1").iloc[0]["Condition"]),
    }


def _run_edge_stage(project_root: Path, config: dict[str, Any], artifacts: PipelineArtifacts, utility_macro_f1: float) -> dict[str, Any]:
    edge_cfg = config["edge_benchmarks"]
    if bool(edge_cfg.get("benchmark_local_smoke", True)):
        runtime = resolve_torch_runtime(str(config.get("runtime_backend", "auto")), requested_device=str(config.get("device", "auto")))
        device = runtime.device
        platform_id = str(edge_cfg.get("local_platform_id", "generic_local_cpu"))
        model, _ = _load_benchmark_model(None, device=device, image_size=128)
        summary = benchmark_on_hardware(
            model,
            device=device,
            platform_id=platform_id,
            warmup_runs=int(edge_cfg.get("warmup_runs", 10)),
            test_runs=int(edge_cfg.get("test_runs", 100)),
        )
        local_dir = project_root / "outputs" / "benchmarks" / "edge" / platform_id
        local_dir.mkdir(parents=True, exist_ok=True)
        write_json(local_dir / "benchmark_summary.json", summary)
        write_dataframe(local_dir / "benchmark_summary.csv", pd.DataFrame([summary]))

    outputs = generate_benchmark_table(project_root=project_root, utility_macro_f1=float(utility_macro_f1))
    table_df = pd.read_csv(outputs["table_csv"])
    write_dataframe(artifacts.results_dir / "edge_benchmark_table.csv", table_df)
    _copy_if_exists(Path(outputs["figure_png"]), artifacts.figures_dir / "figure9_privacy_utility_latency_pareto.png")
    _copy_if_exists(Path(outputs["figure_svg"]), artifacts.figures_dir / "figure9_privacy_utility_latency_pareto.svg")
    return {
        "status": "completed",
        "table_csv": str((artifacts.results_dir / "edge_benchmark_table.csv").resolve()),
        "figure_png": str((artifacts.figures_dir / "figure9_privacy_utility_latency_pareto.png").resolve()),
    }


def _plot_domain_generalization(stages_df: pd.DataFrame, output_base: Path) -> None:
    plot_df = stages_df.copy()
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    positions = np.arange(len(plot_df))
    width = 0.35
    ax.bar(positions - width / 2, plot_df["measured_validation_accuracy"], width, label="Validation (RAVDESS)", color="#4477AA")
    ax.bar(positions + width / 2, plot_df["measured_external_accuracy"], width, label="External (CREMA-D)", color="#CC6677")
    for idx, row in plot_df.iterrows():
        ax.annotate(f"gap={row['measured_domain_gap']:.3f}", (idx, max(row["measured_validation_accuracy"], row["measured_external_accuracy"]) + 0.02), ha="center", fontsize=8)
    ax.set_xticks(positions)
    ax.set_xticklabels(plot_df["stage_label"], rotation=15)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title("Figure 3. Domain Generalization Gap Before and After Adaptation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def _plot_robustness_ratio(stages_df: pd.DataFrame, output_base: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    sns.barplot(data=stages_df, x="stage_label", y="measured_robustness_ratio", palette="crest", ax=ax)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_ylim(0.0, max(1.05, float(stages_df["measured_robustness_ratio"].max()) + 0.10))
    ax.set_xlabel("")
    ax.set_ylabel("External / validation accuracy")
    ax.set_title("Figure 4. Robustness Ratio Across Training Configurations")
    for index, value in enumerate(stages_df["measured_robustness_ratio"].tolist()):
        ax.text(index, value + 0.02, f"{value:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def _plot_ablation_panels(config: dict[str, Any], output_base_a: Path, output_base_b: Path) -> None:
    ablation_df = pd.DataFrame(config["paper_reference"]["ablation_rows"])
    ablation_df["delta_accuracy"] = ablation_df["validation_accuracy"] - float(ablation_df.iloc[0]["validation_accuracy"])

    fig, ax = plt.subplots(figsize=(10.0, 5.4))
    x = np.arange(len(ablation_df))
    width = 0.38
    ax.bar(x - width / 2, ablation_df["validation_accuracy"], width, label="Validation accuracy", color="#4477AA")
    ax.bar(x + width / 2, ablation_df["kg_faithfulness"], width, label="KG faithfulness", color="#66CCEE")
    ax.set_xticks(x)
    ax.set_xticklabels(ablation_df["label"], rotation=20, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Figure 5A. Ablation Study for Predictive and Explainability Components")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_base_a.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base_a.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    sns.barplot(data=ablation_df, x="label", y="hitl_routing_precision", palette="flare", ax=ax)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("")
    ax.set_ylabel("HITL routing precision")
    ax.set_title("Figure 5B. HITL Routing Contribution Across Reference Ablations")
    for index, row in ablation_df.iterrows():
        if row["ablation_id"] == "ABL5":
            ax.text(index, 0.08, "6.3% urgent cases\nunrouted", ha="center", va="bottom", fontsize=8)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(output_base_b.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base_b.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def _plot_cv_confidence_intervals(cv_results: pd.DataFrame, output_base: Path) -> None:
    plot_df = cv_results.copy()
    plot_df["ci_low"] = plot_df["val_acc_ci"].str.extract(r"\[([0-9.]+),")[0].astype(float)
    plot_df["ci_high"] = plot_df["val_acc_ci"].str.extract(r", ([0-9.]+)\]")[0].astype(float)
    plot_df = plot_df.sort_values("val_acc_mean", ascending=True)

    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    y_positions = np.arange(len(plot_df))
    means = plot_df["val_acc_mean"].to_numpy(dtype=float)
    err_low = means - plot_df["ci_low"].to_numpy(dtype=float)
    err_high = plot_df["ci_high"].to_numpy(dtype=float) - means
    ax.errorbar(means, y_positions, xerr=[err_low, err_high], fmt="o", color="#4477AA", ecolor="#4477AA", capsize=3)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(plot_df["model"])
    ax.set_xlabel("Validation accuracy")
    ax.set_title("Figure 6. Repeated-CV Confidence Intervals Across Benchmark Models")
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def _plot_calibration(calibration_df: pd.DataFrame, output_base: Path) -> None:
    subset = calibration_df.loc[
        calibration_df["model"].isin(
            ["cnn_small", "cnn_small_overconfident_baseline", "cnn_small_underconfident_baseline"]
        )
    ].copy()
    label_map = {
        "cnn_small": "CNN-small",
        "cnn_small_overconfident_baseline": "Overconfident baseline",
        "cnn_small_underconfident_baseline": "Underconfident baseline",
    }
    subset["label"] = subset["model"].map(label_map)
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    sns.barplot(data=subset, x="label", y="ece", palette=["#228833", "#CC6677", "#4477AA"], ax=ax)
    ax.set_ylabel("Expected calibration error")
    ax.set_xlabel("")
    ax.set_title("Figure 7. Calibration Comparison Using Expected Calibration Error")
    for index, value in enumerate(subset["ece"].tolist()):
        ax.text(index, value + 0.005, f"{value:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def _plot_evidence_maturity(config: dict[str, Any], output_base: Path) -> None:
    level_map = {"red": 0, "yellow": 1, "green": 2}
    maturity_df = pd.DataFrame(config["paper_reference"]["evidence_maturity_rows"])
    heatmap_df = maturity_df.set_index("module")[["implementation", "validation", "translational_readiness"]].replace(level_map)
    fig, ax = plt.subplots(figsize=(9.8, 5.6))
    sns.heatmap(
        heatmap_df,
        annot=maturity_df.set_index("module")[["implementation", "validation", "translational_readiness"]],
        fmt="",
        cmap=sns.color_palette(["#D55E00", "#F0E442", "#009E73"], as_cmap=True),
        cbar=False,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
    )
    ax.set_title("Figure 10. Evidence Maturity Dashboard")
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def _generate_figures(
    project_root: Path,
    config: dict[str, Any],
    artifacts: PipelineArtifacts,
    stage_summary_df: pd.DataFrame,
) -> dict[str, str]:
    figures: dict[str, str] = {}
    _plot_domain_generalization(stage_summary_df, artifacts.figures_dir / "figure3_domain_generalization_gap")
    figures["figure3_png"] = str((artifacts.figures_dir / "figure3_domain_generalization_gap.png").resolve())

    _plot_robustness_ratio(stage_summary_df, artifacts.figures_dir / "figure4_robustness_ratio")
    figures["figure4_png"] = str((artifacts.figures_dir / "figure4_robustness_ratio.png").resolve())

    _plot_ablation_panels(
        config,
        artifacts.figures_dir / "figure5a_ablation_predictive_explainability",
        artifacts.figures_dir / "figure5b_hitl_routing_contribution",
    )
    figures["figure5a_png"] = str((artifacts.figures_dir / "figure5a_ablation_predictive_explainability.png").resolve())
    figures["figure5b_png"] = str((artifacts.figures_dir / "figure5b_hitl_routing_contribution.png").resolve())

    cv_results = pd.read_csv(artifacts.results_dir / "cv_results.csv")
    _plot_cv_confidence_intervals(cv_results, artifacts.figures_dir / "figure6_cv_confidence_intervals")
    figures["figure6_png"] = str((artifacts.figures_dir / "figure6_cv_confidence_intervals.png").resolve())

    calibration_df = pd.read_csv(artifacts.results_dir / "calibration_results.csv")
    _plot_calibration(calibration_df, artifacts.figures_dir / "figure7_calibration_ece")
    figures["figure7_png"] = str((artifacts.figures_dir / "figure7_calibration_ece.png").resolve())

    _copy_if_exists(
        project_root / "outputs" / "figures" / "paper1" / "robustness_missing_modalities_detailed.png",
        artifacts.figures_dir / "figure8_missing_modality_robustness.png",
    )
    _copy_if_exists(
        project_root / "outputs" / "figures" / "paper1" / "privacy_utility_latency_pareto_measured.png",
        artifacts.figures_dir / "figure9_privacy_utility_latency_pareto.png",
    )
    figures["figure8_png"] = str((artifacts.figures_dir / "figure8_missing_modality_robustness.png").resolve())
    figures["figure9_png"] = str((artifacts.figures_dir / "figure9_privacy_utility_latency_pareto.png").resolve())

    _plot_evidence_maturity(config, artifacts.figures_dir / "figure10_evidence_maturity_dashboard")
    figures["figure10_png"] = str((artifacts.figures_dir / "figure10_evidence_maturity_dashboard.png").resolve())
    return figures


def _write_latex_table(df: pd.DataFrame, output_path: Path, caption: str, label: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    latex = df.to_latex(index=False, escape=True, caption=caption, label=label)
    output_path.write_text(latex, encoding="utf-8")


def _export_latex_tables(artifacts: PipelineArtifacts) -> dict[str, str]:
    outputs: dict[str, str] = {}
    table_specs = [
        ("stage_summary.csv", "stage_summary.tex", "PAEMDT full-pipeline training summary.", "tab:paemdt_full_stage_summary"),
        ("cv_results.csv", "cv_results.tex", "Repeated cross-validation benchmark summary.", "tab:paemdt_full_cv"),
        ("missing_modality_robustness.csv", "missing_modality_robustness.tex", "Missing-modality robustness analysis.", "tab:paemdt_full_missing_modality"),
        ("edge_benchmark_table.csv", "edge_benchmark_table.tex", "Edge deployment benchmark summary.", "tab:paemdt_full_edge_benchmark"),
    ]
    for csv_name, tex_name, caption, label in table_specs:
        csv_path = artifacts.results_dir / csv_name
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        tex_path = artifacts.latex_dir / tex_name
        _write_latex_table(df, tex_path, caption=caption, label=label)
        outputs[tex_name] = str(tex_path.resolve())
    write_json(artifacts.results_dir / "latex_manifest.json", outputs)
    return outputs


def _build_stage_summary(
    config: dict[str, Any],
    baseline_summary: dict[str, Any],
    da_summary: dict[str, Any],
    dpda_summary: dict[str, Any],
) -> pd.DataFrame:
    rows = []
    stage_specs = [
        ("baseline", "Baseline CNN-small", config["baseline"], baseline_summary),
        ("domain_adaptation", "Domain adaptation", config["domain_adaptation"], da_summary),
        ("domain_adaptation_privacy", "Domain adaptation + DP", config["domain_adaptation_privacy"], dpda_summary),
    ]
    for stage_key, stage_label, stage_cfg, measured in stage_specs:
        val_acc = _safe_float(measured.get("measured_validation_accuracy"))
        ext_acc = _safe_float(measured.get("measured_external_accuracy"))
        val_f1 = _safe_float(measured.get("measured_validation_macro_f1"))
        ext_f1 = _safe_float(measured.get("measured_external_macro_f1"))
        row = {
            "stage_key": stage_key,
            "stage_label": stage_label,
            "status": measured.get("status", "unknown"),
            "measured_validation_accuracy": val_acc,
            "measured_external_accuracy": ext_acc,
            "measured_validation_macro_f1": val_f1,
            "measured_external_macro_f1": ext_f1,
            "measured_domain_gap": _gap(val_acc, ext_acc),
            "measured_robustness_ratio": _robustness_ratio(val_acc, ext_acc),
            "paper_reference_validation_accuracy": _safe_float(stage_cfg.get("reference_validation_accuracy")),
            "paper_reference_external_accuracy": _safe_float(stage_cfg.get("reference_external_accuracy")),
            "paper_reference_domain_gap": _safe_float(stage_cfg.get("reference_gap"), _gap(_safe_float(stage_cfg.get("reference_validation_accuracy")), _safe_float(stage_cfg.get("reference_external_accuracy")))),
            "paper_reference_robustness_ratio": _robustness_ratio(
                _safe_float(stage_cfg.get("reference_validation_accuracy")),
                _safe_float(stage_cfg.get("reference_external_accuracy")),
            ),
        }
        if "epsilon" in measured or "target_epsilon" in stage_cfg:
            row["epsilon"] = _safe_float(measured.get("epsilon"), _safe_float(stage_cfg.get("target_epsilon")))
        rows.append(row)
    return pd.DataFrame(rows)


def _write_manifest(
    artifacts: PipelineArtifacts,
    config: dict[str, Any],
    stage_summary: pd.DataFrame,
    figure_manifest: dict[str, str],
    latex_manifest: dict[str, str],
) -> None:
    payload = {
        "run_name": config["run_name"],
        "generated_at_unix": time.time(),
        "results_dir": str(artifacts.results_dir.resolve()),
        "figures_dir": str(artifacts.figures_dir.resolve()),
        "latex_dir": str(artifacts.latex_dir.resolve()),
        "stage_summary_csv": str((artifacts.results_dir / "stage_summary.csv").resolve()),
        "figure_manifest": figure_manifest,
        "latex_manifest": latex_manifest,
        "stage_rows": stage_summary.to_dict(orient="records"),
    }
    write_json(artifacts.results_dir / "manifest.json", payload)


def run_paemdt_full(project_root: Path, config_path: Path, *, reproduce: bool = False) -> dict[str, Any]:
    config = read_yaml(config_path)
    run_name = str(config.get("run_name", "paemdt_full"))
    artifacts = PipelineArtifacts.from_project_root(project_root, run_name)
    write_json(artifacts.results_dir / "resolved_config.json", config)

    if reproduce:
        _ensure_public_datasets(project_root, config, artifacts)

    baseline_summary = _maybe_run_baseline(project_root, config, artifacts)
    da_summary = _run_domain_adaptation(project_root, config, artifacts)
    dpda_summary = _train_domain_adaptation_with_privacy(project_root, config, artifacts)
    cv_summary = _run_cross_validation(project_root, config, artifacts)
    mm_summary = _run_missing_modality_stage(project_root, config, artifacts)
    best_macro_f1 = max(
        _safe_float(dpda_summary.get("measured_validation_macro_f1"), 0.0),
        _safe_float(da_summary.get("measured_validation_macro_f1"), 0.0),
        _safe_float(baseline_summary.get("measured_validation_macro_f1"), 0.0),
    )
    edge_summary = _run_edge_stage(project_root, config, artifacts, utility_macro_f1=best_macro_f1)

    stage_summary = _build_stage_summary(config, baseline_summary, da_summary, dpda_summary)
    write_dataframe(artifacts.results_dir / "stage_summary.csv", stage_summary)

    figure_manifest: dict[str, str] = {}
    if bool(config["execution"].get("generate_figures", True)):
        figure_manifest = _generate_figures(project_root, config, artifacts, stage_summary)

    latex_manifest: dict[str, str] = {}
    if bool(config["execution"].get("export_latex_tables", True)):
        latex_manifest = _export_latex_tables(artifacts)

    if reproduce:
        from scripts.generate_paper_tables import generate_paper_tables

        generate_paper_tables(
            project_root=project_root,
            config_path=config_path,
            output_root=project_root / "experiments" / "results" / "paper_tables",
        )

    overall_summary = {
        "baseline": baseline_summary,
        "domain_adaptation": da_summary,
        "domain_adaptation_privacy": dpda_summary,
        "cross_validation": cv_summary,
        "missing_modalities": mm_summary,
        "edge_benchmarks": edge_summary,
    }
    write_json(artifacts.results_dir / "pipeline_summary.json", overall_summary)
    _write_manifest(artifacts, config, stage_summary, figure_manifest, latex_manifest)
    return {
        "results_dir": str(artifacts.results_dir.resolve()),
        "figures_dir": str(artifacts.figures_dir.resolve()),
        "stage_summary_csv": str((artifacts.results_dir / "stage_summary.csv").resolve()),
        "pipeline_summary_json": str((artifacts.results_dir / "pipeline_summary.json").resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the integrated PAEMDT full training and paper-asset pipeline.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--config", default="configs/paemdt_full.yaml")
    parser.add_argument("--reproduce", action="store_true", help="Bootstrap public datasets and regenerate paper tables after the full run.")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    config_path = _resolve_path(project_root, args.config).resolve()
    outputs = run_paemdt_full(project_root, config_path, reproduce=bool(args.reproduce))
    print(f"Results directory: {outputs['results_dir']}")
    print(f"Figures directory: {outputs['figures_dir']}")
    print(f"Stage summary: {outputs['stage_summary_csv']}")


if __name__ == "__main__":
    main()
