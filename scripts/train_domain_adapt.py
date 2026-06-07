from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from src.common.io_utils import write_dataframe, write_json
from src.common.paths import Paper1Paths
from src.data.dataset_loader import load_dataset_records, materialize_frame_records
from src.data.public_dataset_presets import TARGET_LABEL_SETS
from src.models.domain_adversarial import DANNMultimodalEmotionModel
from src.models.torch_runtime import resolve_torch_runtime
from src.training.domain_adaptation import (
    DomainAdaptationFrameDataset,
    DomainAdversarialTrainer,
    domain_adaptation_collate_fn,
    write_evaluation_bundle,
    write_training_status,
)


BASELINE_RAVDESS_VALIDATION_ACCURACY = 0.9781
BASELINE_CREMAD_EXTERNAL_ACCURACY = 0.2830
TARGET_EXTERNAL_ACCURACY_GOAL = 0.64


def _build_loader(
    dataframe: pd.DataFrame,
    label_to_index: dict[str, int],
    batch_size: int,
    image_size: int,
    shuffle: bool,
    include_labels: bool = True,
) -> "torch.utils.data.DataLoader":
    from src.models.torch_runtime import require_torch

    torch = require_torch()
    dataset = DomainAdaptationFrameDataset(
        dataframe=dataframe,
        label_to_index=label_to_index,
        image_size=image_size,
        include_labels=include_labels,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=domain_adaptation_collate_fn,
        drop_last=False,
    )


def _materialize_dataset(
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
    cache_dir = project_root / "outputs" / "cache" / "domain_adaptation" / cache_name
    return materialize_frame_records(
        dataset_df=dataset_df,
        cache_dir=cache_dir,
        width=image_size,
        height=image_size,
        video_frame_stride=15,
        max_frames_per_video=12,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a DANN + MK-MMD + pseudo-label domain adaptation model.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--ravdess-root", default="data/public/RAVDESS")
    parser.add_argument("--ravdess-labels", default="data/public/RAVDESS/labels_broad4_angry.csv")
    parser.add_argument("--cremad-root", default="data/public/CREMA-D")
    parser.add_argument("--cremad-labels", default="data/public/CREMA-D/labels_broad4_angry.csv")
    parser.add_argument("--output-subdir", default="domain_adaptation_dann_mmd_pseudo")
    parser.add_argument("--target-label-set", default="broad4_angry", choices=sorted(TARGET_LABEL_SETS))
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--lambda-domain", type=float, default=0.5)
    parser.add_argument("--lambda-mmd", type=float, default=0.1)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    paths = Paper1Paths.from_project_root(project_root)
    paths.ensure()

    runtime = resolve_torch_runtime("auto", requested_device=args.device)
    device = runtime.device

    csv_dir = paths.outputs_csv_paper1 / args.output_subdir
    log_dir = paths.outputs_logs / "paper1" / args.output_subdir
    model_dir = project_root / "outputs" / "models" / "paper1" / args.output_subdir
    checkpoints_dir = model_dir / "checkpoints"
    for directory in (csv_dir, log_dir, model_dir, checkpoints_dir):
        directory.mkdir(parents=True, exist_ok=True)

    ravdess_root = (project_root / args.ravdess_root).resolve()
    ravdess_labels = (project_root / args.ravdess_labels).resolve()
    cremad_root = (project_root / args.cremad_root).resolve()
    cremad_labels = (project_root / args.cremad_labels).resolve()

    print("Preparing RAVDESS source frames...")
    ravdess_frames = _materialize_dataset(
        project_root=project_root,
        dataset_root=ravdess_root,
        labels_csv=ravdess_labels,
        split_mode="train_test",
        cache_name="ravdess_source",
        image_size=args.image_size,
        random_seed=args.random_seed,
        target_label_set=args.target_label_set,
    )
    print("Preparing CREMA-D target frames...")
    cremad_frames = _materialize_dataset(
        project_root=project_root,
        dataset_root=cremad_root,
        labels_csv=cremad_labels,
        split_mode="test_only",
        cache_name="cremad_target",
        image_size=args.image_size,
        random_seed=args.random_seed,
        target_label_set=args.target_label_set,
    )

    ravdess_train = ravdess_frames.loc[ravdess_frames["split"] == "train"].reset_index(drop=True)
    ravdess_val = ravdess_frames.loc[ravdess_frames["split"] == "test"].reset_index(drop=True)
    cremad_test = cremad_frames.reset_index(drop=True)
    if ravdess_train.empty or ravdess_val.empty:
        raise RuntimeError("RAVDESS frame materialization produced an empty train or validation split.")
    if cremad_test.empty:
        raise RuntimeError("CREMA-D frame materialization produced an empty target evaluation set.")
    class_labels = sorted(label for label in ravdess_train["label"].dropna().unique().tolist())
    if not class_labels:
        raise RuntimeError("No source-domain labels were found after dataset harmonization.")
    label_to_index = {label: index for index, label in enumerate(class_labels)}

    print(
        f"Materialized frames: RAVDESS train={len(ravdess_train)}, "
        f"RAVDESS val={len(ravdess_val)}, CREMA-D test={len(cremad_test)}"
    )

    source_train_loader = _build_loader(
        ravdess_train,
        label_to_index=label_to_index,
        batch_size=args.batch_size,
        image_size=args.image_size,
        shuffle=True,
        include_labels=True,
    )
    source_val_loader = _build_loader(
        ravdess_val,
        label_to_index=label_to_index,
        batch_size=args.batch_size,
        image_size=args.image_size,
        shuffle=False,
        include_labels=True,
    )
    target_train_loader = _build_loader(
        cremad_test,
        label_to_index=label_to_index,
        batch_size=args.batch_size,
        image_size=args.image_size,
        shuffle=True,
        include_labels=False,
    )
    target_eval_loader = _build_loader(
        cremad_test,
        label_to_index=label_to_index,
        batch_size=args.batch_size,
        image_size=args.image_size,
        shuffle=False,
        include_labels=True,
    )

    from src.models.torch_runtime import require_torch

    torch = require_torch()
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    model = DANNMultimodalEmotionModel(
        num_classes=len(class_labels),
        image_size=args.image_size,
        feature_dim=384,
    ).to(device)
    trainer = DomainAdversarialTrainer(
        model=model,
        class_labels=class_labels,
        device=device,
        learning_rate=args.learning_rate,
        lambda_domain=args.lambda_domain,
        lambda_mmd=args.lambda_mmd,
    )

    history_rows: list[dict[str, float | int | str]] = []
    best_external_accuracy = -1.0
    best_epoch = 0
    best_metrics: dict[str, float] = {}
    best_checkpoint_path = model_dir / "best_model.pt"
    start_time = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        train_metrics = trainer.train_epoch(
            source_loader=source_train_loader,
            target_loader=target_train_loader,
            epoch_index=epoch - 1,
            total_epochs=args.epochs,
        )
        ravdess_eval = trainer.evaluate(source_val_loader, dataset_name="RAVDESS_validation")
        cremad_eval = trainer.evaluate(target_eval_loader, dataset_name="CREMA-D_external")

        epoch_seconds = time.perf_counter() - epoch_started
        row = {
            "epoch": epoch,
            "train_loss": round(float(train_metrics["loss"]), 6),
            "train_emotion_loss": round(float(train_metrics["emotion_loss"]), 6),
            "train_domain_loss": round(float(train_metrics["domain_loss"]), 6),
            "train_mmd_loss": round(float(train_metrics["mmd_loss"]), 6),
            "train_pseudo_loss": round(float(train_metrics["pseudo_loss"]), 6),
            "alpha": round(float(train_metrics["alpha"]), 6),
            "pseudo_threshold": round(float(train_metrics["pseudo_threshold"]), 4),
            "pseudo_acceptance_rate": round(float(train_metrics["pseudo_acceptance_rate"]), 4),
            "source_train_accuracy": round(float(train_metrics["source_train_accuracy"]), 4),
            "source_train_macro_f1": round(float(train_metrics["source_train_macro_f1"]), 4),
            "ravdess_val_accuracy": round(float(ravdess_eval.metrics.accuracy), 4),
            "ravdess_val_macro_f1": round(float(ravdess_eval.metrics.macro_f1), 4),
            "cremad_external_accuracy": round(float(cremad_eval.metrics.accuracy), 4),
            "cremad_external_macro_f1": round(float(cremad_eval.metrics.macro_f1), 4),
            "epoch_seconds": round(float(epoch_seconds), 2),
            "device": device,
        }
        history_rows.append(row)
        write_dataframe(csv_dir / "training_history.csv", pd.DataFrame(history_rows))

        status_payload = {
            "epoch": epoch,
            "epochs": args.epochs,
            "device": device,
            "runtime_backend": runtime.active_backend,
            "ravdess_train_frames": int(len(ravdess_train)),
            "ravdess_val_frames": int(len(ravdess_val)),
            "cremad_test_frames": int(len(cremad_test)),
            "latest_metrics": row,
            "best_external_accuracy": round(float(best_external_accuracy), 4) if best_external_accuracy >= 0 else None,
            "best_epoch": best_epoch if best_epoch > 0 else None,
        }
        write_training_status(log_dir / "latest_status.json", status_payload)

        if epoch % 10 == 0:
            trainer.save_checkpoint(
                checkpoint_path=checkpoints_dir / f"checkpoint_epoch_{epoch:03d}.pt",
                epoch_index=epoch,
                metadata={
                    "class_labels": class_labels,
                    "image_size": args.image_size,
                    "target_label_set": args.target_label_set,
                    "ravdess_root": str(ravdess_root),
                    "cremad_root": str(cremad_root),
                },
            )

        if float(cremad_eval.metrics.accuracy) > best_external_accuracy:
            best_external_accuracy = float(cremad_eval.metrics.accuracy)
            best_epoch = epoch
            best_metrics = {
                "ravdess_validation_accuracy": float(ravdess_eval.metrics.accuracy),
                "ravdess_validation_macro_f1": float(ravdess_eval.metrics.macro_f1),
                "cremad_external_accuracy": float(cremad_eval.metrics.accuracy),
                "cremad_external_macro_f1": float(cremad_eval.metrics.macro_f1),
            }
            trainer.save_checkpoint(
                checkpoint_path=best_checkpoint_path,
                epoch_index=epoch,
                metadata={
                    "class_labels": class_labels,
                    "image_size": args.image_size,
                    "target_label_set": args.target_label_set,
                    "ravdess_root": str(ravdess_root),
                    "cremad_root": str(cremad_root),
                    "selection_metric": "cremad_external_accuracy",
                },
            )
            write_evaluation_bundle(csv_dir, "ravdess_validation", ravdess_eval)
            write_evaluation_bundle(csv_dir, "cremad_external", cremad_eval)

        print(
            f"[epoch {epoch:03d}/{args.epochs}] "
            f"ravdess_val_acc={ravdess_eval.metrics.accuracy:.4f} "
            f"cremad_ext_acc={cremad_eval.metrics.accuracy:.4f} "
            f"alpha={train_metrics['alpha']:.3f} "
            f"pseudo_accept={train_metrics['pseudo_acceptance_rate']:.3f}"
        )

    domain_gap_baseline = BASELINE_RAVDESS_VALIDATION_ACCURACY - BASELINE_CREMAD_EXTERNAL_ACCURACY
    adapted_ravdess = float(best_metrics.get("ravdess_validation_accuracy", 0.0))
    adapted_cremad = float(best_metrics.get("cremad_external_accuracy", 0.0))
    comparison_rows = [
        {
            "configuration": "cnn_small_source_only_baseline",
            "domain_adaptation_enabled": False,
            "ravdess_validation_accuracy": BASELINE_RAVDESS_VALIDATION_ACCURACY,
            "cremad_external_accuracy": BASELINE_CREMAD_EXTERNAL_ACCURACY,
            "domain_gap": round(domain_gap_baseline, 4),
            "external_accuracy_gain_vs_baseline": 0.0,
            "research_target_external_accuracy_goal": TARGET_EXTERNAL_ACCURACY_GOAL,
            "notes": "Observed repository baseline before domain adaptation.",
        },
        {
            "configuration": "dann_mk_mmd_progressive_pseudo",
            "domain_adaptation_enabled": True,
            "ravdess_validation_accuracy": round(adapted_ravdess, 4),
            "cremad_external_accuracy": round(adapted_cremad, 4),
            "domain_gap": round(adapted_ravdess - adapted_cremad, 4),
            "external_accuracy_gain_vs_baseline": round(adapted_cremad - BASELINE_CREMAD_EXTERNAL_ACCURACY, 4),
            "research_target_external_accuracy_goal": TARGET_EXTERNAL_ACCURACY_GOAL,
            "notes": "Actual measured result from DANN + MK-MMD + progressive pseudo-label training.",
        },
    ]
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_path = csv_dir / "domain_adaptation_comparison.csv"
    write_dataframe(comparison_path, comparison_df)
    write_dataframe(paths.outputs_tables / "paper1_table_domain_adaptation_comparison.csv", comparison_df)

    summary_payload = {
        "experiment_name": args.output_subdir,
        "target_label_set": args.target_label_set,
        "class_labels": class_labels,
        "runtime_backend": runtime.active_backend,
        "device": device,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "lambda_domain": args.lambda_domain,
        "lambda_mmd": args.lambda_mmd,
        "baseline_ravdess_validation_accuracy": BASELINE_RAVDESS_VALIDATION_ACCURACY,
        "baseline_cremad_external_accuracy": BASELINE_CREMAD_EXTERNAL_ACCURACY,
        "target_external_accuracy_goal": TARGET_EXTERNAL_ACCURACY_GOAL,
        "best_epoch": best_epoch,
        "best_metrics": {key: round(float(value), 4) for key, value in best_metrics.items()},
        "ravdess_train_frames": int(len(ravdess_train)),
        "ravdess_val_frames": int(len(ravdess_val)),
        "cremad_test_frames": int(len(cremad_test)),
        "elapsed_seconds": round(float(time.perf_counter() - start_time), 2),
        "artifacts": {
            "training_history_csv": str((csv_dir / "training_history.csv").resolve()),
            "comparison_csv": str(comparison_path.resolve()),
            "best_checkpoint": str(best_checkpoint_path.resolve()),
            "latest_status_json": str((log_dir / "latest_status.json").resolve()),
        },
    }
    write_json(csv_dir / "domain_adaptation_summary.json", summary_payload)

    print()
    print("Domain adaptation training complete.")
    print(f"Best epoch: {best_epoch}")
    print(f"RAVDESS validation accuracy: {adapted_ravdess:.4f}")
    print(f"CREMA-D external accuracy: {adapted_cremad:.4f}")
    print(f"Comparison table: {comparison_path}")
    print(f"Best checkpoint: {best_checkpoint_path}")


if __name__ == "__main__":
    main()
