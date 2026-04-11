from __future__ import annotations

import argparse
from pathlib import Path

from src.common.io_utils import write_json
from src.evaluation.build_local_public_comparison import build_local_public_comparison
from src.models.vision.evaluate_image_emotion_classifier import evaluate_image_emotion_classifier
from src.models.vision.train_image_emotion_classifier import train_image_emotion_classifier


def _parse_external_dataset(value: str) -> tuple[str, Path, Path]:
    parts = value.split("::")
    if len(parts) != 3:
        raise ValueError(
            "Each --external-dataset value must be in the form "
            "'dataset_name::dataset_root::labels_csv'."
        )
    dataset_name, dataset_root, labels_csv = parts
    return dataset_name.strip(), Path(dataset_root), Path(labels_csv)


def run_cross_dataset_generalization(
    project_root: Path,
    train_dataset_root: Path,
    train_labels_csv: Path,
    external_datasets: list[tuple[str, Path, Path]],
    output_subdir: str = "ravdess_generalization_suite",
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
    project_root = project_root.resolve()
    train_outputs = train_image_emotion_classifier(
        project_root=project_root,
        dataset_root=train_dataset_root,
        labels_csv=train_labels_csv,
        output_subdir=output_subdir,
        target_label_set=target_label_set,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        image_size=image_size,
        random_seed=random_seed,
        device=device,
        log_every_epochs=log_every_epochs,
        log_every_steps=log_every_steps,
    )

    public_metrics_paths: list[Path] = []
    external_output_map: dict[str, dict[str, str]] = {}
    for dataset_name, dataset_root, labels_csv in external_datasets:
        eval_subdir = f"{output_subdir}_{dataset_name.lower()}_eval"
        eval_outputs = evaluate_image_emotion_classifier(
            project_root=project_root,
            model_path=Path(train_outputs["model_path"]),
            dataset_root=dataset_root,
            labels_csv=labels_csv,
            output_subdir=eval_subdir,
            target_label_set=target_label_set,
            batch_size=batch_size,
            device=device,
        )
        public_metrics_paths.append(Path(eval_outputs["metrics_csv"]))
        external_output_map[dataset_name] = eval_outputs

    comparison_path = build_local_public_comparison(
        project_root=project_root,
        local_metrics_csv=Path(train_outputs["metrics_csv"]),
        public_metrics_csvs=public_metrics_paths,
        comparison_group="ravdess_train_then_external_public_test",
        output_path=project_root / "outputs" / "tables" / "paper1_table_local_vs_public_metrics.csv",
    )

    manifest_path = project_root / "outputs" / "csv" / "paper1" / output_subdir / "cross_dataset_manifest.json"
    write_json(
        manifest_path,
        {
            "train_outputs": train_outputs,
            "external_outputs": external_output_map,
            "comparison_table": str(comparison_path),
        },
    )
    return {
        "comparison_table": str(comparison_path),
        "manifest_json": str(manifest_path),
        **train_outputs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train on one dataset and evaluate the trained model on multiple public datasets.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--train-dataset-root", required=True)
    parser.add_argument("--train-labels-csv", required=True)
    parser.add_argument("--external-dataset", action="append", required=True)
    parser.add_argument("--output-subdir", default="ravdess_generalization_suite")
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

    outputs = run_cross_dataset_generalization(
        project_root=Path(args.project_root),
        train_dataset_root=Path(args.train_dataset_root),
        train_labels_csv=Path(args.train_labels_csv),
        external_datasets=[_parse_external_dataset(value) for value in args.external_dataset],
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
    print(outputs["comparison_table"])


if __name__ == "__main__":
    main()
