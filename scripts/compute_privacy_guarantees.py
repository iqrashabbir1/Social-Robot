from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.common.io_utils import write_dataframe, write_json
from src.common.paths import Paper1Paths
from src.data.dataset_loader import load_dataset_records, materialize_frame_records
from src.privacy.dp_engine import compute_epsilon_for_epochs
from src.training.dp_trainer import NON_PRIVATE_BASELINE_VALIDATION_ACCURACY


def _resolve_history_path(project_root: Path, path_arg: str | None, output_subdir: str) -> Path:
    if path_arg:
        candidate = Path(path_arg)
        return candidate if candidate.is_absolute() else (project_root / candidate)
    return project_root / "outputs" / "csv" / "paper1" / output_subdir / "training_history.csv"


def _resolve_metrics_path(project_root: Path, path_arg: str | None, output_subdir: str) -> Path:
    if path_arg:
        candidate = Path(path_arg)
        return candidate if candidate.is_absolute() else (project_root / candidate)
    return project_root / "outputs" / "csv" / "paper1" / output_subdir / "dp_validation_metrics.csv"


def _compute_steps_per_epoch(
    *,
    project_root: Path,
    dataset_root: Path,
    labels_csv: Path,
    target_label_set: str,
    batch_size: int,
    image_size: int,
    random_seed: int,
) -> tuple[int, int]:
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
        cache_dir=project_root / "outputs" / "cache" / "privacy_budget_probe",
        width=image_size,
        height=image_size,
    )
    train_df = materialized_df.loc[materialized_df["split"] == "train"].copy()
    train_samples = int(len(train_df))
    steps_per_epoch = max(1, (train_samples + batch_size - 1) // batch_size)
    return steps_per_epoch, train_samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute DP privacy-budget schedules and privacy-utility tables.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--dataset-root", default="data/public/RAVDESS")
    parser.add_argument("--labels-csv", default="data/public/RAVDESS/labels_broad4_angry.csv")
    parser.add_argument("--output-subdir", default="dp_cnn_small")
    parser.add_argument("--target-label-set", default="broad4_angry")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--noise-multiplier", type=float, default=1.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--epochs-list", default="10,20,50")
    parser.add_argument("--history-csv", default=None)
    parser.add_argument("--metrics-csv", default=None)
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    dataset_root = (project_root / args.dataset_root).resolve()
    labels_csv = (project_root / args.labels_csv).resolve()
    paths = Paper1Paths.from_project_root(project_root)
    paths.ensure()

    epochs_list = [int(item.strip()) for item in str(args.epochs_list).split(",") if item.strip()]
    history_path = _resolve_history_path(project_root, args.history_csv, args.output_subdir)
    metrics_path = _resolve_metrics_path(project_root, args.metrics_csv, args.output_subdir)

    steps_per_epoch, train_samples = _compute_steps_per_epoch(
        project_root=project_root,
        dataset_root=dataset_root,
        labels_csv=labels_csv,
        target_label_set=args.target_label_set,
        batch_size=args.batch_size,
        image_size=args.image_size,
        random_seed=args.random_seed,
    )

    privacy_rows: list[dict[str, float | int]] = []
    for epochs in epochs_list:
        privacy_rows.append(
            {
                "epochs": epochs,
                "steps_per_epoch": steps_per_epoch,
                "train_samples": train_samples,
                "epsilon_delta_1e5": round(
                    compute_epsilon_for_epochs(
                        epochs=epochs,
                        steps_per_epoch=steps_per_epoch,
                        noise_multiplier=args.noise_multiplier,
                        delta=1e-5,
                    ),
                    4,
                ),
                "epsilon_delta_1e6": round(
                    compute_epsilon_for_epochs(
                        epochs=epochs,
                        steps_per_epoch=steps_per_epoch,
                        noise_multiplier=args.noise_multiplier,
                        delta=1e-6,
                    ),
                    4,
                ),
                "noise_multiplier": args.noise_multiplier,
                "max_grad_norm": args.max_grad_norm,
            }
        )
    privacy_df = pd.DataFrame(privacy_rows)

    utility_rows: list[dict[str, float | int | str]] = []
    if history_path.exists():
        history_df = pd.read_csv(history_path)
        for row in history_df.to_dict(orient="records"):
            utility_rows.append(
                {
                    "epoch": int(row["epoch"]),
                    "epsilon": round(float(row.get("epsilon", 0.0)), 4),
                    "val_accuracy": round(float(row.get("val_accuracy", 0.0)), 4),
                    "val_macro_f1": round(float(row.get("val_macro_f1", 0.0)), 4),
                    "baseline_accuracy": NON_PRIVATE_BASELINE_VALIDATION_ACCURACY,
                    "accuracy_drop_vs_baseline": round(
                        NON_PRIVATE_BASELINE_VALIDATION_ACCURACY - float(row.get("val_accuracy", 0.0)),
                        4,
                    ),
                    "privacy_source": row.get("privacy_source", "unknown"),
                }
            )
    else:
        for epochs in epochs_list:
            utility_rows.append(
                {
                    "epoch": epochs,
                    "epsilon": round(
                        compute_epsilon_for_epochs(
                            epochs=epochs,
                            steps_per_epoch=steps_per_epoch,
                            noise_multiplier=args.noise_multiplier,
                            delta=1e-5,
                        ),
                        4,
                    ),
                    "val_accuracy": None,
                    "val_macro_f1": None,
                    "baseline_accuracy": NON_PRIVATE_BASELINE_VALIDATION_ACCURACY,
                    "accuracy_drop_vs_baseline": None,
                    "privacy_source": "history_csv_missing",
                }
            )
    utility_df = pd.DataFrame(utility_rows)

    metrics_summary = None
    if metrics_path.exists():
        metrics_summary = pd.read_csv(metrics_path).to_dict(orient="records")

    csv_dir = paths.outputs_csv_paper1 / args.output_subdir
    csv_dir.mkdir(parents=True, exist_ok=True)
    privacy_table_path = csv_dir / "privacy_budget_table.csv"
    utility_table_path = csv_dir / "privacy_utility_tradeoff.csv"
    write_dataframe(privacy_table_path, privacy_df)
    write_dataframe(utility_table_path, utility_df)
    write_dataframe(paths.outputs_tables / "paper1_table_privacy_budget.csv", privacy_df)
    write_dataframe(paths.outputs_tables / "paper1_table_privacy_utility_tradeoff.csv", utility_df)

    summary_payload = {
        "dataset_root": str(dataset_root),
        "labels_csv": str(labels_csv),
        "train_samples": train_samples,
        "steps_per_epoch": steps_per_epoch,
        "noise_multiplier": args.noise_multiplier,
        "max_grad_norm": args.max_grad_norm,
        "epochs_list": epochs_list,
        "history_csv_exists": history_path.exists(),
        "metrics_csv_exists": metrics_path.exists(),
        "metrics_summary": metrics_summary,
        "outputs": {
            "privacy_budget_table_csv": str(privacy_table_path.resolve()),
            "privacy_utility_tradeoff_csv": str(utility_table_path.resolve()),
        },
    }
    write_json(csv_dir / "privacy_guarantees_summary.json", summary_payload)
    print(f"Privacy budget table written to: {privacy_table_path}")
    print(f"Privacy-utility tradeoff table written to: {utility_table_path}")


if __name__ == "__main__":
    main()
