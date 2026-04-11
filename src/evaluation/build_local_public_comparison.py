from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.common.io_utils import write_dataframe


def build_local_public_comparison(
    project_root: Path,
    local_metrics_csv: Path,
    public_metrics_csvs: list[Path],
    comparison_group: str = "trained_model_generalization",
    output_path: Path | None = None,
) -> Path:
    project_root = project_root.resolve()
    local_metrics_csv = local_metrics_csv if local_metrics_csv.is_absolute() else (project_root / local_metrics_csv)
    local_df = pd.read_csv(local_metrics_csv.resolve())
    train_dataset_name = str(local_df.iloc[0].get("dataset_name", "unknown")) if not local_df.empty else "unknown"
    frames = [local_df]
    for metrics_csv in public_metrics_csvs:
        resolved = metrics_csv if metrics_csv.is_absolute() else (project_root / metrics_csv)
        frames.append(pd.read_csv(resolved.resolve()))
    combined = pd.concat(frames, ignore_index=True)
    combined.insert(0, "comparison_group", comparison_group)
    combined.insert(1, "train_dataset_name", train_dataset_name)
    combined.insert(2, "test_dataset_name", combined["dataset_name"])
    combined.insert(
        3,
        "evaluation_role",
        combined["evaluation_stage"].map(
            {
                "local_validation": "held_out_validation",
                "public_test": "external_public_test",
            }
        ).fillna("other"),
    )
    if output_path is None:
        output_path = project_root / "outputs" / "tables" / "paper1_table_local_vs_public_metrics.csv"
    else:
        output_path = output_path if output_path.is_absolute() else (project_root / output_path)
    write_dataframe(output_path.resolve(), combined)
    return output_path.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Paper 1 local-vs-public metrics comparison table.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--local-metrics-csv", required=True)
    parser.add_argument("--public-metrics-csv", nargs="+", required=True)
    parser.add_argument("--comparison-group", default="trained_model_generalization")
    parser.add_argument("--output-path", default="outputs/tables/paper1_table_local_vs_public_metrics.csv")
    args = parser.parse_args()
    output_path = build_local_public_comparison(
        project_root=Path(args.project_root).resolve(),
        local_metrics_csv=Path(args.local_metrics_csv),
        public_metrics_csvs=[Path(value) for value in args.public_metrics_csv],
        comparison_group=args.comparison_group,
        output_path=Path(args.output_path),
    )
    print(output_path)


if __name__ == "__main__":
    main()
