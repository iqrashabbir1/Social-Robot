from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.public_dataset_presets import build_ravdess_labels_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a Paper 1 label CSV for RAVDESS.")
    parser.add_argument("--dataset-root", required=True, help="Root folder of the extracted RAVDESS subset")
    parser.add_argument("--output-csv", required=True, help="Where to write the generated labels CSV")
    parser.add_argument("--target-label-set", default="broad4_angry", choices=["broad4_angry", "paper1_legacy4"])
    args = parser.parse_args()

    output_path = build_ravdess_labels_csv(
        dataset_root=Path(args.dataset_root),
        output_csv=Path(args.output_csv),
        target_label_set=args.target_label_set,
    )
    print(output_path)


if __name__ == "__main__":
    main()
