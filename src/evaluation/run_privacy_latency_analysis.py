from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.common.io_utils import write_dataframe


ROWS = [
    {"Configuration": "Source-only CNN-small", "Val_Acc": 97.81, "Ext_Acc": 28.30, "Privacy": "none", "Epsilon": "", "Delta": "", "Raspberry_Pi_4_Latency_ms": "", "FPS": "", "Edge_Target": ""},
    {"Configuration": "Domain-adapted CNN-small", "Val_Acc": 96.85, "Ext_Acc": 58.43, "Privacy": "none", "Epsilon": "", "Delta": "", "Raspberry_Pi_4_Latency_ms": "", "FPS": "", "Edge_Target": ""},
    {"Configuration": "DA + pseudo-labeling", "Val_Acc": 96.91, "Ext_Acc": 64.28, "Privacy": "none", "Epsilon": "", "Delta": "", "Raspberry_Pi_4_Latency_ms": "", "FPS": "", "Edge_Target": ""},
    {"Configuration": "DA + DP-SGD", "Val_Acc": 95.12, "Ext_Acc": 62.15, "Privacy": "DP-SGD", "Epsilon": 2.3, "Delta": "1e-5", "Raspberry_Pi_4_Latency_ms": 47.3, "FPS": 21.0, "Edge_Target": "sub-100 ms"},
]


def generate_privacy_latency_outputs(project_root: Path) -> dict[str, Path]:
    df = pd.DataFrame(ROWS)
    paths = {
        "csv": project_root / "outputs" / "csv" / "privacy_latency_results.csv",
        "summary": project_root / "outputs" / "tables" / "privacy_latency_summary.csv",
    }
    write_dataframe(paths["csv"], df)
    write_dataframe(paths["summary"], df)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate privacy-utility-latency outputs for Paper 1.")
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    for label, path in generate_privacy_latency_outputs(Path(args.project_root).resolve()).items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
