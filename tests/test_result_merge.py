from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.evaluation.benchmark_runner import merge_result_tables


def test_merge_result_tables_sorts_by_primary_metrics(tmp_path: Path) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    pd.DataFrame(
        [
            {"experiment_name": "a", "accuracy": 0.8, "macro_f1": 0.78, "weighted_f1": 0.79},
        ]
    ).to_csv(first, index=False)
    pd.DataFrame(
        [
            {"experiment_name": "b", "accuracy": 0.9, "macro_f1": 0.88, "weighted_f1": 0.87},
        ]
    ).to_csv(second, index=False)

    merged = merge_result_tables([first, second])

    assert list(merged["experiment_name"]) == ["b", "a"]
