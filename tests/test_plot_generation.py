from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.visualization.plot_cs3 import plot_model_comparison


def test_plot_generation_writes_png_and_svg(tmp_path: Path) -> None:
    csv_path = tmp_path / "summary.csv"
    pd.DataFrame(
        [
            {"model_id": "B0", "accuracy": 0.8, "macro_f1": 0.75, "weighted_f1": 0.78},
            {"model_id": "B1", "accuracy": 0.9, "macro_f1": 0.88, "weighted_f1": 0.89},
        ]
    ).to_csv(csv_path, index=False)

    output_base = tmp_path / "model_plot"
    plot_model_comparison(csv_path, output_base)

    assert output_base.with_suffix(".png").exists()
    assert output_base.with_suffix(".svg").exists()
