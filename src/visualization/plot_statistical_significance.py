from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save_all(fig: plt.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")


def plot_repeated_cv_confidence_intervals(summary_csv: Path, output_base: Path) -> dict[str, Path]:
    df = pd.read_csv(summary_csv)
    rename_map = {
        "Model": "model",
        "Mean_Val_Accuracy": "mean_val_accuracy",
        "CI_Low": "ci_low",
        "CI_High": "ci_high",
    }
    df = df.rename(columns={key: value for key, value in rename_map.items() if key in df.columns})
    required = {"model", "mean_val_accuracy", "ci_low", "ci_high"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {summary_csv}: {sorted(missing)}")

    plot_df = df.copy()
    for column in ["mean_val_accuracy", "ci_low", "ci_high"]:
        plot_df[column] = pd.to_numeric(plot_df[column], errors="coerce")
    plot_df = plot_df.dropna(subset=["mean_val_accuracy", "ci_low", "ci_high"])
    plot_df = plot_df.sort_values("mean_val_accuracy", ascending=False).reset_index(drop=True)

    means = plot_df["mean_val_accuracy"].to_numpy(dtype=float)
    err_low = means - plot_df["ci_low"].to_numpy(dtype=float)
    err_high = plot_df["ci_high"].to_numpy(dtype=float) - means
    y_positions = np.arange(len(plot_df))

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.errorbar(
        means,
        y_positions,
        xerr=np.vstack([err_low, err_high]),
        fmt="o",
        color="#1b4f72",
        ecolor="#1b4f72",
        elinewidth=1.4,
        capsize=4,
        markersize=5,
    )

    for y_pos, row in zip(y_positions, plot_df.to_dict("records")):
        label = f"{row['mean_val_accuracy']:.1f} [{row['ci_low']:.1f}, {row['ci_high']:.1f}]"
        ax.text(
            float(row["ci_high"]) + 0.35,
            y_pos,
            label,
            va="center",
            ha="left",
            fontsize=8,
            fontweight="bold",
            color="#222222",
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(plot_df["model"].tolist())
    ax.invert_yaxis()
    ax.set_xlim(80.0, 100.0)
    ax.set_xlabel("Validation accuracy (%)", fontweight="bold")
    ax.set_title("Figure 6. Repeated CV Confidence Intervals", fontweight="bold", pad=10)
    ax.grid(axis="x", linestyle=":", linewidth=0.7, alpha=0.45)
    ax.grid(axis="y", visible=False)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    _save_all(fig, output_base)
    plt.close(fig)

    return {
        "png": output_base.with_suffix(".png"),
        "pdf": output_base.with_suffix(".pdf"),
        "svg": output_base.with_suffix(".svg"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Figure 6 repeated-CV confidence intervals.")
    parser.add_argument("--summary-csv", default="outputs/tables/repeated_cv_summary.csv")
    parser.add_argument("--output-base", default="outputs/figures/Figure_6_Repeated_CV_Confidence_Intervals")
    args = parser.parse_args()

    paths = plot_repeated_cv_confidence_intervals(
        summary_csv=Path(args.summary_csv).resolve(),
        output_base=Path(args.output_base).resolve(),
    )
    print(f"Figure 6 PNG: {paths['png']}")
    print(f"Figure 6 PDF: {paths['pdf']}")
    print(f"Figure 6 SVG: {paths['svg']}")


if __name__ == "__main__":
    main()
