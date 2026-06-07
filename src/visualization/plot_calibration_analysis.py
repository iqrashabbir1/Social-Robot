from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _style() -> None:
    plt.rcParams.update({"font.family": "Arial", "font.size": 10, "font.weight": "bold", "axes.titleweight": "bold", "axes.labelweight": "bold"})


def _save(fig: plt.Figure, base: Path) -> dict[str, Path]:
    base.parent.mkdir(parents=True, exist_ok=True)
    paths = {"png": base.with_suffix(".png"), "pdf": base.with_suffix(".pdf"), "svg": base.with_suffix(".svg")}
    fig.savefig(paths["png"], dpi=600, bbox_inches="tight")
    fig.savefig(paths["pdf"], bbox_inches="tight")
    fig.savefig(paths["svg"], bbox_inches="tight")
    plt.close(fig)
    return paths


def plot_calibration_analysis(source_csv: Path, output_base: Path) -> dict[str, Path]:
    _style()
    df = pd.read_csv(source_csv)
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    colors = ["#228833", "#4477AA", "#CC6677", "#CCBB44"]
    bars = ax.bar(df["confidence_profile"], df["ECE"], color=colors, edgecolor="#222222", linewidth=0.8)
    threshold = 0.05
    ax.axhline(threshold, color="#222222", linestyle="--", linewidth=1.2)
    ax.text(0.03, threshold + 0.003, "Acceptable ECE threshold = 0.05", transform=ax.get_yaxis_transform(), ha="left", va="bottom", fontsize=8, fontweight="bold", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 2})
    for bar in bars:
        value = float(bar.get_height())
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.004, f"{value:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_ylim(0, 0.15)
    ax.set_ylabel("Expected calibration error (ECE)")
    ax.set_xlabel("Evaluated confidence profile")
    ax.set_title("Figure 7. Expected Calibration Error Comparison")
    ax.tick_params(axis="x", rotation=12)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.55)
    fig.tight_layout()
    return _save(fig, output_base)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Paper 1 Figure 7 calibration analysis.")
    parser.add_argument("--source-csv", default="outputs/csv/calibration_results.csv")
    parser.add_argument("--output-base", default="outputs/figures/Figure_7_ECE_Comparison")
    args = parser.parse_args()
    for label, path in plot_calibration_analysis(Path(args.source_csv).resolve(), Path(args.output_base).resolve()).items():
        print(f"Figure 7 {label.upper()}: {path}")


if __name__ == "__main__":
    main()
