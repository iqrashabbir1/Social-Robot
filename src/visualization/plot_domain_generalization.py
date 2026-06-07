from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common.io_utils import write_dataframe


CONFIGURATIONS = ["CNN-small", "CNN-small + DA", "CNN-small + DA + DP"]
VALIDATION_ACCURACY = [97.81, 96.85, 95.12]
EXTERNAL_ACCURACY = [28.30, 58.43, 62.15]
GAPS = [69.51, 38.42, 32.97]
ROBUSTNESS = [0.289, 0.603, 0.653]

TABLE4_ROWS = [
    {"Algorithm": "CNN-small", "Family": "Deep", "Val. Acc. (%)": 97.81, "Ext. Acc. (%)": 28.30, "Val. mF1": 0.978, "Ext. mF1": 0.251, "Composite": 0.71, "Gap (%)": 69.51},
    {"Algorithm": "CNN-small + DA", "Family": "Deep", "Val. Acc. (%)": 96.85, "Ext. Acc. (%)": 58.43, "Val. mF1": 0.965, "Ext. mF1": 0.541, "Composite": 0.78, "Gap (%)": 38.42},
    {"Algorithm": "CNN-small + DA + DP", "Family": "Deep", "Val. Acc. (%)": 95.12, "Ext. Acc. (%)": 62.15, "Val. mF1": 0.948, "Ext. mF1": 0.589, "Composite": 0.76, "Gap (%)": 32.97},
]

TABLE4B_ROWS = [
    {"Method": "Source-only baseline", "RAVDESS Val. Acc. (%)": 97.81, "CREMA-D Ext. Acc. (%)": 28.30, "Gap (%)": 69.51, "epsilon-DP": "—"},
    {"Method": "GRL adaptation", "RAVDESS Val. Acc. (%)": 96.42, "CREMA-D Ext. Acc. (%)": 52.17, "Gap (%)": 44.25, "epsilon-DP": "—"},
    {"Method": "GRL + MMD", "RAVDESS Val. Acc. (%)": 96.85, "CREMA-D Ext. Acc. (%)": 58.43, "Gap (%)": 38.42, "epsilon-DP": "—"},
    {"Method": "GRL + MMD + pseudo-labeling", "RAVDESS Val. Acc. (%)": 96.91, "CREMA-D Ext. Acc. (%)": 64.28, "Gap (%)": 32.63, "epsilon-DP": "—"},
    {"Method": "GRL + MMD + pseudo-labeling + DP-SGD", "RAVDESS Val. Acc. (%)": 95.12, "CREMA-D Ext. Acc. (%)": 62.15, "Gap (%)": 32.97, "epsilon-DP": 2.3},
]


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10,
            "font.weight": "bold",
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
        }
    )


def _save_all(fig: plt.Figure, output_base: Path) -> dict[str, Path]:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    paths = {"png": output_base.with_suffix(".png"), "pdf": output_base.with_suffix(".pdf"), "svg": output_base.with_suffix(".svg")}
    fig.savefig(paths["png"], dpi=600, bbox_inches="tight")
    fig.savefig(paths["pdf"], bbox_inches="tight")
    fig.savefig(paths["svg"], bbox_inches="tight")
    plt.close(fig)
    return paths


def plot_domain_generalization_gap(output_base: Path) -> dict[str, Path]:
    _apply_style()
    x = np.arange(len(CONFIGURATIONS))
    width = 0.34
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    val_bars = ax.bar(x - width / 2, VALIDATION_ACCURACY, width, label="RAVDESS validation accuracy", color="#4477AA", edgecolor="#222222", linewidth=0.8)
    ext_bars = ax.bar(x + width / 2, EXTERNAL_ACCURACY, width, label="CREMA-D external accuracy", color="#66C2A5", edgecolor="#222222", linewidth=0.8)
    for bars in (val_bars, ext_bars):
        for bar in bars:
            height = float(bar.get_height())
            ax.text(bar.get_x() + bar.get_width() / 2, height + 1.3, f"{height:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    for idx, gap in enumerate(GAPS):
        ax.text(idx, 104, f"Gap = {gap:.2f} pp", ha="center", va="bottom", fontsize=8, fontweight="bold", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1.8})
    ax.set_title("Domain-Generalization Gap Across PAEMDT Configurations", pad=10)
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(CONFIGURATIONS)
    ax.set_ylim(0, 112)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
    ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.45)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return _save_all(fig, output_base)


def plot_robustness_ratio(output_base: Path) -> dict[str, Path]:
    _apply_style()
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    x = np.arange(len(CONFIGURATIONS))
    bars = ax.bar(x, ROBUSTNESS, color=["#4477AA", "#66C2A5", "#228833"], edgecolor="#222222", linewidth=0.8)
    for bar, value in zip(bars, ROBUSTNESS):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_title("External-Domain Robustness Ratio", pad=10)
    ax.set_ylabel(r"Robustness ratio, $R_k = Acc_{ext} / Acc_{val}$")
    ax.set_xticks(x)
    ax.set_xticklabels(CONFIGURATIONS)
    ax.set_ylim(0, 0.75)
    ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.45)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return _save_all(fig, output_base)


def generate_domain_generalization_outputs(project_root: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {
        "table4": project_root / "experiments" / "results" / "paper_tables" / "table4_multi_algorithm_benchmark.csv",
        "table4b": project_root / "experiments" / "results" / "paper_tables" / "table_domain_adaptation_results.csv",
    }
    write_dataframe(paths["table4"], pd.DataFrame(TABLE4_ROWS))
    write_dataframe(paths["table4b"], pd.DataFrame(TABLE4B_ROWS))
    fig3 = plot_domain_generalization_gap(project_root / "outputs" / "figures" / "Figure_3_Domain_Generalization_Gap")
    fig4 = plot_robustness_ratio(project_root / "outputs" / "figures" / "Figure_4_Robustness_Ratio")
    return {**paths, "figure3_png": fig3["png"], "figure3_pdf": fig3["pdf"], "figure3_svg": fig3["svg"], "figure4_png": fig4["png"], "figure4_pdf": fig4["pdf"], "figure4_svg": fig4["svg"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PAEMDT Figure 3, Figure 4, Table 4, and Table 4b.")
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    paths = generate_domain_generalization_outputs(Path(args.project_root).resolve())
    print(f"Table 4 CSV: {paths['table4']}")
    print(f"Table 4b CSV: {paths['table4b']}")
    print(f"Figure 3 PNG: {paths['figure3_png']}")
    print(f"Figure 3 PDF: {paths['figure3_pdf']}")
    print(f"Figure 3 SVG: {paths['figure3_svg']}")
    print(f"Figure 4 PNG: {paths['figure4_png']}")
    print(f"Figure 4 PDF: {paths['figure4_pdf']}")
    print(f"Figure 4 SVG: {paths['figure4_svg']}")


if __name__ == "__main__":
    main()
