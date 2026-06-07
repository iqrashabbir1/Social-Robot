from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


STATUS_MAP = {"Implemented": 2, "Partial": 1, "Future required": 0}


def _save(fig: plt.Figure, base: Path) -> dict[str, Path]:
    base.parent.mkdir(parents=True, exist_ok=True)
    paths = {"png": base.with_suffix(".png"), "pdf": base.with_suffix(".pdf"), "svg": base.with_suffix(".svg")}
    fig.savefig(paths["png"], dpi=600, bbox_inches="tight")
    fig.savefig(paths["pdf"], bbox_inches="tight")
    fig.savefig(paths["svg"], bbox_inches="tight")
    plt.close(fig)
    return paths


def plot_evidence_maturity(source_csv: Path, output_base: Path) -> dict[str, Path]:
    plt.rcParams.update({"font.family": "Arial", "font.size": 9, "font.weight": "bold", "axes.titleweight": "bold", "axes.labelweight": "bold"})
    df = pd.read_csv(source_csv)
    df = df.rename(
        columns={
            "Experimental validation": "Experimental_Validation",
            "Translational readiness": "Translational_Readiness",
        }
    )
    cols = ["Implementation", "Experimental_Validation", "Translational_Readiness"]
    matrix = df[cols].replace(STATUS_MAP).astype(float).to_numpy()
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    im = ax.imshow(matrix, cmap=plt.cm.get_cmap("RdYlGn"), vmin=0, vmax=2, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(["Implementation", "Experimental validation", "Translational readiness"], rotation=15, ha="right")
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["Module"])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            text = df.iloc[i][cols[j]]
            ax.text(j, i, text, ha="center", va="center", fontsize=7, fontweight="bold")
    ax.set_title("Figure 10. Evidence Maturity Dashboard", pad=10)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    fig.tight_layout()
    return _save(fig, output_base)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Paper 1 Figure 10 evidence maturity dashboard.")
    parser.add_argument("--source-csv", default="outputs/csv/evidence_maturity_matrix.csv")
    parser.add_argument("--output-base", default="outputs/figures/Figure_10_Evidence_Maturity_Dashboard")
    args = parser.parse_args()
    for label, path in plot_evidence_maturity(Path(args.source_csv).resolve(), Path(args.output_base).resolve()).items():
        print(f"Figure 10 {label.upper()}: {path}")


if __name__ == "__main__":
    main()
