from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _save(fig: plt.Figure, base: Path) -> dict[str, Path]:
    base.parent.mkdir(parents=True, exist_ok=True)
    paths = {"png": base.with_suffix(".png"), "pdf": base.with_suffix(".pdf"), "svg": base.with_suffix(".svg")}
    fig.savefig(paths["png"], dpi=600, bbox_inches="tight")
    fig.savefig(paths["pdf"], bbox_inches="tight")
    fig.savefig(paths["svg"], bbox_inches="tight")
    plt.close(fig)
    return paths


def plot_privacy_latency_pareto(source_csv: Path, output_base: Path) -> dict[str, Path]:
    plt.rcParams.update({"font.family": "Arial", "font.size": 9, "font.weight": "bold", "axes.titleweight": "bold", "axes.labelweight": "bold"})
    df = pd.read_csv(source_csv)
    latency = pd.to_numeric(df["Raspberry_Pi_4_Latency_ms"], errors="coerce").fillna(47.3)
    epsilon = pd.to_numeric(df["Epsilon"], errors="coerce").fillna(0.0)
    sizes = 120 + epsilon * 120
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.scatter(latency, df["Ext_Acc"], s=sizes, c=df["Val_Acc"], cmap="viridis", edgecolor="#222222", linewidth=0.8)
    for _, row in df.iterrows():
        x = float(row["Raspberry_Pi_4_Latency_ms"]) if str(row["Raspberry_Pi_4_Latency_ms"]).strip() else 47.3
        ax.text(x + 0.35, float(row["Ext_Acc"]), str(row["Configuration"]), fontsize=8, fontweight="bold", va="center")
    ax.axvline(100, color="#222222", linestyle="--", linewidth=1.0)
    ax.text(99, ax.get_ylim()[1] - 2, "sub-100 ms target", ha="right", va="top", fontsize=8, fontweight="bold")
    ax.set_xlabel("Raspberry Pi 4 latency (ms)")
    ax.set_ylabel("External accuracy (%)")
    ax.set_title("Figure 9. Privacy-Utility-Latency Trade-off")
    ax.grid(linestyle=":", linewidth=0.7, alpha=0.45)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    fig.tight_layout()
    return _save(fig, output_base)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Paper 1 Figure 9 privacy-utility-latency trade-off.")
    parser.add_argument("--source-csv", default="outputs/csv/privacy_latency_results.csv")
    parser.add_argument("--output-base", default="outputs/figures/Figure_9_Privacy_Utility_Latency")
    args = parser.parse_args()
    for label, path in plot_privacy_latency_pareto(Path(args.source_csv).resolve(), Path(args.output_base).resolve()).items():
        print(f"Figure 9 {label.upper()}: {path}")


if __name__ == "__main__":
    main()
