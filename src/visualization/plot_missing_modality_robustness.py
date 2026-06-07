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


def plot_missing_modality_robustness(source_csv: Path, output_base: Path) -> dict[str, Path]:
    plt.rcParams.update({"font.family": "Arial", "font.size": 9, "font.weight": "bold", "axes.titleweight": "bold", "axes.labelweight": "bold"})
    df = pd.read_csv(source_csv)
    colors = df["Safety_Region"].map({"Safe": "#228833", "Marginal": "#CCBB44", "Escalate": "#CC3311"}).fillna("#4477AA")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), gridspec_kw={"width_ratios": [1.25, 1.0]})

    ax_f1 = axes[0]
    bars = ax_f1.barh(df["Condition"], df["Macro_F1"], color=colors, edgecolor="#222222", linewidth=0.7)
    ax_f1.axvspan(0.85, 1.0, color="#228833", alpha=0.08)
    ax_f1.axvspan(0.70, 0.85, color="#CCBB44", alpha=0.10)
    ax_f1.axvspan(0.0, 0.70, color="#CC3311", alpha=0.08)
    ax_f1.axvline(0.85, color="#222222", linestyle="--", linewidth=0.9)
    ax_f1.axvline(0.70, color="#222222", linestyle="--", linewidth=0.9)
    for bar, value in zip(bars, df["Macro_F1"]):
        ax_f1.text(float(value) + 0.008, bar.get_y() + bar.get_height() / 2, f"{float(value):.3f}", ha="left", va="center", fontsize=8, fontweight="bold")
    ax_f1.set_xlim(0.65, 1.0)
    ax_f1.invert_yaxis()
    ax_f1.set_xlabel("Macro-F1")
    ax_f1.set_title("(a) Macro-F1 under degraded sensing")
    ax_f1.grid(axis="x", linestyle=":", linewidth=0.7, alpha=0.45)

    ax_hitl = axes[1]
    escalation = pd.to_numeric(df["Escalation_Percent"], errors="coerce")
    bars2 = ax_hitl.barh(df["Condition"], escalation, color=colors, edgecolor="#222222", linewidth=0.7)
    for bar, value in zip(bars2, escalation):
        ax_hitl.text(float(value) + 0.6, bar.get_y() + bar.get_height() / 2, f"{float(value):.1f}%", ha="left", va="center", fontsize=8, fontweight="bold")
    ax_hitl.set_xlim(0, max(40, float(escalation.max()) + 5))
    ax_hitl.invert_yaxis()
    ax_hitl.set_yticklabels([])
    ax_hitl.set_xlabel("HITL escalation rate (%)")
    ax_hitl.set_title("(b) HITL escalation response")
    ax_hitl.grid(axis="x", linestyle=":", linewidth=0.7, alpha=0.45)

    for ax in axes:
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
    fig.suptitle("Missing-Modality Robustness and HITL Escalation", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return _save(fig, output_base)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Paper 1 Figure 8 missing-modality robustness.")
    parser.add_argument("--source-csv", default="outputs/csv/missing_modality_results.csv")
    parser.add_argument("--output-base", default="outputs/figures/Figure_8_Missing_Modality_Robustness")
    args = parser.parse_args()
    for label, path in plot_missing_modality_robustness(Path(args.source_csv).resolve(), Path(args.output_base).resolve()).items():
        print(f"Figure 8 {label.upper()}: {path}")


if __name__ == "__main__":
    main()
