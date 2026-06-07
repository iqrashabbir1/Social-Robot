from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common.io_utils import write_dataframe


CONFIGS = ["ABL0", "ABL1", "ABL2", "ABL3", "ABL4", "ABL5", "ABL6"]
REMOVED = [
    "None",
    "KG grounding",
    "Speech stream",
    "Digital twin",
    "Cross-attention fusion",
    "HITL gate",
    "Privacy gate",
]
VAL_ACC = [97.81, 97.78, 90.12, 97.80, 94.41, 97.78, 97.79]
KG_FAITH = [89, 27, 89, 89, 89, 89, 89]
HITL_PREC = [0.94, 0.91, 0.91, 0.87, 0.89, None, 0.94]
MAIN_FINDINGS = [
    "Full-system baseline",
    "Explanation faithfulness collapses",
    "Predictive performance drops",
    "Routing precision degrades",
    "Multimodal contextual fusion weakens",
    "6.3% urgent cases unrouted",
    "Privacy constraints violated",
]


def ablation_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Config": CONFIGS,
            "Removed component": REMOVED,
            "Val. Acc. (%)": [f"{value:.2f}" for value in VAL_ACC],
            "KG Faith.": [f"{value / 100.0:.2f}" for value in KG_FAITH],
            "HITL Prec.": ["Unsafe" if value is None else f"{value:.2f}" for value in HITL_PREC],
            "Main finding": MAIN_FINDINGS,
        }
    )


def _save_all(fig: plt.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")


def _label_bars(ax: plt.Axes, bars, fmt: str = "{:.1f}", color: str = "#222222") -> None:
    for bar in bars:
        height = float(bar.get_height())
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1.2,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            color=color,
        )


def plot_ablation_analysis(output_base: Path) -> dict[str, Path]:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10,
            "font.weight": "bold",
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
        }
    )

    x = np.arange(len(CONFIGS))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))

    ax_a = axes[0]
    acc_bars = ax_a.bar(
        x - width / 2,
        VAL_ACC,
        width,
        label="Validation accuracy (%)",
        color="#4477AA",
        edgecolor="#222222",
        linewidth=0.8,
    )
    kg_bars = ax_a.bar(
        x + width / 2,
        KG_FAITH,
        width,
        label="KG faithfulness (%)",
        color="#66C2A5",
        edgecolor="#222222",
        linewidth=0.8,
    )
    _label_bars(ax_a, acc_bars, "{:.1f}")
    _label_bars(ax_a, kg_bars, "{:.0f}")
    ax_a.set_title("(a) Predictive performance and explanation faithfulness", pad=10)
    ax_a.set_ylabel("Score (%)")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(CONFIGS)
    ax_a.set_ylim(0, 105)
    ax_a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False)
    ax_a.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.45)

    ax_b = axes[1]
    hitl_values = [0.08 if value is None else value for value in HITL_PREC]
    colors = ["#4477AA" if value is not None else "#CC3311" for value in HITL_PREC]
    hatches = [None if value is not None else "///" for value in HITL_PREC]
    hitl_bars = ax_b.bar(
        x,
        hitl_values,
        color=colors,
        edgecolor="#222222",
        linewidth=0.8,
    )
    for bar, hatch, value in zip(hitl_bars, hatches, HITL_PREC):
        if hatch:
            bar.set_hatch(hatch)
        if value is None:
            ax_b.text(
                bar.get_x() + bar.get_width() / 2,
                0.16,
                "UNSAFE",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                color="#CC3311",
                rotation=90,
            )
        else:
            ax_b.text(
                bar.get_x() + bar.get_width() / 2,
                float(value) + 0.025,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
                color="#222222",
            )
    ax_b.axhline(0.94, color="#222222", linestyle="--", linewidth=1.1)
    ax_b.text(
        0.98,
        0.975,
        "Full-system reference = 0.94",
        transform=ax_b.get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=8,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 2},
    )
    ax_b.set_title("(b) HITL routing precision under component removal", pad=10)
    ax_b.set_ylabel("HITL precision")
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(CONFIGS)
    ax_b.set_ylim(0, 1.05)
    ax_b.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.45)

    for ax in axes:
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    fig.text(
        0.5,
        0.035,
        "ABL0 = full system; ABL1 = no KG; ABL2 = no speech; ABL3 = no DT; "
        "ABL4 = no cross-attention; ABL5 = no HITL; ABL6 = no privacy gate.",
        ha="center",
        va="center",
        fontsize=8,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.006,
        "Ablation results distinguish predictive performance, explanation faithfulness, and safety-routing contribution.",
        ha="center",
        va="center",
        fontsize=8,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    _save_all(fig, output_base)
    plt.close(fig)

    return {
        "png": output_base.with_suffix(".png"),
        "pdf": output_base.with_suffix(".pdf"),
        "svg": output_base.with_suffix(".svg"),
    }


def generate_ablation_outputs(project_root: Path) -> dict[str, Path]:
    table = ablation_table()
    table_paths = {
        "summary_table": project_root / "outputs" / "tables" / "ablation_summary.csv",
        "outputs_table": project_root / "outputs" / "tables" / "paper1_table_ablation_summary.csv",
        "paper_table": project_root / "experiments" / "results" / "paper_tables" / "table5_ablation.csv",
    }
    for path in table_paths.values():
        write_dataframe(path, table)
    figure_paths = plot_ablation_analysis(project_root / "outputs" / "figures" / "Figure_5_Ablation_Analysis")
    return {**table_paths, **figure_paths}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PAEMDT Figure 5 ablation analysis and Table 5.")
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    paths = generate_ablation_outputs(Path(args.project_root).resolve())
    print(f"Table 5 outputs CSV: {paths['outputs_table']}")
    print(f"Table 5 paper CSV: {paths['paper_table']}")
    print(f"Figure 5 PNG: {paths['png']}")
    print(f"Figure 5 PDF: {paths['pdf']}")
    print(f"Figure 5 SVG: {paths['svg']}")


if __name__ == "__main__":
    main()
