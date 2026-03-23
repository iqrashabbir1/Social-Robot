from __future__ import annotations

import argparse
import csv
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
except Exception as exc:
    raise RuntimeError(
        "matplotlib and numpy are required to generate figures. "
        "Install or expose a standard Python runtime before executing this script."
    ) from exc

from src.visualization.plot_style import COLOR_PALETTE, apply_publication_style


CAPABILITY_COLUMNS = [
    "predictive_health_risk_capability",
    "emotion_awareness",
    "medication_adherence_support",
    "ros2_digital_twin_support",
    "iot_edge_deployment",
    "explainability",
    "hitl_safety_governance",
    "privacy_awareness",
    "telepresence_cultural_adaptation",
    "real_world_readiness",
]


def _read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _save(fig, base_path: Path) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".svg", ".pdf"):
        fig.savefig(base_path.with_suffix(suffix), bbox_inches="tight")


def plot_literature_heatmap(matrix_path: Path, output_base: Path) -> None:
    rows = _read_csv_dicts(matrix_path)
    approaches = [row["approach_id"] for row in rows]
    data = np.array([[int(row[column]) for column in CAPABILITY_COLUMNS] for row in rows])

    fig, ax = plt.subplots(figsize=(13, 5.5))
    im = ax.imshow(data, cmap="YlGnBu", vmin=0, vmax=3, aspect="auto")
    ax.set_xticks(range(len(CAPABILITY_COLUMNS)))
    ax.set_xticklabels(
        [
            "Risk",
            "Emotion",
            "Adherence",
            "ROS2/Twin",
            "IoT/Edge",
            "Explain",
            "HITL",
            "Privacy",
            "Telepresence",
            "Readiness",
        ],
        rotation=30,
        ha="right",
    )
    ax.set_yticks(range(len(approaches)))
    ax.set_yticklabels(approaches)
    ax.set_title("Literature-Aligned Capability Coverage (0-3 Qualitative Rubric)")
    for row_idx in range(data.shape[0]):
        for col_idx in range(data.shape[1]):
            ax.text(col_idx, row_idx, str(data[row_idx, col_idx]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, shrink=0.9, label="Capability score")
    _save(fig, output_base)
    plt.close(fig)


def plot_literature_bar(matrix_path: Path, output_base: Path) -> None:
    rows = _read_csv_dicts(matrix_path)
    labels = [row["approach_id"] for row in rows]
    totals = [sum(int(row[column]) for column in CAPABILITY_COLUMNS) for row in rows]

    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    colors = [COLOR_PALETTE["teal"] if label == "A8" else COLOR_PALETTE["slate"] for label in labels]
    ax.bar(labels, totals, color=colors)
    ax.set_title("Integrated Capability Comparison Across Approaches")
    ax.set_ylabel("Total qualitative capability score")
    ax.set_xlabel("Approach")
    ax.set_ylim(0, max(totals) + 3)
    _save(fig, output_base)
    plt.close(fig)


def plot_system_architecture(module_map_path: Path, output_base: Path) -> None:
    rows = _read_csv_dicts(module_map_path)

    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis("off")

    layer_positions = {
        "Layer 1": (1.0, 7.5, COLOR_PALETTE["navy"]),
        "Layer 2": (4.0, 7.5, COLOR_PALETTE["teal"]),
        "Layer 3": (7.0, 7.5, COLOR_PALETTE["gold"]),
        "Layer 4": (10.0, 7.5, COLOR_PALETTE["orange"]),
        "Layer 5": (10.0, 4.0, COLOR_PALETTE["red"]),
    }

    module_offsets: dict[str, int] = {}
    for row in rows:
        layer = row["layer"]
        base_x, base_y, color = layer_positions[layer]
        offset = module_offsets.get(layer, 0)
        x = base_x
        y = base_y - (offset * 1.4)
        module_offsets[layer] = offset + 1

        box = FancyBboxPatch(
            (x, y),
            2.2,
            0.9,
            boxstyle="round,pad=0.03,rounding_size=0.08",
            facecolor=color,
            edgecolor="white",
            alpha=0.92,
        )
        ax.add_patch(box)
        ax.text(x + 1.1, y + 0.45, row["module_id"], ha="center", va="center", color="white", fontsize=12, fontweight="bold")
        ax.text(x + 2.45, y + 0.45, row["module_name"], ha="left", va="center", color="#222222", fontsize=10)

    arrows = [
        ((3.4, 7.8), (4.0, 7.8)),
        ((6.4, 7.8), (7.0, 7.8)),
        ((9.4, 7.8), (10.0, 7.8)),
        ((11.1, 5.7), (11.1, 5.0)),
    ]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=18, linewidth=2, color="#444444"))

    ax.set_title("System Architecture Overview")
    _save(fig, output_base)
    plt.close(fig)


def plot_case_study_dashboard(case_registry_path: Path, output_base: Path) -> None:
    rows = _read_csv_dicts(case_registry_path)
    status_order = [
        "implemented_real_baseline plus planned extensions",
        "simulation_based_evaluation",
        "planned_experiment",
    ]
    counts = [sum(row["evidence_status"] == status for row in rows) for status in status_order]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    colors = [COLOR_PALETTE["teal"], COLOR_PALETTE["gold"], COLOR_PALETTE["red"]]
    ax.bar(
        ["Implemented+Planned", "Simulation", "Planned"],
        counts,
        color=colors,
    )
    ax.set_title("Case Study Evidence Distribution")
    ax.set_ylabel("Number of case studies")
    _save(fig, output_base)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate publication-style figures from CSV artifacts.")
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()

    apply_publication_style(plt.matplotlib)
    project_root = Path(args.project_root).resolve()

    plot_literature_heatmap(
        project_root / "outputs" / "tables" / "literature_comparison_matrix.csv",
        project_root / "outputs" / "figures" / "literature_gap_heatmap",
    )
    plot_literature_bar(
        project_root / "outputs" / "tables" / "literature_comparison_matrix.csv",
        project_root / "outputs" / "figures" / "literature_radar_or_bar_comparison",
    )
    plot_system_architecture(
        project_root / "outputs" / "csv" / "system_module_map.csv",
        project_root / "outputs" / "figures" / "system_architecture_overview",
    )
    plot_case_study_dashboard(
        project_root / "outputs" / "csv" / "case_study_registry.csv",
        project_root / "outputs" / "figures" / "case_study_summary_dashboard",
    )


if __name__ == "__main__":
    main()
