from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

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


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _save(fig, base_path: Path) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".svg", ".pdf"):
        fig.savefig(base_path.with_suffix(suffix), bbox_inches="tight")


def plot_literature_heatmap(matrix_path: Path, output_base: Path) -> None:
    df = _read_csv(matrix_path)
    heat_df = df[["approach_id", *CAPABILITY_COLUMNS]].set_index("approach_id")
    fig, ax = plt.subplots(figsize=(13, 5.5))
    sns.heatmap(heat_df, annot=True, fmt=".0f", cmap="YlGnBu", vmin=0, vmax=3, cbar_kws={"label": "Capability score"}, ax=ax)
    ax.set_title("Literature-Aligned Capability Coverage (0-3 Qualitative Rubric)")
    ax.set_xlabel("Capability")
    ax.set_ylabel("Approach")
    _save(fig, output_base)
    plt.close(fig)


def plot_literature_bar(matrix_path: Path, output_base: Path) -> None:
    df = _read_csv(matrix_path)
    df["total_score"] = df[CAPABILITY_COLUMNS].sum(axis=1)
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    colors = [COLOR_PALETTE["teal"] if label == "A8" else COLOR_PALETTE["slate"] for label in df["approach_id"]]
    ax.bar(df["approach_id"], df["total_score"], color=colors)
    ax.set_title("Integrated Capability Comparison Across Approaches")
    ax.set_ylabel("Total qualitative capability score")
    ax.set_xlabel("Approach")
    _save(fig, output_base)
    plt.close(fig)


def plot_system_architecture(module_map_path: Path, output_base: Path) -> None:
    rows = _read_csv(module_map_path).to_dict(orient="records")
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
        box = FancyBboxPatch((x, y), 2.2, 0.9, boxstyle="round,pad=0.03,rounding_size=0.08", facecolor=color, edgecolor="white", alpha=0.92)
        ax.add_patch(box)
        ax.text(x + 1.1, y + 0.45, row["module_id"], ha="center", va="center", color="white", fontsize=12, fontweight="bold")
        ax.text(x + 2.45, y + 0.45, row["module_name"], ha="left", va="center", color="#222222", fontsize=10)
    arrows = [((3.4, 7.8), (4.0, 7.8)), ((6.4, 7.8), (7.0, 7.8)), ((9.4, 7.8), (10.0, 7.8)), ((11.1, 5.7), (11.1, 5.0))]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=18, linewidth=2, color="#444444"))
    ax.set_title("System Architecture Overview")
    _save(fig, output_base)
    plt.close(fig)


def plot_case_study_dashboard(case_registry_path: Path, output_base: Path) -> None:
    df = _read_csv(case_registry_path)
    order = [
        "implemented_real_baseline plus planned extensions",
        "simulation_based_evaluation",
        "planned_experiment",
    ]
    counts = [int((df["evidence_status"] == item).sum()) for item in order]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(["Implemented+Planned", "Simulation", "Planned"], counts, color=[COLOR_PALETTE["teal"], COLOR_PALETTE["gold"], COLOR_PALETTE["red"]])
    ax.set_title("Case Study Evidence Distribution")
    ax.set_ylabel("Number of case studies")
    _save(fig, output_base)
    plt.close(fig)


def plot_confusion_matrix(csv_path: Path, output_base: Path, title: str) -> None:
    df = _read_csv(csv_path)
    matrix = df.set_index("true_label")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt=".0f", cmap="Blues", cbar=False, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    _save(fig, output_base)
    plt.close(fig)


def plot_calibration_curve(csv_path: Path, output_base: Path, title: str, x_label: str, y_label: str) -> None:
    df = _read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(df.iloc[:, 0], df.iloc[:, 1], marker="o", color=COLOR_PALETTE["teal"], linewidth=2, label="Observed")
    ax.plot([0, 1], [0, 1], linestyle="--", color=COLOR_PALETTE["slate"], label="Perfect calibration")
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(frameon=False)
    _save(fig, output_base)
    plt.close(fig)


def plot_multiclass_curves(roc_path: Path, pr_path: Path, output_base: Path) -> None:
    roc_df = _read_csv(roc_path)
    pr_df = _read_csv(pr_path)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for class_label, subset in roc_df.groupby("class_label"):
        axes[0].plot(subset["fpr"], subset["tpr"], label=f"{class_label} (AUC={subset['auc'].iloc[0]:.2f})")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color=COLOR_PALETTE["slate"])
    axes[0].set_title("Speech Baseline ROC Curves")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")

    for class_label, subset in pr_df.groupby("class_label"):
        axes[1].plot(subset["recall"], subset["precision"], label=f"{class_label} (AUC={subset['auc'].iloc[0]:.2f})")
    axes[1].set_title("Speech Baseline PR Curves")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")

    for ax in axes:
        ax.legend(frameon=False, fontsize=9)
    _save(fig, output_base)
    plt.close(fig)


def plot_training_curve(csv_path: Path, output_base: Path) -> None:
    df = _read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.plot(df["epoch"], df["train_score"], marker="o", label="Train", color=COLOR_PALETTE["navy"])
    ax.plot(df["epoch"], df["validation_score"], marker="o", label="Validation", color=COLOR_PALETTE["teal"])
    ax.set_title("MER Training Curve Simulation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.legend(frameon=False)
    _save(fig, output_base)
    plt.close(fig)


def plot_health_risk_curves(roc_path: Path, pr_path: Path, output_base: Path) -> None:
    roc_df = _read_csv(roc_path)
    pr_df = _read_csv(pr_path)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(roc_df["fpr"], roc_df["tpr"], color=COLOR_PALETTE["red"], linewidth=2)
    axes[0].plot([0, 1], [0, 1], linestyle="--", color=COLOR_PALETTE["slate"])
    axes[0].set_title(f"Health Risk ROC (AUC={roc_df['auc'].iloc[0]:.2f})")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")

    axes[1].plot(pr_df["recall"], pr_df["precision"], color=COLOR_PALETTE["orange"], linewidth=2)
    axes[1].set_title(f"Health Risk PR (AUC={pr_df['auc'].iloc[0]:.2f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    _save(fig, output_base)
    plt.close(fig)


def plot_anomaly_timeline(csv_path: Path, output_base: Path) -> None:
    df = _read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.plot(pd.to_datetime(df["timestamp"]), df["anomaly_score"], marker="o", color=COLOR_PALETTE["orange"])
    ax.set_title("Anomaly Timeline")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Anomaly score")
    fig.autofmt_xdate()
    _save(fig, output_base)
    plt.close(fig)


def plot_reason_distribution(csv_path: Path, output_base: Path) -> None:
    df = _read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(df["reason"], df["count"], color=COLOR_PALETTE["teal"])
    ax.set_title("Medication Adherence Reason Distribution")
    ax.set_xlabel("Reason")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=25)
    _save(fig, output_base)
    plt.close(fig)


def plot_explainability_scores(csv_path: Path, output_base: Path) -> None:
    df = _read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.22
    x = range(len(df))
    ax.bar([i - width for i in x], df["faithfulness_score"], width=width, label="Faithfulness", color=COLOR_PALETTE["navy"])
    ax.bar(x, df["citation_coverage"], width=width, label="Citation coverage", color=COLOR_PALETTE["teal"])
    ax.bar([i + width for i in x], df["clinician_usefulness"], width=width, label="Clinician usefulness", color=COLOR_PALETTE["gold"])
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["method"], rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title("Explainability Score Comparison")
    ax.legend(frameon=False)
    _save(fig, output_base)
    plt.close(fig)


def plot_latency_resource_tradeoff(csv_path: Path, output_base: Path) -> None:
    df = _read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(df["latency_ms"], df["clinical_utility_score"], s=df["memory_mb"] / 2, c=df["cpu_percent"], cmap="viridis", alpha=0.8)
    for _, row in df.iterrows():
        ax.text(row["latency_ms"] + 2, row["clinical_utility_score"] + 0.005, row["mode"], fontsize=9)
    ax.set_title("Latency and Resource Tradeoff")
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Clinical utility score")
    fig.colorbar(scatter, ax=ax, label="CPU usage (%)")
    _save(fig, output_base)
    plt.close(fig)


def plot_privacy_utility_tradeoff(csv_path: Path, output_base: Path) -> None:
    df = _read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["privacy_score"], df["utility_score"], marker="o", color=COLOR_PALETTE["red"])
    for _, row in df.iterrows():
        ax.text(row["privacy_score"] + 0.005, row["utility_score"] + 0.01, row["profile"], fontsize=9)
    ax.set_title("Privacy-Utility Tradeoff")
    ax.set_xlabel("Privacy score")
    ax.set_ylabel("Utility score")
    _save(fig, output_base)
    plt.close(fig)


def plot_module_contribution(csv_path: Path, output_base: Path) -> None:
    df = _read_csv(csv_path).sort_values("contribution_score", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(df["module"], df["contribution_score"], color=COLOR_PALETTE["navy"])
    ax.set_title("Module-wise Contribution")
    ax.set_xlabel("Contribution score")
    _save(fig, output_base)
    plt.close(fig)


def plot_end_to_end_workflow(csv_path: Path, output_base: Path) -> None:
    df = _read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.axis("off")
    x_start = 0.3
    for _, row in df.iterrows():
        box = FancyBboxPatch((x_start, 0.45), 1.15, 0.35, boxstyle="round,pad=0.03,rounding_size=0.06", facecolor=COLOR_PALETTE["mint"], edgecolor="white")
        ax.add_patch(box)
        ax.text(x_start + 0.575, 0.625, f"{int(row['step_order'])}. {row['step_name']}", ha="center", va="center", fontsize=9)
        ax.text(x_start + 0.575, 0.5, f"{row['median_latency_ms']} ms", ha="center", va="center", fontsize=8, color="#334e68")
        if row["step_order"] < df["step_order"].max():
            ax.add_patch(FancyArrowPatch((x_start + 1.17, 0.625), (x_start + 1.45, 0.625), arrowstyle="->", mutation_scale=16, linewidth=1.8, color="#555"))
        x_start += 1.45
    ax.set_xlim(0, x_start + 0.2)
    ax.set_ylim(0, 1.2)
    ax.set_title("End-to-End Workflow")
    _save(fig, output_base)
    plt.close(fig)


def plot_pilot_readiness(csv_path: Path, output_base: Path) -> None:
    df = _read_csv(csv_path)
    counts = df["status"].value_counts()
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.bar(counts.index, counts.values, color=[COLOR_PALETTE["teal"], COLOR_PALETTE["orange"]])
    ax.set_title("Pilot Validation Readiness")
    ax.set_xlabel("Status")
    ax.set_ylabel("Count")
    _save(fig, output_base)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate publication-style figures from CSV artifacts.")
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()

    apply_publication_style(plt.matplotlib)
    sns.set_style("whitegrid")
    project_root = Path(args.project_root).resolve()

    plot_literature_heatmap(project_root / "outputs" / "tables" / "literature_comparison_matrix.csv", project_root / "outputs" / "figures" / "literature_gap_heatmap")
    plot_literature_bar(project_root / "outputs" / "tables" / "literature_comparison_matrix.csv", project_root / "outputs" / "figures" / "literature_radar_or_bar_comparison")
    plot_system_architecture(project_root / "outputs" / "csv" / "system_module_map.csv", project_root / "outputs" / "figures" / "system_architecture_overview")
    plot_case_study_dashboard(project_root / "outputs" / "csv" / "case_study_registry.csv", project_root / "outputs" / "figures" / "case_study_summary_dashboard")

    plot_confusion_matrix(project_root / "outputs" / "csv" / "vision_confusion_matrix.csv", project_root / "outputs" / "figures" / "vision_confusion_matrix", "Vision Baseline Confusion Matrix")
    plot_confusion_matrix(project_root / "outputs" / "csv" / "speech_confusion_matrix.csv", project_root / "outputs" / "figures" / "speech_confusion_matrix", "Speech Baseline Confusion Matrix")
    plot_calibration_curve(project_root / "outputs" / "csv" / "speech_calibration_curve.csv", project_root / "outputs" / "figures" / "speech_calibration_plot", "Speech Baseline Calibration", "Predicted confidence", "Observed accuracy")
    plot_calibration_curve(project_root / "outputs" / "csv" / "risk_calibration_curve.csv", project_root / "outputs" / "figures" / "risk_calibration_plot", "Health Risk Calibration", "Predicted risk", "Observed event rate")
    plot_multiclass_curves(project_root / "outputs" / "csv" / "speech_roc_curve.csv", project_root / "outputs" / "csv" / "speech_pr_curve.csv", project_root / "outputs" / "figures" / "speech_roc_pr_curves")
    plot_training_curve(project_root / "outputs" / "csv" / "training_curve_simulation.csv", project_root / "outputs" / "figures" / "training_curve_simulation")
    plot_health_risk_curves(project_root / "outputs" / "csv" / "health_risk_roc_curve.csv", project_root / "outputs" / "csv" / "health_risk_pr_curve.csv", project_root / "outputs" / "figures" / "health_risk_curves")
    plot_anomaly_timeline(project_root / "outputs" / "csv" / "anomaly_timeline.csv", project_root / "outputs" / "figures" / "anomaly_timeline")
    plot_reason_distribution(project_root / "outputs" / "csv" / "adherence_reason_distribution.csv", project_root / "outputs" / "figures" / "adherence_reason_distribution")
    plot_explainability_scores(project_root / "outputs" / "csv" / "explainability_score_comparison.csv", project_root / "outputs" / "figures" / "explainability_score_comparison")
    plot_latency_resource_tradeoff(project_root / "outputs" / "csv" / "latency_resource_tradeoff.csv", project_root / "outputs" / "figures" / "latency_resource_tradeoff")
    plot_privacy_utility_tradeoff(project_root / "outputs" / "csv" / "privacy_utility_tradeoff.csv", project_root / "outputs" / "figures" / "privacy_utility_tradeoff")
    plot_module_contribution(project_root / "outputs" / "csv" / "module_contribution.csv", project_root / "outputs" / "figures" / "module_contribution")
    plot_end_to_end_workflow(project_root / "outputs" / "csv" / "end_to_end_workflow.csv", project_root / "outputs" / "figures" / "end_to_end_workflow")
    plot_pilot_readiness(project_root / "outputs" / "tables" / "pilot_validation_readiness.csv", project_root / "outputs" / "figures" / "pilot_validation_readiness")


if __name__ == "__main__":
    main()
