from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.visualization.plot_style import COLOR_PALETTE, apply_publication_style, save_figure_bundle


def plot_modality_heatmap(availability_csv: Path, output_base: Path) -> None:
    df = pd.read_csv(availability_csv).set_index("window_id")
    heatmap_df = df[
        [
            "video_available_ratio",
            "audio_available_ratio",
            "context_available_ratio",
            "physiology_available_ratio",
        ]
    ]
    apply_publication_style(plt.matplotlib)
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(9, 5.8))
    sns.heatmap(heatmap_df, cmap="YlGnBu", ax=ax, cbar_kws={"label": "Availability ratio"})
    ax.set_title("CS2 Modality Availability Heatmap")
    ax.set_xlabel("Modality")
    ax.set_ylabel("Window ID")
    save_figure_bundle(fig, output_base)
    plt.close(fig)


def plot_sync_quality(sync_csv: Path, output_base: Path) -> None:
    df = pd.read_csv(sync_csv)
    apply_publication_style(plt.matplotlib)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(df["condition"], df["mean_alignment_error_ms"], color=[COLOR_PALETTE["navy"], COLOR_PALETTE["orange"]])
    ax.set_title("CS2 Synchronization Quality Comparison")
    ax.set_xlabel("Condition")
    ax.set_ylabel("Mean alignment error (ms)")
    save_figure_bundle(fig, output_base)
    plt.close(fig)


def plot_missing_modality(sync_csv: Path, output_base: Path) -> None:
    df = pd.read_csv(sync_csv)
    compare = df.melt(
        id_vars=["condition"],
        value_vars=["video_available_rate", "audio_available_rate", "context_available_rate", "physiology_available_rate"],
        var_name="metric",
        value_name="value",
    )
    apply_publication_style(plt.matplotlib)
    fig, ax = plt.subplots(figsize=(9.5, 5))
    sns.barplot(data=compare, x="metric", y="value", hue="condition", ax=ax, palette="Set2")
    ax.set_title("CS2 Missing-Modality Robustness Overview")
    ax.set_xlabel("Availability metric")
    ax.set_ylabel("Rate")
    ax.tick_params(axis="x", rotation=20)
    save_figure_bundle(fig, output_base)
    plt.close(fig)


def generate_cs2_figures(project_root: Path) -> None:
    plot_modality_heatmap(project_root / "outputs" / "csv" / "cs2" / "modality_availability.csv", project_root / "outputs" / "figures" / "cs2" / "modality_availability_heatmap")
    plot_sync_quality(project_root / "outputs" / "csv" / "cs2" / "sync_quality_metrics.csv", project_root / "outputs" / "figures" / "cs2" / "synchronization_quality_comparison")
    plot_missing_modality(project_root / "outputs" / "csv" / "cs2" / "sync_quality_metrics.csv", project_root / "outputs" / "figures" / "cs2" / "missing_modality_robustness")
