from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from src.visualization.plot_style import COLOR_PALETTE, apply_publication_style, save_figure_bundle


def plot_system_architecture(interface_csv: Path, output_base: Path) -> None:
    df = pd.read_csv(interface_csv)
    apply_publication_style(plt.matplotlib)
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    ax.axis("off")
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)

    x_positions = [0.7, 2.5, 4.3, 6.1, 7.9, 9.7, 11.5]
    y = 4.2
    colors = [
        COLOR_PALETTE["navy"],
        COLOR_PALETTE["teal"],
        COLOR_PALETTE["gold"],
        COLOR_PALETTE["orange"],
        COLOR_PALETTE["red"],
        COLOR_PALETTE["slate"],
        COLOR_PALETTE["mint"],
    ]
    for idx, row in enumerate(df.to_dict(orient="records")):
        box = FancyBboxPatch((x_positions[idx], y), 1.4, 1.0, boxstyle="round,pad=0.04,rounding_size=0.08", facecolor=colors[idx], edgecolor="white")
        ax.add_patch(box)
        ax.text(x_positions[idx] + 0.7, y + 0.63, row["topic"], ha="center", va="center", color="white", fontsize=9)
        ax.text(x_positions[idx] + 0.7, y + 0.28, row["producer"], ha="center", va="center", color="white", fontsize=8)
        if idx < len(df) - 1:
            ax.add_patch(FancyArrowPatch((x_positions[idx] + 1.42, y + 0.5), (x_positions[idx + 1] - 0.1, y + 0.5), arrowstyle="->", mutation_scale=15, linewidth=1.8, color="#444444"))
    ax.set_title("Paper 1 System Architecture and ROS2 Interface Chain")
    save_figure_bundle(fig, output_base)
    plt.close(fig)


def plot_latency_distribution(latency_csv: Path, output_base: Path) -> None:
    df = pd.read_csv(latency_csv)
    apply_publication_style(plt.matplotlib)
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(8.5, 5))
    sns.boxplot(data=df, x="mode", y="latency_ms", hue="mode", ax=ax, palette="Set2", legend=False)
    ax.set_title("CS1 End-to-End Latency Distribution")
    ax.set_xlabel("Experiment mode")
    ax.set_ylabel("Latency (ms)")
    save_figure_bundle(fig, output_base)
    plt.close(fig)


def plot_sync_error(sync_csv: Path, output_base: Path) -> None:
    df = pd.read_csv(sync_csv)
    summary = df.groupby(["mode", "step"], as_index=False)["sync_error_ms"].mean()
    apply_publication_style(plt.matplotlib)
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.lineplot(data=summary, x="step", y="sync_error_ms", hue="mode", linewidth=2, ax=ax)
    ax.set_title("CS1 Synchronization Error Over Time")
    ax.set_xlabel("Simulation step")
    ax.set_ylabel("Synchronization error (ms)")
    save_figure_bundle(fig, output_base)
    plt.close(fig)


def plot_task_success(latency_csv: Path, output_base: Path) -> None:
    df = pd.read_csv(latency_csv)
    summary = df.groupby("mode", as_index=False)["success_flag"].mean().rename(columns={"success_flag": "task_success_rate"})
    apply_publication_style(plt.matplotlib)
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.bar(summary["mode"], summary["task_success_rate"], color=COLOR_PALETTE["teal"])
    ax.set_ylim(0, 1.05)
    ax.set_title("CS1 Task Success Comparison")
    ax.set_xlabel("Experiment mode")
    ax.set_ylabel("Task success rate")
    save_figure_bundle(fig, output_base)
    plt.close(fig)


def plot_resource_usage(latency_csv: Path, output_base: Path) -> None:
    df = pd.read_csv(latency_csv)
    summary = df.groupby("mode", as_index=False)[["cpu_percent", "memory_mb"]].mean()
    apply_publication_style(plt.matplotlib)
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()
    ax1.bar(summary["mode"], summary["cpu_percent"], color=COLOR_PALETTE["orange"], width=0.45, label="CPU (%)")
    ax2.plot(summary["mode"], summary["memory_mb"], color=COLOR_PALETTE["navy"], marker="o", linewidth=2, label="Memory (MB)")
    ax1.set_title("CS1 Resource Usage by Experiment Mode")
    ax1.set_ylabel("CPU usage (%)")
    ax2.set_ylabel("Memory (MB)")
    ax1.set_xlabel("Experiment mode")
    save_figure_bundle(fig, output_base)
    plt.close(fig)


def plot_simulator_vs_playback(comparison_csv: Path, output_base: Path) -> None:
    df = pd.read_csv(comparison_csv)
    apply_publication_style(plt.matplotlib)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.bar(df["source"], df["mean_latency_ms"], color=[COLOR_PALETTE["navy"], COLOR_PALETTE["orange"]])
    ax.set_title("CS1 Simulator vs Playback-Grounded Comparison")
    ax.set_xlabel("Runtime source")
    ax.set_ylabel("Mean latency (ms)")
    save_figure_bundle(fig, output_base)
    plt.close(fig)


def generate_cs1_figures(project_root: Path) -> None:
    plot_system_architecture(project_root / "outputs" / "csv" / "cs1" / "interface_spec.csv", project_root / "outputs" / "figures" / "cs1" / "system_architecture_diagram")
    plot_latency_distribution(project_root / "outputs" / "csv" / "cs1" / "latency_metrics.csv", project_root / "outputs" / "figures" / "cs1" / "latency_distribution")
    plot_sync_error(project_root / "outputs" / "csv" / "cs1" / "sync_error_timeseries.csv", project_root / "outputs" / "figures" / "cs1" / "synchronization_error_over_time")
    plot_task_success(project_root / "outputs" / "csv" / "cs1" / "latency_metrics.csv", project_root / "outputs" / "figures" / "cs1" / "task_success_comparison")
    plot_resource_usage(project_root / "outputs" / "csv" / "cs1" / "latency_metrics.csv", project_root / "outputs" / "figures" / "cs1" / "resource_usage")
