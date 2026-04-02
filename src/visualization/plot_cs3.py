from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.visualization.plot_style import apply_publication_style, save_figure_bundle


def _plot_confusion(csv_path: Path, output_base: Path, title: str) -> None:
    df = pd.read_csv(csv_path).set_index("true_label")
    apply_publication_style(plt.matplotlib)
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(df, annot=True, fmt=".0f", cmap="Blues", ax=ax, cbar=False)
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    save_figure_bundle(fig, output_base)
    plt.close(fig)


def plot_model_comparison(summary_csv: Path, output_base: Path) -> None:
    df = pd.read_csv(summary_csv)
    plot_df = df[["model_id", "accuracy", "macro_f1", "weighted_f1"]].melt(id_vars="model_id", var_name="metric", value_name="value")
    apply_publication_style(plt.matplotlib)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=plot_df, x="model_id", y="value", hue="metric", ax=ax, palette="Set2")
    ax.set_title("CS3 Model Performance Comparison")
    ax.set_xlabel("Model family")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    save_figure_bundle(fig, output_base)
    plt.close(fig)


def plot_ablation(ablation_csv: Path, output_base: Path) -> None:
    df = pd.read_csv(ablation_csv)
    plot_df = df.loc[df["condition"] == "nominal"].copy()
    apply_publication_style(plt.matplotlib)
    fig, ax = plt.subplots(figsize=(10, 5.2))
    sns.barplot(data=plot_df, x="ablation_name", y="macro_f1", hue="model_id", ax=ax, palette="Set2")
    ax.set_title("CS3 Ablation Comparison")
    ax.set_xlabel("Ablation setting")
    ax.set_ylabel("Macro F1")
    ax.tick_params(axis="x", rotation=15)
    save_figure_bundle(fig, output_base)
    plt.close(fig)


def plot_training_curves(curves_csv: Path, output_base: Path) -> None:
    df = pd.read_csv(curves_csv)
    apply_publication_style(plt.matplotlib)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    sns.lineplot(data=df, x="epoch", y="val_macro_f1", hue="model_id", linewidth=2.2, ax=ax)
    ax.set_title("CS3 Training Curves")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation macro F1")
    ax.set_ylim(0, 1.05)
    save_figure_bundle(fig, output_base)
    plt.close(fig)


def generate_cs3_figures(project_root: Path) -> None:
    plot_model_comparison(project_root / "outputs" / "csv" / "cs3" / "model_performance_summary.csv", project_root / "outputs" / "figures" / "cs3" / "model_comparison_barplot")
    _plot_confusion(project_root / "outputs" / "csv" / "cs3" / "confusion_matrix_baseline.csv", project_root / "outputs" / "figures" / "cs3" / "confusion_matrix_baseline", "CS3 Baseline Confusion Matrix")
    _plot_confusion(project_root / "outputs" / "csv" / "cs3" / "confusion_matrix_deep.csv", project_root / "outputs" / "figures" / "cs3" / "confusion_matrix_deep", "CS3 Deep Fusion Confusion Matrix")
    _plot_confusion(project_root / "outputs" / "csv" / "cs3" / "confusion_matrix_transformer.csv", project_root / "outputs" / "figures" / "cs3" / "confusion_matrix_transformer", "CS3 Transformer Fusion Confusion Matrix")
    plot_ablation(project_root / "outputs" / "csv" / "cs3" / "ablation_results.csv", project_root / "outputs" / "figures" / "cs3" / "ablation_comparison")
    plot_training_curves(project_root / "outputs" / "csv" / "cs3" / "training_curves.csv", project_root / "outputs" / "figures" / "cs3" / "training_curves")
