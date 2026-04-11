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
    label_col = "algorithm_name" if "algorithm_name" in df.columns else "model_id"
    plot_source = df.copy()
    fallback = plot_source["model_id"] if "model_id" in plot_source.columns else plot_source.get("experiment_name")
    plot_source["display_name"] = plot_source[label_col].fillna(fallback)
    if "macro_f1" in plot_source.columns:
        plot_source = plot_source.sort_values("macro_f1", ascending=False)
    plot_df = plot_source[["display_name", "accuracy", "macro_f1", "weighted_f1"]].melt(id_vars="display_name", var_name="metric", value_name="value")
    apply_publication_style(plt.matplotlib)
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    sns.barplot(data=plot_df, x="display_name", y="value", hue="metric", ax=ax, palette="Set2")
    ax.set_title("CS3 Model Performance Comparison")
    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=20)
    save_figure_bundle(fig, output_base)
    plt.close(fig)


def plot_ablation(ablation_csv: Path, output_base: Path) -> None:
    df = pd.read_csv(ablation_csv)
    plot_df = df.loc[df["condition"] == "nominal"].copy() if "condition" in df.columns else df.copy()
    if "ablation_name" not in plot_df.columns:
        plot_df["ablation_name"] = plot_df.get("modality_setting", plot_df.get("experiment_name", "unknown"))
    if "model_id" not in plot_df.columns:
        plot_df["model_id"] = plot_df.get("algorithm_name", plot_df.get("experiment_name", "unknown"))
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
    if "algorithm_name" in df.columns:
        df["display_name"] = df["algorithm_name"].fillna(df["model_id"])
    else:
        df["display_name"] = df["model_id"]
    apply_publication_style(plt.matplotlib)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    sns.lineplot(data=df, x="epoch", y="val_macro_f1", hue="display_name", linewidth=2.2, ax=ax)
    ax.set_title("CS3 Training Curves")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation macro F1")
    ax.set_ylim(0, 1.05)
    save_figure_bundle(fig, output_base)
    plt.close(fig)


def plot_inference_latency(summary_csv: Path, output_base: Path) -> None:
    df = pd.read_csv(summary_csv)
    if "inference_latency_ms" not in df.columns:
        return
    plot_source = df.loc[df["inference_latency_ms"].notna()].copy()
    if plot_source.empty:
        return
    label_col = "algorithm_name" if "algorithm_name" in plot_source.columns else "model_id"
    fallback = plot_source["model_id"] if "model_id" in plot_source.columns else plot_source.get("experiment_name")
    plot_source["display_name"] = plot_source[label_col].fillna(fallback)
    plot_source = plot_source.sort_values("inference_latency_ms", ascending=True)
    apply_publication_style(plt.matplotlib)
    fig, ax = plt.subplots(figsize=(9.5, 5))
    sns.barplot(data=plot_source, x="display_name", y="inference_latency_ms", ax=ax, color="#2a9d8f")
    ax.set_title("CS3 Inference Latency Comparison")
    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Latency (ms)")
    ax.tick_params(axis="x", rotation=20)
    save_figure_bundle(fig, output_base)
    plt.close(fig)


def generate_cs3_figures(
    project_root: Path,
    summary_csv: Path | None = None,
    ablation_csv: Path | None = None,
    output_dir: Path | None = None,
) -> None:
    summary_path = summary_csv or project_root / "outputs" / "tables" / "cs3_master_model_summary.csv"
    if not summary_path.exists():
        summary_path = project_root / "outputs" / "csv" / "cs3" / "model_performance_summary.csv"
    ablation_path = ablation_csv or project_root / "outputs" / "tables" / "cs3_ablation_summary.csv"
    if not ablation_path.exists():
        ablation_path = project_root / "outputs" / "csv" / "cs3" / "ablation_results.csv"
    figure_dir = output_dir or project_root / "outputs" / "figures" / "cs3"
    figure_dir.mkdir(parents=True, exist_ok=True)

    if summary_path.exists():
        plot_model_comparison(summary_path, figure_dir / "model_comparison_barplot")
        plot_inference_latency(summary_path, figure_dir / "inference_latency_comparison")
    if ablation_path.exists():
        plot_ablation(ablation_path, figure_dir / "ablation_comparison")
    training_curve_path = project_root / "outputs" / "csv" / "cs3" / "training_curves.csv"
    if training_curve_path.exists():
        plot_training_curves(training_curve_path, figure_dir / "training_curves")
    for confusion_name in ("baseline", "deep", "transformer"):
        candidate = project_root / "outputs" / "csv" / "cs3" / f"confusion_matrix_{confusion_name}.csv"
        if candidate.exists():
            _plot_confusion(candidate, figure_dir / f"confusion_matrix_{confusion_name}", f"CS3 {confusion_name.title()} Confusion Matrix")
