from __future__ import annotations

from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.visualization.plot_style import COLOR_PALETTE, apply_publication_style, save_figure_bundle


def _save_text_figure(output_base: Path, message: str, title: str) -> None:
    apply_publication_style(plt.matplotlib)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=12)
    save_figure_bundle(fig, output_base)
    plt.close(fig)


def plot_dataset_sample_panel(predictions_csv: Path, output_base: Path, max_items: int = 6) -> None:
    df = pd.read_csv(predictions_csv).head(max_items)
    if df.empty:
        _save_text_figure(output_base, "No dataset samples were available.", "Dataset Sample Panel")
        return
    cols = 3
    rows = (len(df) + cols - 1) // cols
    apply_publication_style(plt.matplotlib)
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4.2 * rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for axis, row in zip(axes, df.to_dict(orient="records")):
        axis.imshow(mpimg.imread(row["media_path"]))
        axis.axis("off")
        true_label = row.get("true_label") if pd.notna(row.get("true_label")) else "unlabeled"
        axis.set_title(f"true: {true_label}", fontsize=10)
    for axis in axes[len(df) :]:
        axis.axis("off")
    fig.suptitle("Controlled Dataset Sample Panel", fontsize=15)
    fig.tight_layout()
    save_figure_bundle(fig, output_base)
    plt.close(fig)


def plot_dataset_prediction_panel(predictions_csv: Path, output_base: Path, max_items: int = 6) -> None:
    df = pd.read_csv(predictions_csv).copy()
    if df.empty:
        _save_text_figure(output_base, "No dataset predictions were available.", "Dataset Prediction Panel")
        return
    df["confidence"] = pd.to_numeric(df.get("confidence"), errors="coerce")
    df = df.sort_values("confidence", ascending=False, na_position="last").head(max_items)
    cols = 3
    rows = (len(df) + cols - 1) // cols
    apply_publication_style(plt.matplotlib)
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4.2 * rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for axis, row in zip(axes, df.to_dict(orient="records")):
        axis.imshow(mpimg.imread(row["media_path"]))
        axis.axis("off")
        conf = row.get("confidence")
        conf_text = f"{conf:.1f}" if conf is not None and conf == conf else "n/a"
        axis.set_title(f"pred: {row['predicted_label']} | conf: {conf_text}", fontsize=10)
    for axis in axes[len(df) :]:
        axis.axis("off")
    fig.suptitle("Dataset Prediction Panel", fontsize=15)
    fig.tight_layout()
    save_figure_bundle(fig, output_base)
    plt.close(fig)


def plot_dataset_replay_sequence(sequence_csv: Path, output_base: Path, max_items: int = 4) -> None:
    df = pd.read_csv(sequence_csv).copy()
    if df.empty:
        _save_text_figure(output_base, "No replay sequence could be generated.", "Dataset Replay Sequence")
        return
    sort_cols = [column for column in ("timestamp_ms", "frame_index", "sample_id") if column in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, na_position="last")
    df = df.head(max_items)
    apply_publication_style(plt.matplotlib)
    fig, axes = plt.subplots(1, len(df), figsize=(4.1 * len(df), 4.8))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for axis, row in zip(axes, df.to_dict(orient="records")):
        axis.imshow(mpimg.imread(row["media_path"]))
        axis.axis("off")
        ts = row.get("timestamp_ms")
        ts_text = f"{float(ts)/1000.0:.2f}s" if ts is not None and ts == ts else "sequence order"
        axis.set_title(ts_text, fontsize=10)
    fig.suptitle("Dataset Replay Through ROS2 Pipeline", fontsize=15)
    fig.tight_layout()
    save_figure_bundle(fig, output_base)
    plt.close(fig)


def plot_dataset_confusion_matrix(confusion_csv: Path, output_base: Path) -> None:
    df = pd.read_csv(confusion_csv)
    if "note" in df.columns:
        _save_text_figure(
            output_base,
            str(df["note"].iloc[0]),
            "Dataset Confusion Matrix",
        )
        return
    matrix = df.set_index("true_label")
    apply_publication_style(plt.matplotlib)
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    sns.heatmap(matrix, annot=True, fmt=".0f", cmap="Blues", cbar=False, ax=ax)
    ax.set_title("Dataset Evaluation Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    save_figure_bundle(fig, output_base)
    plt.close(fig)


def plot_dataset_metrics_barplot(metrics_csv: Path, output_base: Path) -> None:
    df = pd.read_csv(metrics_csv)
    plot_df = df.loc[df["value"].notna()].copy()
    if plot_df.empty:
        _save_text_figure(output_base, "No numeric dataset metrics were available.", "Dataset Metrics")
        return
    apply_publication_style(plt.matplotlib)
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    sns.barplot(data=plot_df, x="metric", y="value", color=COLOR_PALETTE["teal"], ax=ax)
    ax.set_title("Dataset Evaluation Metrics")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Value")
    ax.tick_params(axis="x", rotation=18)
    save_figure_bundle(fig, output_base)
    plt.close(fig)


def generate_dataset_figures(project_root: Path, dataset_csv_dir: Path, dataset_fig_dir: Path) -> None:
    predictions_csv = dataset_csv_dir / "dataset_predictions.csv"
    sequence_csv = dataset_csv_dir / "dataset_sequence_manifest.csv"
    confusion_csv = dataset_csv_dir / "dataset_confusion_matrix.csv"
    metrics_csv = dataset_csv_dir / "dataset_metrics_summary.csv"
    plot_dataset_sample_panel(predictions_csv, dataset_fig_dir / "dataset_sample_panel")
    plot_dataset_prediction_panel(predictions_csv, dataset_fig_dir / "dataset_prediction_panel")
    plot_dataset_replay_sequence(sequence_csv, dataset_fig_dir / "dataset_replay_sequence")
    plot_dataset_confusion_matrix(confusion_csv, dataset_fig_dir / "dataset_confusion_matrix")
    plot_dataset_metrics_barplot(metrics_csv, dataset_fig_dir / "dataset_metrics_barplot")

    # Promote the latest dataset figures to the paper1 root for manuscript convenience.
    for name in (
        "dataset_sample_panel",
        "dataset_prediction_panel",
        "dataset_replay_sequence",
        "dataset_confusion_matrix",
        "dataset_metrics_barplot",
    ):
        source_png = dataset_fig_dir / f"{name}.png"
        if source_png.exists():
            target = project_root / "outputs" / "figures" / "paper1" / f"{name}.png"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source_png.read_bytes())
