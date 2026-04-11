from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd

from src.common.io_utils import write_dataframe
from src.common.paths import Paper1Paths
from src.visualization.plot_style import apply_publication_style


def _load_frame_manifest(frame_csv: Path | None, frame_dir: Path | None) -> pd.DataFrame:
    if frame_csv is not None and frame_csv.exists():
        manifest = pd.read_csv(frame_csv)
        if "frame_path" in manifest.columns:
            return manifest.loc[manifest["frame_path"].notna()].copy()
    if frame_dir is not None and frame_dir.exists():
        images = sorted(list(frame_dir.glob("*.jpg")) + list(frame_dir.glob("*.png")))
        if images:
            return pd.DataFrame({"frame_path": [str(path) for path in images], "frame_index": list(range(len(images)))})
    return pd.DataFrame(columns=["frame_path", "frame_index"])


def _save_placeholder(output_path: Path, message: str) -> None:
    apply_publication_style(plt.matplotlib)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=12)
    ax.set_axis_off()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_camera_sample_frames(
    project_root: Path,
    frame_csv: Path | None = None,
    frame_dir: Path | None = None,
    label: str = "Live hybrid Windows-camera to WSL-ROS stream",
) -> dict[str, str]:
    paths = Paper1Paths.from_project_root(project_root)
    paths.ensure()

    if frame_csv is None:
        default_manifest = paths.outputs_csv_paper1 / "hybrid_camera_frame_manifest.csv"
        if default_manifest.exists():
            frame_csv = default_manifest

    frame_manifest = _load_frame_manifest(frame_csv, frame_dir)
    sample_frame_path = paths.outputs_figures_paper1 / "hybrid_camera_sample_frame.png"
    sample_panel_path = paths.outputs_figures_paper1 / "hybrid_camera_sample_panel.png"
    manifest_output_path = paths.outputs_csv_paper1 / "hybrid_camera_frame_manifest.csv"

    if frame_manifest.empty:
        placeholder_message = (
            "No hybrid camera frame export is tracked in the local repo.\n"
            "Run the hybrid runtime and export a rosbag or frame manifest,\n"
            "then regenerate this figure from docs/paper1/how_to_regenerate_hybrid_figures.md."
        )
        _save_placeholder(sample_frame_path, placeholder_message)
        _save_placeholder(sample_panel_path, placeholder_message)
        write_dataframe(
            manifest_output_path,
            pd.DataFrame(
                [
                    {
                        "frame_path": "",
                        "frame_index": None,
                        "source_status": "missing",
                        "runtime_type": "ros2_live_windows_stream_wsl_core",
                        "evidence_level": "pilot_demonstration",
                        "label": label,
                    }
                ]
            ),
        )
        return {
            "sample_frame": str(sample_frame_path),
            "sample_panel": str(sample_panel_path),
            "manifest_csv": str(manifest_output_path),
        }

    frame_manifest = frame_manifest.reset_index(drop=True)
    selected_indices = sorted({0, len(frame_manifest) // 3, (2 * len(frame_manifest)) // 3, len(frame_manifest) - 1})
    selected = frame_manifest.iloc[selected_indices].copy()
    selected["source_status"] = "available"
    selected["runtime_type"] = "ros2_live_windows_stream_wsl_core"
    selected["evidence_level"] = "pilot_demonstration"
    selected["label"] = label
    write_dataframe(manifest_output_path, selected)

    representative = mpimg.imread(selected.iloc[len(selected) // 2]["frame_path"])
    apply_publication_style(plt.matplotlib)
    fig, ax = plt.subplots(figsize=(7.8, 5.6))
    ax.imshow(representative)
    ax.axis("off")
    ax.set_title(label)
    sample_frame_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(sample_frame_path, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8))
    for axis, row in zip(axes.flatten(), selected.to_dict(orient="records")):
        axis.imshow(mpimg.imread(row["frame_path"]))
        axis.axis("off")
        axis.set_title(f"Frame {int(row.get('frame_index', 0))}")
    fig.suptitle(label, fontsize=14)
    fig.tight_layout()
    fig.savefig(sample_panel_path, bbox_inches="tight")
    plt.close(fig)
    return {
        "sample_frame": str(sample_frame_path),
        "sample_panel": str(sample_panel_path),
        "manifest_csv": str(manifest_output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Save representative Paper 1 camera sample frames.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--frame-csv", default="")
    parser.add_argument("--frame-dir", default="")
    parser.add_argument("--label", default="Live hybrid Windows-camera to WSL-ROS stream")
    args = parser.parse_args()
    save_camera_sample_frames(
        project_root=Path(args.project_root).resolve(),
        frame_csv=Path(args.frame_csv).resolve() if args.frame_csv else None,
        frame_dir=Path(args.frame_dir).resolve() if args.frame_dir else None,
        label=args.label,
    )


if __name__ == "__main__":
    main()
