from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from src.visualization.plot_style import COLOR_PALETTE, apply_publication_style, save_figure_bundle


def plot_hybrid_system_architecture(nodes_csv: Path, edges_csv: Path, output_base: Path) -> None:
    nodes_df = pd.read_csv(nodes_csv)
    edges_df = pd.read_csv(edges_csv)
    positions = {
        "windows_webcam": (0.7, 3.4),
        "camera_streamer": (3.0, 3.4),
        "tcp_bridge": (5.4, 3.4),
        "camera_node": (7.8, 3.4),
        "image_topic": (10.2, 3.4),
        "digital_twin": (12.6, 4.8),
        "emotion_inference": (12.6, 3.3),
        "event_logger": (12.6, 1.8),
        "rosbag": (15.0, 1.8),
    }
    color_map = {
        "capture": COLOR_PALETTE["navy"],
        "stream": COLOR_PALETTE["teal"],
        "transport": COLOR_PALETTE["gold"],
        "bridge": COLOR_PALETTE["orange"],
        "topic": COLOR_PALETTE["slate"],
        "processing": COLOR_PALETTE["red"],
        "logging": COLOR_PALETTE["mint"],
    }
    apply_publication_style(plt.matplotlib)
    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(16.5, 6.2))
    ax.axis("off")
    ax.set_xlim(0, 17.5)
    ax.set_ylim(0.8, 6.4)
    ax.text(2.0, 5.8, "Windows host", fontsize=15, weight="bold", color=COLOR_PALETTE["navy"])
    ax.text(10.6, 5.8, "WSL ROS 2 core", fontsize=15, weight="bold", color=COLOR_PALETTE["red"])

    for row in nodes_df.to_dict(orient="records"):
        x, y = positions[row["node_id"]]
        box = FancyBboxPatch(
            (x, y),
            1.85,
            0.9,
            boxstyle="round,pad=0.04,rounding_size=0.08",
            facecolor=color_map.get(row["layer"], COLOR_PALETTE["slate"]),
            edgecolor="white",
            linewidth=1.2,
        )
        ax.add_patch(box)
        ax.text(x + 0.925, y + 0.55, row["label"], ha="center", va="center", color="white", fontsize=9.5)
        ax.text(x + 0.925, y + 0.22, row["host"], ha="center", va="center", color="white", fontsize=8)

    for row in edges_df.to_dict(orient="records"):
        start_x, start_y = positions[row["source"]]
        end_x, end_y = positions[row["target"]]
        arrow = FancyArrowPatch(
            (start_x + 1.85, start_y + 0.45),
            (end_x - 0.08, end_y + 0.45),
            arrowstyle="->",
            mutation_scale=14,
            linewidth=1.8,
            color="#3b3b3b",
        )
        ax.add_patch(arrow)
        mid_x = (start_x + end_x) / 2.0 + 0.9
        mid_y = (start_y + end_y) / 2.0 + 0.75
        ax.text(mid_x, mid_y, row["edge_label"], fontsize=8.2, color="#444444", ha="center")

    ax.set_title("Paper 1 Hybrid Windows-Camera and WSL-ROS Runtime Architecture")
    save_figure_bundle(fig, output_base)
    plt.close(fig)


def plot_runtime_verification(verification_csv: Path, output_base: Path) -> None:
    df = pd.read_csv(verification_csv)
    verification_order = ["verified", "partial", "planned"]
    df["verification_score"] = df["verification_state"].map({"verified": 1.0, "partial": 0.5, "planned": 0.0}).fillna(0.0)
    apply_publication_style(plt.matplotlib)
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    sns.barplot(data=df, x="component", y="verification_score", hue="verification_state", hue_order=verification_order, dodge=False, palette="Set2", ax=ax)
    ax.set_title("Verified Elements of the Hybrid ROS 2 Runtime")
    ax.set_xlabel("Runtime component")
    ax.set_ylabel("Verification score")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=18)
    if ax.legend_ is not None:
        ax.legend_.set_title("State")
    save_figure_bundle(fig, output_base)
    plt.close(fig)


def plot_hybrid_camera_fps(fps_csv: Path, output_base: Path) -> None:
    df = pd.read_csv(fps_csv)
    apply_publication_style(plt.matplotlib)
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(9, 5))
    available = df.loc[df["source_status"] == "available"].copy() if "source_status" in df.columns else df.copy()
    if available.empty:
        ax.text(0.5, 0.5, "No hybrid frame-rate log exported in the local repo.\nRegenerate after a live rosbag or event-log export.", ha="center", va="center", fontsize=12)
        ax.set_axis_off()
    else:
        sns.lineplot(data=available, x="time_bin_sec", y="estimated_fps", marker="o", color=COLOR_PALETTE["teal"], linewidth=2.5, ax=ax)
        ax.set_title("Hybrid Camera Frame Rate Over Time")
        ax.set_xlabel("Runtime time (s)")
        ax.set_ylabel("Estimated FPS")
    save_figure_bundle(fig, output_base)
    plt.close(fig)


def plot_system_health(system_csv: Path, output_base: Path) -> None:
    df = pd.read_csv(system_csv)
    apply_publication_style(plt.matplotlib)
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    available = df.loc[df["source_status"] == "available"].copy() if "source_status" in df.columns else df.copy()
    if available.empty or available["received_time_sec"].dropna().empty:
        ax.text(0.5, 0.5, "No hybrid system-health CSV is available yet.\nThe live graph remains unchanged; rerun event_logger_node after a hybrid session.", ha="center", va="center", fontsize=11.5)
        ax.set_axis_off()
    else:
        available["received_time_sec"] = pd.to_numeric(available["received_time_sec"], errors="coerce")
        available["cpu_percent"] = pd.to_numeric(available["cpu_percent"], errors="coerce")
        available["memory_percent"] = pd.to_numeric(available["memory_percent"], errors="coerce")
        origin = available["received_time_sec"].dropna().min()
        available["relative_time_sec"] = available["received_time_sec"] - origin
        ax2 = ax.twinx()
        sns.lineplot(data=available, x="relative_time_sec", y="cpu_percent", color=COLOR_PALETTE["orange"], linewidth=2.4, marker="o", ax=ax)
        sns.lineplot(data=available, x="relative_time_sec", y="memory_percent", color=COLOR_PALETTE["navy"], linewidth=2.4, marker="s", ax=ax2)
        ax.set_title("Hybrid Runtime System Health Over Time")
        ax.set_xlabel("Runtime time (s)")
        ax.set_ylabel("CPU (%)", color=COLOR_PALETTE["orange"])
        ax2.set_ylabel("Memory (%)", color=COLOR_PALETTE["navy"])
    save_figure_bundle(fig, output_base)
    plt.close(fig)


def plot_runtime_mode_comparison(comparison_csv: Path, output_base: Path) -> None:
    df = pd.read_csv(comparison_csv)
    compare_df = df.melt(
        id_vars=["runtime_type"],
        value_vars=["live_runtime_verified", "camera_input_available", "rosbag_available", "image_topic_verified", "hardware_dependency_robustness"],
        var_name="metric",
        value_name="value",
    )
    compare_df["metric"] = compare_df["metric"].str.replace("_", " ", regex=False).str.title()
    apply_publication_style(plt.matplotlib)
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(11.5, 5.5))
    sns.barplot(data=compare_df, x="metric", y="value", hue="runtime_type", ax=ax, palette="Set2")
    ax.set_title("Runtime Mode Comparison for Paper 1")
    ax.set_xlabel("Criterion")
    ax.set_ylabel("Score / availability")
    ax.tick_params(axis="x", rotation=18)
    save_figure_bundle(fig, output_base)
    plt.close(fig)


def plot_hybrid_system_architecture_paper_ready(nodes_csv: Path, edges_csv: Path, output_base: Path) -> None:
    _ = pd.read_csv(edges_csv)
    nodes_df = pd.read_csv(nodes_csv)
    labels = {row["node_id"]: row["label"] for row in nodes_df.to_dict(orient="records")}
    positions = {
        "windows_webcam": (0.6, 5.1),
        "camera_streamer": (3.1, 5.1),
        "tcp_bridge": (5.6, 5.1),
        "camera_node": (8.1, 5.1),
        "image_topic": (10.6, 5.1),
        "digital_twin": (13.1, 6.3),
        "emotion_inference": (13.1, 5.1),
        "event_logger": (13.1, 3.9),
        "rosbag": (15.6, 3.9),
    }
    color_map = {
        "windows_webcam": COLOR_PALETTE["navy"],
        "camera_streamer": COLOR_PALETTE["teal"],
        "tcp_bridge": COLOR_PALETTE["gold"],
        "camera_node": COLOR_PALETTE["orange"],
        "image_topic": COLOR_PALETTE["slate"],
        "digital_twin": COLOR_PALETTE["red"],
        "emotion_inference": COLOR_PALETTE["red"],
        "event_logger": COLOR_PALETTE["mint"],
        "rosbag": COLOR_PALETTE["mint"],
    }
    apply_publication_style(plt.matplotlib)
    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(16.8, 7.4))
    ax.axis("off")
    ax.set_xlim(0, 18.1)
    ax.set_ylim(3.0, 7.4)
    ax.text(2.1, 6.95, "Windows capture side", fontsize=16, weight="bold", color=COLOR_PALETTE["navy"])
    ax.text(12.3, 6.95, "WSL ROS 2 processing side", fontsize=16, weight="bold", color=COLOR_PALETTE["red"])

    for node_id, (x, y) in positions.items():
        face_color = color_map[node_id]
        box = FancyBboxPatch(
            (x, y),
            1.9,
            0.62,
            boxstyle="round,pad=0.04,rounding_size=0.08",
            facecolor=face_color,
            edgecolor="white",
            linewidth=1.2,
        )
        ax.add_patch(box)
        ax.text(x + 0.95, y + 0.33, labels[node_id], ha="center", va="center", color="white", fontsize=10)

    linear_chain = ["windows_webcam", "camera_streamer", "tcp_bridge", "camera_node", "image_topic"]
    for source, target in zip(linear_chain, linear_chain[1:]):
        sx, sy = positions[source]
        tx, ty = positions[target]
        ax.add_patch(
            FancyArrowPatch(
                (sx + 1.9, sy + 0.31),
                (tx - 0.06, ty + 0.31),
                arrowstyle="->",
                mutation_scale=16,
                linewidth=2.0,
                color="#444444",
            )
        )

    branch_targets = ["digital_twin", "emotion_inference", "event_logger", "rosbag"]
    for target in branch_targets:
        tx, ty = positions[target]
        ax.add_patch(
            FancyArrowPatch(
                (positions["image_topic"][0] + 1.9, positions["image_topic"][1] + 0.31),
                (tx - 0.08, ty + 0.31),
                connectionstyle="arc3,rad=0.0",
                arrowstyle="->",
                mutation_scale=15,
                linewidth=2.0,
                color="#444444",
            )
        )

    ax.text(4.3, 5.82, "OpenCV frames", fontsize=9.5, color="#444444", ha="center")
    ax.text(6.8, 5.82, "JPEG/TCP", fontsize=9.5, color="#444444", ha="center")
    ax.text(9.3, 5.82, "WSL bridge", fontsize=9.5, color="#444444", ha="center")
    ax.text(11.85, 5.82, "sensor_msgs/Image", fontsize=9.5, color="#444444", ha="center")
    ax.set_title("Paper 1 Hybrid Windows-Camera to WSL-ROS Runtime", pad=18)
    save_figure_bundle(fig, output_base)
    plt.close(fig)


def plot_runtime_mode_heatmap_paper_ready(comparison_csv: Path, output_base: Path) -> None:
    df = pd.read_csv(comparison_csv).copy()
    display = df.loc[df["runtime_type"] != "ros2_live_robot"].copy()
    display["runtime_type"] = display["runtime_type"].replace(
        {
            "ros2_playback_grounded": "Playback-grounded",
            "ros2_live_laptop_sensors": "Live WSL laptop sensors",
            "ros2_live_windows_stream_wsl_core": "Hybrid Windows stream + WSL core",
        }
    )
    metric_columns = [
        "live_runtime_verified",
        "camera_input_available",
        "rosbag_available",
        "image_topic_verified",
        "hardware_dependency_robustness",
    ]
    heatmap_df = display.set_index("runtime_type")[metric_columns]
    heatmap_df.columns = [
        "Live runtime verified",
        "Camera input available",
        "Rosbag available",
        "Image topic verified",
        "Hardware robustness",
    ]
    apply_publication_style(plt.matplotlib)
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10.8, 4.8))
    sns.heatmap(heatmap_df, annot=True, fmt=".0f", cmap="YlGnBu", cbar_kws={"label": "Score"}, ax=ax)
    ax.set_title("Paper 1 Runtime Evidence Matrix")
    ax.set_xlabel("Criterion")
    ax.set_ylabel("Runtime mode")
    save_figure_bundle(fig, output_base)
    plt.close(fig)


def save_pilot_real_anchor_panel(frame_manifest_csv: Path, output_base: Path) -> None:
    frame_df = pd.read_csv(frame_manifest_csv)
    if frame_df.empty or "frame_path" not in frame_df.columns:
        return
    selected = frame_df.copy()
    if len(selected) > 4:
        indices = sorted({0, len(selected) // 3, (2 * len(selected)) // 3, len(selected) - 1})
        selected = selected.iloc[indices]
    apply_publication_style(plt.matplotlib)
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.6))
    for axis, row in zip(axes.flatten(), selected.to_dict(orient="records")):
        axis.imshow(mpimg.imread(row["frame_path"]))
        axis.axis("off")
        axis.set_title(f"t = {float(row.get('timestamp_ms', 0.0)) / 1000.0:.2f}s", fontsize=10)
    for axis in axes.flatten()[len(selected) :]:
        axis.axis("off")
    fig.suptitle("Pilot Real-Anchor Visual Capture Example", fontsize=15)
    fig.tight_layout()
    save_figure_bundle(fig, output_base)
    plt.close(fig)
