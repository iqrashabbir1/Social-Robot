from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.common.io_utils import write_dataframe, write_json
from src.common.paths import Paper1Paths


def _find_latest_file(root: Path, filename: str) -> Path | None:
    matches = sorted(root.rglob(filename), key=lambda item: item.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def _frame_timeseries_from_event_log(event_log: pd.DataFrame) -> pd.DataFrame:
    camera_df = event_log.loc[event_log["topic"] == "/camera/image_raw"].copy()
    if camera_df.empty:
        return pd.DataFrame(
            columns=[
                "time_bin_sec",
                "frame_count",
                "estimated_fps",
                "runtime_type",
                "data_source_type",
                "evidence_level",
                "source_status",
            ]
        )
    camera_df["received_time_sec"] = pd.to_numeric(camera_df["received_time_sec"], errors="coerce")
    camera_df = camera_df.dropna(subset=["received_time_sec"]).sort_values("received_time_sec")
    origin = float(camera_df["received_time_sec"].iloc[0])
    camera_df["relative_time_sec"] = camera_df["received_time_sec"] - origin
    camera_df["time_bin_sec"] = camera_df["relative_time_sec"].floordiv(1).astype(int)
    fps_df = camera_df.groupby("time_bin_sec", as_index=False).size().rename(columns={"size": "frame_count"})
    fps_df["estimated_fps"] = fps_df["frame_count"].astype(float)
    fps_df["runtime_type"] = camera_df["runtime_type"].iloc[0]
    fps_df["data_source_type"] = "mixed"
    fps_df["evidence_level"] = camera_df["evidence_level"].iloc[0]
    fps_df["source_status"] = "available"
    return fps_df


def _health_timeseries_from_csv(health_csv: Path | None) -> pd.DataFrame:
    if health_csv is None or not health_csv.exists():
        return pd.DataFrame(
            [
                {
                    "received_time_sec": None,
                    "cpu_percent": None,
                    "memory_percent": None,
                    "runtime_label": "ros2_live_windows_stream_wsl_core",
                    "runtime_type": "ros2_live_windows_stream_wsl_core",
                    "source_status": "missing",
                    "evidence_level": "pilot_demonstration",
                    "assumption_note": "No ros2_system_health.csv was found in the local repo.",
                }
            ]
        )
    df = pd.read_csv(health_csv)
    if df.empty:
        return pd.DataFrame(
            [
                {
                    "received_time_sec": None,
                    "cpu_percent": None,
                    "memory_percent": None,
                    "runtime_label": "ros2_live_windows_stream_wsl_core",
                    "runtime_type": "ros2_live_windows_stream_wsl_core",
                    "source_status": "empty",
                    "evidence_level": "pilot_demonstration",
                    "assumption_note": "The health CSV exists but contained no rows.",
                }
            ]
        )
    df["cpu_percent"] = pd.to_numeric(df.get("cpu_percent"), errors="coerce")
    df["memory_percent"] = pd.to_numeric(df.get("memory_percent"), errors="coerce")
    df["received_time_sec"] = pd.to_numeric(df.get("received_time_sec"), errors="coerce")
    df["source_status"] = "available"
    if "assumption_note" not in df.columns:
        df["assumption_note"] = ""
    return df


def collect_hybrid_runtime_metrics(project_root: Path, event_log: Path | None = None, system_health: Path | None = None) -> dict[str, str]:
    paths = Paper1Paths.from_project_root(project_root)
    paths.ensure()

    event_log = event_log or _find_latest_file(paths.outputs_logs, "ros2_event_log.csv")
    system_health = system_health or _find_latest_file(paths.outputs_logs, "ros2_system_health.csv")

    fps_csv_path = paths.outputs_csv_paper1 / "hybrid_camera_fps_timeseries.csv"
    health_csv_path = paths.outputs_csv_paper1 / "system_health_timeseries.csv"
    metrics_csv_path = paths.outputs_csv_paper1 / "hybrid_runtime_metrics.csv"
    summary_json_path = paths.outputs_csv_paper1 / "hybrid_runtime_summary.json"
    verification_csv_path = paths.outputs_csv_paper1 / "ros2_runtime_verification.csv"
    architecture_nodes_path = paths.outputs_csv_paper1 / "hybrid_system_architecture_nodes.csv"
    architecture_edges_path = paths.outputs_csv_paper1 / "hybrid_system_architecture_edges.csv"
    runtime_mode_comparison_path = paths.outputs_csv_paper1 / "runtime_mode_comparison.csv"

    architecture_nodes = pd.DataFrame(
        [
            {"node_id": "windows_webcam", "label": "Windows webcam", "layer": "capture", "host": "Windows"},
            {"node_id": "camera_streamer", "label": "camera_streamer.py", "layer": "stream", "host": "Windows"},
            {"node_id": "tcp_bridge", "label": "TCP JPEG stream", "layer": "transport", "host": "Windows->WSL"},
            {"node_id": "camera_node", "label": "camera_node (windows_stream_bridge)", "layer": "bridge", "host": "WSL"},
            {"node_id": "image_topic", "label": "/camera/image_raw", "layer": "topic", "host": "WSL"},
            {"node_id": "digital_twin", "label": "digital_twin_node", "layer": "processing", "host": "WSL"},
            {"node_id": "emotion_inference", "label": "emotion_inference_node", "layer": "processing", "host": "WSL"},
            {"node_id": "event_logger", "label": "event_logger_node", "layer": "logging", "host": "WSL"},
            {"node_id": "rosbag", "label": "rosbag record", "layer": "logging", "host": "WSL"},
        ]
    )
    architecture_edges = pd.DataFrame(
        [
            {"source": "windows_webcam", "target": "camera_streamer", "edge_label": "OpenCV frames"},
            {"source": "camera_streamer", "target": "tcp_bridge", "edge_label": "JPEG/TCP"},
            {"source": "tcp_bridge", "target": "camera_node", "edge_label": "socket receive"},
            {"source": "camera_node", "target": "image_topic", "edge_label": "sensor_msgs/Image"},
            {"source": "image_topic", "target": "digital_twin", "edge_label": "subscribe"},
            {"source": "image_topic", "target": "emotion_inference", "edge_label": "subscribe"},
            {"source": "image_topic", "target": "event_logger", "edge_label": "logged"},
            {"source": "digital_twin", "target": "event_logger", "edge_label": "event_log"},
            {"source": "image_topic", "target": "rosbag", "edge_label": "record"},
        ]
    )
    write_dataframe(architecture_nodes_path, architecture_nodes)
    write_dataframe(architecture_edges_path, architecture_edges)

    verification_df = pd.DataFrame(
        [
            {
                "component": "WSL ROS 2 Jazzy runtime",
                "verification_state": "verified",
                "runtime_type": "ros2_live_windows_stream_wsl_core",
                "evidence_level": "framework_validation",
                "note": "Verified in the user WSL runtime, not directly rerun from this Windows shell.",
            },
            {
                "component": "Windows camera streamer",
                "verification_state": "verified",
                "runtime_type": "ros2_live_windows_stream_wsl_core",
                "evidence_level": "pilot_demonstration",
                "note": "User confirmed the Windows streamer was listening on TCP port 5001.",
            },
            {
                "component": "camera_node windows_stream_bridge mode",
                "verification_state": "verified",
                "runtime_type": "ros2_live_windows_stream_wsl_core",
                "evidence_level": "framework_validation",
                "note": "User launch log showed camera_node starting in windows_stream_bridge mode.",
            },
            {
                "component": "/camera/image_raw topic",
                "verification_state": "verified",
                "runtime_type": "ros2_live_windows_stream_wsl_core",
                "evidence_level": "pilot_demonstration",
                "note": "Current verified status from the project conversation states the image topic was confirmed in ROS 2.",
            },
        ]
    )
    write_dataframe(verification_csv_path, verification_df)

    mode_comparison_df = pd.DataFrame(
        [
            {
                "runtime_type": "ros2_playback_grounded",
                "camera_source": "recorded playback or emulated stream",
                "live_runtime_verified": 1,
                "camera_input_available": 1,
                "rosbag_available": 1,
                "image_topic_verified": 1,
                "hardware_dependency_robustness": 3,
                "evidence_level": "framework_validation",
                "current_paper_role": "controlled fallback baseline",
                "limitations": "No live sensing; suitable for repeatable system validation only.",
            },
            {
                "runtime_type": "ros2_live_laptop_sensors",
                "camera_source": "WSL local webcam capture",
                "live_runtime_verified": 1,
                "camera_input_available": 1,
                "rosbag_available": 1,
                "image_topic_verified": 1,
                "hardware_dependency_robustness": 1,
                "evidence_level": "pilot_demonstration",
                "current_paper_role": "legacy live mode",
                "limitations": "Direct webcam access inside WSL is fragile and host-dependent.",
            },
            {
                "runtime_type": "ros2_live_windows_stream_wsl_core",
                "camera_source": "Windows webcam streamer over TCP to WSL",
                "live_runtime_verified": 1,
                "camera_input_available": 1,
                "rosbag_available": 1,
                "image_topic_verified": 1,
                "hardware_dependency_robustness": 2,
                "evidence_level": "pilot_demonstration",
                "current_paper_role": "Paper 1 live baseline",
                "limitations": "Requires both Windows streamer and WSL ROS graph; still not a robot deployment.",
            },
            {
                "runtime_type": "ros2_live_robot",
                "camera_source": "future robot-mounted sensors",
                "live_runtime_verified": 0,
                "camera_input_available": 0,
                "rosbag_available": 0,
                "image_topic_verified": 0,
                "hardware_dependency_robustness": 0,
                "evidence_level": "deployment_not_claimed",
                "current_paper_role": "future extension only",
                "limitations": "Not implemented in Paper 1.",
            },
        ]
    )
    write_dataframe(runtime_mode_comparison_path, mode_comparison_df)

    if event_log is not None and event_log.exists():
        event_df = pd.read_csv(event_log)
        fps_df = _frame_timeseries_from_event_log(event_df)
        duration_sec = None
        if not fps_df.empty and len(event_df.loc[event_df["topic"] == "/camera/image_raw"]) >= 2:
            camera_times = pd.to_numeric(event_df.loc[event_df["topic"] == "/camera/image_raw", "received_time_sec"], errors="coerce").dropna()
            if not camera_times.empty:
                duration_sec = float(camera_times.max() - camera_times.min())
        frame_count = int(len(event_df.loc[event_df["topic"] == "/camera/image_raw"]))
        summary = {
            "runtime_type": "ros2_live_windows_stream_wsl_core",
            "source_status": "available",
            "event_log_path": str(event_log),
            "system_health_path": str(system_health) if system_health is not None else None,
            "frame_count": frame_count,
            "runtime_duration_sec": duration_sec,
            "mean_estimated_fps": float(fps_df["estimated_fps"].mean()) if not fps_df.empty else None,
            "data_source_type": "mixed",
            "evidence_level": "pilot_demonstration",
        }
    else:
        fps_df = pd.DataFrame(
            [
                {
                    "time_bin_sec": None,
                    "frame_count": None,
                    "estimated_fps": None,
                    "runtime_type": "ros2_live_windows_stream_wsl_core",
                    "data_source_type": "mixed",
                    "evidence_level": "pilot_demonstration",
                    "source_status": "missing",
                }
            ]
        )
        summary = {
            "runtime_type": "ros2_live_windows_stream_wsl_core",
            "source_status": "missing",
            "event_log_path": None,
            "system_health_path": str(system_health) if system_health is not None else None,
            "frame_count": None,
            "runtime_duration_sec": None,
            "mean_estimated_fps": None,
            "data_source_type": "mixed",
            "evidence_level": "pilot_demonstration",
            "assumption_note": "No ros2_event_log.csv was found in the local repo. Use how_to_record_hybrid_rosbag.md and how_to_regenerate_hybrid_figures.md after a live hybrid run.",
        }
    health_df = _health_timeseries_from_csv(system_health)

    metrics_df = pd.DataFrame(
        [
            {
                "metric": "frame_count",
                "value": summary.get("frame_count"),
                "runtime_type": summary["runtime_type"],
                "data_source_type": summary["data_source_type"],
                "evidence_level": summary["evidence_level"],
                "source_status": summary["source_status"],
                "assumption_note": summary.get("assumption_note", ""),
            },
            {
                "metric": "runtime_duration_sec",
                "value": summary.get("runtime_duration_sec"),
                "runtime_type": summary["runtime_type"],
                "data_source_type": summary["data_source_type"],
                "evidence_level": summary["evidence_level"],
                "source_status": summary["source_status"],
                "assumption_note": summary.get("assumption_note", ""),
            },
            {
                "metric": "mean_estimated_fps",
                "value": summary.get("mean_estimated_fps"),
                "runtime_type": summary["runtime_type"],
                "data_source_type": summary["data_source_type"],
                "evidence_level": summary["evidence_level"],
                "source_status": summary["source_status"],
                "assumption_note": summary.get("assumption_note", ""),
            },
            {
                "metric": "mean_cpu_percent",
                "value": float(health_df["cpu_percent"].dropna().mean()) if "cpu_percent" in health_df.columns and not health_df["cpu_percent"].dropna().empty else None,
                "runtime_type": summary["runtime_type"],
                "data_source_type": summary["data_source_type"],
                "evidence_level": summary["evidence_level"],
                "source_status": "available" if "cpu_percent" in health_df.columns and not health_df["cpu_percent"].dropna().empty else "missing",
                "assumption_note": "" if "cpu_percent" in health_df.columns and not health_df["cpu_percent"].dropna().empty else "System health CSV not yet exported from a live hybrid run.",
            },
            {
                "metric": "mean_memory_percent",
                "value": float(health_df["memory_percent"].dropna().mean()) if "memory_percent" in health_df.columns and not health_df["memory_percent"].dropna().empty else None,
                "runtime_type": summary["runtime_type"],
                "data_source_type": summary["data_source_type"],
                "evidence_level": summary["evidence_level"],
                "source_status": "available" if "memory_percent" in health_df.columns and not health_df["memory_percent"].dropna().empty else "missing",
                "assumption_note": "" if "memory_percent" in health_df.columns and not health_df["memory_percent"].dropna().empty else "System health CSV not yet exported from a live hybrid run.",
            },
        ]
    )

    write_dataframe(fps_csv_path, fps_df)
    write_dataframe(health_csv_path, health_df)
    write_dataframe(metrics_csv_path, metrics_df)
    write_json(summary_json_path, summary)
    return {
        "fps_csv": str(fps_csv_path),
        "health_csv": str(health_csv_path),
        "metrics_csv": str(metrics_csv_path),
        "summary_json": str(summary_json_path),
        "verification_csv": str(verification_csv_path),
        "architecture_nodes_csv": str(architecture_nodes_path),
        "architecture_edges_csv": str(architecture_edges_path),
        "runtime_mode_comparison_csv": str(runtime_mode_comparison_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Paper 1 hybrid runtime metrics from available logs.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--event-log", default="")
    parser.add_argument("--system-health", default="")
    args = parser.parse_args()
    collect_hybrid_runtime_metrics(
        project_root=Path(args.project_root).resolve(),
        event_log=Path(args.event_log).resolve() if args.event_log else None,
        system_health=Path(args.system_health).resolve() if args.system_health else None,
    )


if __name__ == "__main__":
    main()
