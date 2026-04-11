from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from social_robot.config_helpers import nested_ros_params, resolve_ros_config
from social_robot.message_utils import image_to_rosmsg
from social_robot.runtime import ensure_rclpy


class DatasetReplayNodeBase:
    def __init__(self) -> None:
        self._rclpy, Node = ensure_rclpy()

        class DatasetReplayNode(Node):
            def __init__(inner_self) -> None:
                super().__init__("dataset_replay_node")
                default_config = Path(__file__).resolve().parents[1] / "config" / "dataset_replay.yaml"
                inner_self.declare_parameter("config_path", str(default_config))
                inner_self.declare_parameter("runtime_type", "")
                config_path = Path(str(inner_self.get_parameter("config_path").value or default_config)).expanduser().resolve()
                config = resolve_ros_config(str(config_path), default_config)
                params = nested_ros_params(config, "dataset_replay_node")
                repo_root = config_path.parents[1]

                runtime_override = str(inner_self.get_parameter("runtime_type").value or "").strip()
                inner_self.runtime_type = str(runtime_override or params.get("runtime_type", "ros2_dataset_replay"))
                dataset_root = Path(str(params.get("dataset_root", "data/pilot/sessions/paper1_anchor_demo/frames")))
                inner_self.dataset_root = dataset_root if dataset_root.is_absolute() else (repo_root / dataset_root)
                inner_self.dataset_root = inner_self.dataset_root.resolve()
                labels_csv = str(params.get("labels_csv", "") or "").strip()
                if labels_csv:
                    labels_path = Path(labels_csv)
                    inner_self.labels_csv = labels_path if labels_path.is_absolute() else (repo_root / labels_path)
                    inner_self.labels_csv = inner_self.labels_csv.resolve()
                else:
                    inner_self.labels_csv = None
                inner_self.playback_rate_hz = float(params.get("playback_rate_hz", 2.0))
                inner_self.loop = bool(params.get("loop", False))
                inner_self.camera_topic = str(params.get("camera_topic", "/camera/image_raw"))
                inner_self.event_topic = str(params.get("event_topic", "/event_log"))
                inner_self.frame_id_prefix = str(params.get("frame_id_prefix", "dataset_replay"))
                inner_self.width = int(params.get("width", 640))
                inner_self.height = int(params.get("height", 480))
                inner_self.video_frame_stride = int(params.get("video_frame_stride", 15))
                inner_self.max_frames_per_video = int(params.get("max_frames_per_video", 12))
                inner_self.dataset_name = str(params.get("dataset_name", inner_self.dataset_root.name))
                inner_self.target_label_set = str(params.get("target_label_set", "broad4_angry") or "").strip() or None
                inner_self.dataset_backend_error = None
                inner_self.records = None

                inner_self.publisher = inner_self.create_publisher(__import__("sensor_msgs.msg", fromlist=["Image"]).Image, inner_self.camera_topic, 10)
                inner_self.event_pub = inner_self.create_publisher(__import__("std_msgs.msg", fromlist=["String"]).String, inner_self.event_topic, 10)
                inner_self.stop_event = threading.Event()
                try:
                    from src.data.dataset_loader import (
                        load_dataset_records,
                        load_image_frame,
                        materialize_frame_records,
                    )

                    inner_self._load_image_frame = load_image_frame
                    records = load_dataset_records(
                        dataset_root=inner_self.dataset_root,
                        labels_csv=inner_self.labels_csv,
                        split_mode="test_only",
                        target_label_set=inner_self.target_label_set,
                    )
                    inner_self.records = materialize_frame_records(
                        records,
                        cache_dir=repo_root / "outputs" / "csv" / "paper1" / "dataset_replay_cache",
                        width=inner_self.width,
                        height=inner_self.height,
                        video_frame_stride=inner_self.video_frame_stride,
                        max_frames_per_video=inner_self.max_frames_per_video,
                    )
                    inner_self.worker = threading.Thread(target=inner_self.publish_loop, daemon=True)
                    inner_self.get_logger().info(
                        f"dataset_replay_node starting with {len(inner_self.records)} samples from {inner_self.dataset_root} at {inner_self.playback_rate_hz:.2f} Hz using target_label_set={inner_self.target_label_set or 'none'}"
                    )
                    inner_self.worker.start()
                except Exception as exc:
                    inner_self._load_image_frame = None
                    inner_self.worker = None
                    inner_self.dataset_backend_error = str(exc)
                    inner_self.get_logger().warning(
                        "dataset_replay_node disabled because dataset dependencies are unavailable. "
                        f"Install the Paper 1 Python requirements in WSL and relaunch. Detail: {exc}"
                    )

            def publish_loop(inner_self) -> None:
                if inner_self.records is None or inner_self._load_image_frame is None:
                    return
                delay = 1.0 / max(inner_self.playback_rate_hz, 0.1)
                while not inner_self.stop_event.is_set():
                    for row in inner_self.records.to_dict(orient="records"):
                        frame = inner_self._load_image_frame(
                            Path(row.get("frame_path") or row["media_path"]),
                            width=inner_self.width,
                            height=inner_self.height,
                        )
                        stamp = inner_self.get_clock().now().to_msg()
                        frame_id = f"{inner_self.frame_id_prefix}/{inner_self.dataset_name}/{row['sample_id']}"
                        msg = image_to_rosmsg(frame, stamp, frame_id=frame_id)
                        inner_self.publisher.publish(msg)

                        event_msg = __import__("std_msgs.msg", fromlist=["String"]).String()
                        event_msg.data = json.dumps(
                            {
                                "topic": inner_self.camera_topic,
                                "runtime_type": inner_self.runtime_type,
                                "dataset_name": inner_self.dataset_name,
                                "sample_id": row["sample_id"],
                                "timestamp_ms": row.get("timestamp_ms"),
                                "frame_index": row.get("frame_index"),
                                "label": row.get("label"),
                                "evidence_level": "benchmark_preliminary",
                            }
                        )
                        inner_self.event_pub.publish(event_msg)
                        time.sleep(delay)
                        if inner_self.stop_event.is_set():
                            break
                    if not inner_self.loop:
                        break

            def destroy_node(inner_self) -> bool:
                inner_self.stop_event.set()
                if inner_self.worker is not None and inner_self.worker.is_alive():
                    inner_self.worker.join(timeout=1.0)
                return super().destroy_node()

        self.node_cls = DatasetReplayNode


def main() -> None:
    runtime = DatasetReplayNodeBase()
    rclpy = runtime._rclpy
    rclpy.init()
    node = runtime.node_cls()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
