from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from social_robot.config_helpers import nested_ros_params, resolve_ros_config
from social_robot.message_utils import float_audio_to_msg, image_to_rosmsg, json_string_msg
from social_robot.runtime import ensure_rclpy
from src.data.real_anchor_loader import resolve_latest_session
from src.ros2.bag_or_emulated_replay import load_or_emulate_topic_stream


class PlaybackAdapterNodeBase:
    def __init__(self) -> None:
        self._rclpy, Node = ensure_rclpy()

        class PlaybackAdapterNode(Node):
            def __init__(inner_self) -> None:
                super().__init__("playback_adapter_node")
                default_config = Path(__file__).resolve().parents[1] / "config" / "playback_grounded.yaml"
                inner_self.declare_parameter("config_path", str(default_config))
                config = resolve_ros_config(inner_self.get_parameter("config_path").value, default_config)
                params = nested_ros_params(config, "playback_adapter_node")

                inner_self.project_root = Path(params.get("project_root", Path(__file__).resolve().parents[1])).resolve()
                inner_self.playback_source = str(params.get("playback_source", ""))
                inner_self.publish_rate = float(params.get("publish_rate", 10.0))
                inner_self.topic_df, inner_self.runtime_meta = load_or_emulate_topic_stream(
                    inner_self.project_root,
                    inner_self.playback_source,
                    seed=int(params.get("seed", 42)),
                    steps=int(params.get("steps", 60)),
                )
                inner_self.step_iter = iter(sorted(inner_self.topic_df["step"].unique()))
                inner_self.current_step = None

                std_msgs = __import__("std_msgs.msg", fromlist=["String"])
                sensor_msgs = __import__("sensor_msgs.msg", fromlist=["Image"])
                geometry_msgs = __import__("geometry_msgs.msg", fromlist=["PoseStamped"])
                diagnostic_msgs = __import__("diagnostic_msgs.msg", fromlist=["DiagnosticArray"])

                inner_self.image_pub = inner_self.create_publisher(sensor_msgs.Image, "/camera/image_raw", 10)
                inner_self.audio_pub = inner_self.create_publisher(std_msgs.Float32MultiArray, "/audio/stream", 10)
                inner_self.pose_pub = inner_self.create_publisher(geometry_msgs.PoseStamped, "/robot_pose", 10)
                inner_self.event_pub = inner_self.create_publisher(std_msgs.String, "/event_log", 10)
                inner_self.health_pub = inner_self.create_publisher(diagnostic_msgs.DiagnosticArray, "/system_health", 10)
                inner_self.timer = inner_self.create_timer(max(0.05, 1.0 / max(inner_self.publish_rate, 0.1)), inner_self.publish_next_step)

            def publish_next_step(inner_self) -> None:
                if inner_self.current_step is None:
                    try:
                        inner_self.current_step = next(inner_self.step_iter)
                    except StopIteration:
                        inner_self.get_logger().info("Playback sequence finished.")
                        inner_self.timer.cancel()
                        return
                step_df = inner_self.topic_df[inner_self.topic_df["step"] == inner_self.current_step]
                for row in step_df.to_dict(orient="records"):
                    inner_self.publish_event(row)
                try:
                    inner_self.current_step = next(inner_self.step_iter)
                except StopIteration:
                    inner_self.current_step = None

            def publish_event(inner_self, row: dict[str, object]) -> None:
                topic = str(row["topic"])
                stamp = inner_self.get_clock().now().to_msg()
                if topic == "/camera/image_raw":
                    frame = inner_self._load_or_placeholder_frame()
                    inner_self.image_pub.publish(image_to_rosmsg(frame, stamp))
                elif topic == "/audio/stream":
                    t = np.linspace(0, 0.1, 1600, endpoint=False)
                    audio = 0.05 * np.sin(2 * math.pi * 440.0 * t)
                    inner_self.audio_pub.publish(float_audio_to_msg(audio))
                elif topic == "/robot_pose":
                    PoseStamped = __import__("geometry_msgs.msg", fromlist=["PoseStamped"]).PoseStamped
                    pose = PoseStamped()
                    pose.header.stamp = stamp
                    pose.header.frame_id = "playback_frame"
                    pose.pose.orientation.w = 1.0
                    pose.pose.position.x = float(row.get("step", 0)) * 0.01
                    pose.pose.position.y = 0.0
                    pose.pose.position.z = 0.0
                    inner_self.pose_pub.publish(pose)
                elif topic == "/system_health":
                    DiagnosticArray = __import__("diagnostic_msgs.msg", fromlist=["DiagnosticArray"]).DiagnosticArray
                    DiagnosticStatus = __import__("diagnostic_msgs.msg", fromlist=["DiagnosticStatus"]).DiagnosticStatus
                    health = DiagnosticArray()
                    health.header.stamp = stamp
                    status = DiagnosticStatus()
                    status.level = DiagnosticStatus.OK
                    status.name = "playback_adapter"
                    status.message = inner_self.runtime_meta["runtime_type"]
                    health.status = [status]
                    inner_self.health_pub.publish(health)
                else:
                    inner_self.event_pub.publish(
                        json_string_msg(
                            {
                                "topic": topic,
                                "payload_note": row.get("payload_note", ""),
                                "runtime_type": inner_self.runtime_meta["runtime_type"],
                                "replay_mode": inner_self.runtime_meta["replay_mode"],
                            }
                        )
                    )

            def _load_or_placeholder_frame(inner_self) -> np.ndarray:
                try:
                    session_dir = resolve_latest_session(inner_self.project_root)
                    video_csv = pd.read_csv(session_dir / "video_frames.csv")
                    frame_path = Path(video_csv.iloc[0]["frame_path"])
                    frame = cv2.imread(str(frame_path))
                    if frame is not None:
                        return frame
                except Exception:
                    pass
                return np.zeros((480, 640, 3), dtype=np.uint8)

        self.node_cls = PlaybackAdapterNode


def main() -> None:
    runtime = PlaybackAdapterNodeBase()
    rclpy = runtime._rclpy
    rclpy.init()
    node = runtime.node_cls()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
