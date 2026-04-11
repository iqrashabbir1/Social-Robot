from __future__ import annotations

import csv
from pathlib import Path

from social_robot.config_helpers import nested_ros_params, resolve_ros_config
from social_robot.message_utils import decode_json_string
from social_robot.runtime import ensure_rclpy, normalize_path


class EventLoggerNodeBase:
    def __init__(self) -> None:
        self._rclpy, Node = ensure_rclpy()

        class EventLoggerNode(Node):
            def __init__(inner_self) -> None:
                super().__init__("event_logger_node")
                default_config = Path(__file__).resolve().parents[1] / "config" / "live_emotion_demo.yaml"
                inner_self.declare_parameter("config_path", str(default_config))
                inner_self.declare_parameter("output_dir", "")
                inner_self.declare_parameter("runtime_type", "")
                config = resolve_ros_config(inner_self.get_parameter("config_path").value, default_config)
                params = nested_ros_params(config, "event_logger_node")
                output_dir = str(inner_self.get_parameter("output_dir").value or params.get("output_dir", "outputs/logs/ros2_live"))
                inner_self.output_dir = normalize_path(output_dir)
                inner_self.output_dir.mkdir(parents=True, exist_ok=True)
                inner_self.csv_path = inner_self.output_dir / "ros2_event_log.csv"
                inner_self.system_health_csv_path = inner_self.output_dir / "ros2_system_health.csv"
                inner_self.evidence_label = str(params.get("evidence_label", "pilot_demonstration"))
                runtime_override = str(inner_self.get_parameter("runtime_type").value or "").strip()
                inner_self.runtime_type = str(runtime_override or params.get("runtime_type", "ros2_live_laptop_sensors"))

                inner_self._writer_handle = inner_self.csv_path.open("w", newline="", encoding="utf-8")
                inner_self._writer = csv.DictWriter(
                    inner_self._writer_handle,
                    fieldnames=[
                        "topic",
                        "received_time_sec",
                        "payload_excerpt",
                        "runtime_type",
                        "evidence_level",
                    ],
                )
                inner_self._writer.writeheader()
                inner_self._health_writer_handle = inner_self.system_health_csv_path.open("w", newline="", encoding="utf-8")
                inner_self._health_writer = csv.DictWriter(
                    inner_self._health_writer_handle,
                    fieldnames=[
                        "received_time_sec",
                        "runtime_type",
                        "evidence_level",
                        "status_name",
                        "status_message",
                        "cpu_percent",
                        "memory_percent",
                        "runtime_label",
                    ],
                )
                inner_self._health_writer.writeheader()

                std_msgs = __import__("std_msgs.msg", fromlist=["String"])
                sensor_msgs = __import__("sensor_msgs.msg", fromlist=["Image"])
                geometry_msgs = __import__("geometry_msgs.msg", fromlist=["PoseStamped"])
                diagnostic_msgs = __import__("diagnostic_msgs.msg", fromlist=["DiagnosticArray"])
                inner_self.create_subscription(sensor_msgs.Image, "/camera/image_raw", lambda msg: inner_self.log_row("/camera/image_raw", f"{msg.width}x{msg.height}"), 10)
                inner_self.create_subscription(std_msgs.Float32MultiArray, "/audio/stream", lambda msg: inner_self.log_row("/audio/stream", f"samples={len(msg.data)}"), 10)
                inner_self.create_subscription(geometry_msgs.PoseStamped, "/robot_pose", lambda msg: inner_self.log_row("/robot_pose", f"x={msg.pose.position.x:.3f},y={msg.pose.position.y:.3f}"), 10)
                inner_self.create_subscription(std_msgs.String, "/event_log", lambda msg: inner_self.log_row("/event_log", decode_json_string(msg.data)), 10)
                inner_self.create_subscription(std_msgs.String, "/emotion_state", lambda msg: inner_self.log_row("/emotion_state", decode_json_string(msg.data)), 10)
                inner_self.create_subscription(diagnostic_msgs.DiagnosticArray, "/system_health", inner_self.on_system_health, 10)

            def log_row(inner_self, topic: str, payload) -> None:
                inner_self._writer.writerow(
                    {
                        "topic": topic,
                        "received_time_sec": f"{inner_self.get_clock().now().nanoseconds / 1e9:.6f}",
                        "payload_excerpt": str(payload),
                        "runtime_type": inner_self.runtime_type,
                        "evidence_level": inner_self.evidence_label,
                    }
                )
                inner_self._writer_handle.flush()

            def on_system_health(inner_self, msg) -> None:
                primary_status = msg.status[0] if msg.status else None
                inner_self.log_row("/system_health", primary_status.message if primary_status else "no_status")
                if primary_status is None:
                    return
                values = {item.key: item.value for item in primary_status.values}
                inner_self._health_writer.writerow(
                    {
                        "received_time_sec": f"{inner_self.get_clock().now().nanoseconds / 1e9:.6f}",
                        "runtime_type": inner_self.runtime_type,
                        "evidence_level": inner_self.evidence_label,
                        "status_name": primary_status.name,
                        "status_message": primary_status.message,
                        "cpu_percent": values.get("cpu_percent", ""),
                        "memory_percent": values.get("memory_percent", ""),
                        "runtime_label": values.get("runtime_label", ""),
                    }
                )
                inner_self._health_writer_handle.flush()

            def destroy_node(inner_self) -> bool:
                inner_self._writer_handle.close()
                inner_self._health_writer_handle.close()
                return super().destroy_node()

        self.node_cls = EventLoggerNode


def main() -> None:
    runtime = EventLoggerNodeBase()
    rclpy = runtime._rclpy
    rclpy.init()
    node = runtime.node_cls()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
