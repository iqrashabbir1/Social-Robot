from __future__ import annotations

from pathlib import Path

from social_robot.config_helpers import nested_ros_params, resolve_ros_config
from social_robot.message_utils import decode_json_string, json_string_msg, now_ms
from social_robot.runtime import ensure_rclpy
from src.digital_twin.twin_state import DigitalTwinState


class DigitalTwinNodeBase:
    def __init__(self) -> None:
        self._rclpy, Node = ensure_rclpy()

        class DigitalTwinNode(Node):
            def __init__(inner_self) -> None:
                super().__init__("digital_twin_node")
                default_config = Path(__file__).resolve().parents[1] / "config" / "live_emotion_demo.yaml"
                inner_self.declare_parameter("config_path", str(default_config))
                inner_self.declare_parameter("runtime_type", "")
                config = resolve_ros_config(inner_self.get_parameter("config_path").value, default_config)
                params = nested_ros_params(config, "digital_twin_node")

                inner_self.sync_publish_rate = float(params.get("sync_publish_rate", 2.0))
                inner_self.event_topic = str(params.get("event_topic", "/event_log"))
                inner_self.head_cmd_topic = str(params.get("head_cmd_topic", "/head_cmd"))
                inner_self.speech_cmd_topic = str(params.get("speech_cmd_topic", "/speech_cmd"))
                runtime_override = str(inner_self.get_parameter("runtime_type").value or "").strip()
                inner_self.runtime_type = str(runtime_override or params.get("runtime_type", "ros2_live_laptop_sensors"))

                std_msgs = __import__("std_msgs.msg", fromlist=["String"])
                sensor_msgs = __import__("sensor_msgs.msg", fromlist=["Image"])
                geometry_msgs = __import__("geometry_msgs.msg", fromlist=["PoseStamped"])
                diagnostic_msgs = __import__("diagnostic_msgs.msg", fromlist=["DiagnosticArray"])

                inner_self.state = DigitalTwinState()
                inner_self.latest_emotion = "unknown"
                inner_self.event_pub = inner_self.create_publisher(std_msgs.String, inner_self.event_topic, 10)
                inner_self.head_pub = inner_self.create_publisher(std_msgs.String, inner_self.head_cmd_topic, 10)
                inner_self.speech_pub = inner_self.create_publisher(std_msgs.String, inner_self.speech_cmd_topic, 10)
                inner_self.string_cls = std_msgs.String

                inner_self.create_subscription(sensor_msgs.Image, "/camera/image_raw", inner_self.on_image, 10)
                inner_self.create_subscription(std_msgs.Float32MultiArray, "/audio/stream", inner_self.on_audio, 10)
                inner_self.create_subscription(geometry_msgs.PoseStamped, "/robot_pose", inner_self.on_pose, 10)
                inner_self.create_subscription(diagnostic_msgs.DiagnosticArray, "/system_health", inner_self.on_health, 10)
                inner_self.create_subscription(std_msgs.String, "/emotion_state", inner_self.on_emotion, 10)

                inner_self.timer = inner_self.create_timer(max(0.2, 1.0 / max(inner_self.sync_publish_rate, 0.1)), inner_self.publish_sync_state)

            def on_image(inner_self, msg) -> None:
                timestamp_ms = now_ms()
                inner_self.state.update("/camera/image_raw", timestamp_ms, {"payload_note": f"{msg.width}x{msg.height}"})

            def on_audio(inner_self, msg) -> None:
                inner_self.state.update("/audio/stream", now_ms(), {"payload_note": f"samples={len(msg.data)}"})

            def on_pose(inner_self, msg) -> None:
                inner_self.state.update(
                    "/robot_pose",
                    now_ms(),
                    {"payload_note": f"x={msg.pose.position.x:.3f},y={msg.pose.position.y:.3f}"},
                )

            def on_health(inner_self, msg) -> None:
                health_state = "nominal"
                if msg.status:
                    health_state = msg.status[0].message or "nominal"
                inner_self.state.update("/system_health", now_ms(), {"health_state": health_state})

            def on_emotion(inner_self, msg) -> None:
                payload = decode_json_string(msg.data)
                inner_self.latest_emotion = str(payload.get("fused_emotion", payload.get("emotion", "unknown")))

            def publish_sync_state(inner_self) -> None:
                sync_error = inner_self.state.synchronization_error_ms()
                event_payload = {
                    "topic": "/event_log",
                    "runtime_type": inner_self.runtime_type,
                    "sync_error_ms": round(sync_error, 4),
                    "event_counter": inner_self.state.event_counter,
                    "health_state": inner_self.state.health_state,
                    "latest_emotion": inner_self.latest_emotion,
                    "timestamp_ms": round(now_ms(), 4),
                }
                inner_self.event_pub.publish(json_string_msg(event_payload))

                head_cmd = inner_self.string_cls()
                head_cmd.data = "track_face" if sync_error < 200.0 else "hold_pose"
                speech_cmd = inner_self.string_cls()
                speech_cmd.data = "empathetic_prompt" if inner_self.latest_emotion in {"sad", "fear", "angry"} else "neutral_prompt"
                inner_self.head_pub.publish(head_cmd)
                inner_self.speech_pub.publish(speech_cmd)

        self.node_cls = DigitalTwinNode


def main() -> None:
    runtime = DigitalTwinNodeBase()
    rclpy = runtime._rclpy
    rclpy.init()
    node = runtime.node_cls()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
