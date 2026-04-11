from __future__ import annotations

from pathlib import Path

import numpy as np

from social_robot.config_helpers import nested_ros_params, resolve_ros_config
from social_robot.message_utils import float_audio_to_msg
from social_robot.runtime import ensure_rclpy


class AudioNodeBase:
    def __init__(self) -> None:
        self._rclpy, Node = ensure_rclpy()

        class AudioNode(Node):
            def __init__(inner_self) -> None:
                super().__init__("audio_node")
                default_config = Path(__file__).resolve().parents[1] / "config" / "live_sensing.yaml"
                inner_self.declare_parameter("config_path", str(default_config))
                config = resolve_ros_config(inner_self.get_parameter("config_path").value, default_config)
                params = nested_ros_params(config, "audio_node")

                inner_self.enabled = bool(params.get("enable_audio", True))
                inner_self.sample_rate = int(params.get("sample_rate", 16000))
                inner_self.chunk_size = int(params.get("chunk_size", 1600))
                inner_self.device_index = params.get("audio_device_index", None)
                inner_self.topic = str(params.get("audio_topic", "/audio/stream"))
                inner_self.health_topic = str(params.get("health_topic", "/system_health"))
                inner_self.device_available = False
                inner_self.audio_backend_ready = False
                inner_self.status_message = "audio_disabled"

                inner_self.publisher = inner_self.create_publisher(__import__("std_msgs.msg", fromlist=["Float32MultiArray"]).Float32MultiArray, inner_self.topic, 10)
                inner_self.health_publisher = inner_self.create_publisher(
                    __import__("diagnostic_msgs.msg", fromlist=["DiagnosticArray"]).DiagnosticArray,
                    inner_self.health_topic,
                    10,
                )

                if not inner_self.enabled:
                    inner_self.get_logger().info("Audio node is disabled by configuration; no audio stream will be published.")
                    inner_self.sd = None
                    inner_self.timer = None
                else:
                    inner_self.sd = inner_self._load_sounddevice()
                    if inner_self.sd is not None and inner_self._detect_input_device():
                        inner_self.audio_backend_ready = True
                        inner_self.status_message = "audio_ready"
                        timer_period = max(0.05, inner_self.chunk_size / max(inner_self.sample_rate, 1))
                        inner_self.timer = inner_self.create_timer(timer_period, inner_self.publish_audio)
                    else:
                        inner_self.timer = None
                inner_self.health_timer = inner_self.create_timer(2.0, inner_self.publish_status)

            def _load_sounddevice(inner_self):
                try:
                    import sounddevice as sounddevice

                    return sounddevice
                except Exception as exc:
                    inner_self.status_message = "sounddevice_unavailable"
                    inner_self.get_logger().warning(f"sounddevice is unavailable; audio capture disabled: {exc}")
                    return None

            def _detect_input_device(inner_self) -> bool:
                if inner_self.sd is None:
                    return False
                try:
                    devices = inner_self.sd.query_devices()
                    has_input = any(int(device.get("max_input_channels", 0)) > 0 for device in devices)
                    if not has_input:
                        inner_self.status_message = "no_audio_input_device"
                        inner_self.get_logger().warning("No input audio device detected; audio capture disabled.")
                        return False
                    inner_self.device_available = True
                    return True
                except Exception as exc:
                    inner_self.status_message = "audio_device_query_failed"
                    inner_self.get_logger().warning(f"Audio device query failed; audio capture disabled: {exc}")
                    return False

            def publish_audio(inner_self) -> None:
                if not inner_self.enabled or inner_self.sd is None or not inner_self.audio_backend_ready:
                    return
                try:
                    clip = inner_self.sd.rec(
                        inner_self.chunk_size,
                        samplerate=inner_self.sample_rate,
                        channels=1,
                        dtype="float32",
                        device=inner_self.device_index,
                    )
                    inner_self.sd.wait()
                    audio = np.asarray(clip).reshape(-1)
                    inner_self.publisher.publish(float_audio_to_msg(audio))
                    inner_self.device_available = True
                    inner_self.status_message = "audio_streaming"
                except Exception as exc:
                    inner_self.device_available = False
                    inner_self.status_message = "audio_capture_failed"
                    inner_self.get_logger().warning(f"Audio capture failed: {exc}")

            def publish_status(inner_self) -> None:
                DiagnosticArray = __import__("diagnostic_msgs.msg", fromlist=["DiagnosticArray"]).DiagnosticArray
                DiagnosticStatus = __import__("diagnostic_msgs.msg", fromlist=["DiagnosticStatus"]).DiagnosticStatus
                KeyValue = __import__("diagnostic_msgs.msg", fromlist=["KeyValue"]).KeyValue

                if not inner_self.enabled:
                    level = DiagnosticStatus.OK
                else:
                    level = DiagnosticStatus.OK if inner_self.device_available and inner_self.audio_backend_ready else DiagnosticStatus.WARN

                status = DiagnosticStatus()
                status.name = "social_robot/audio"
                status.hardware_id = "microphone_input"
                status.level = level
                status.message = inner_self.status_message
                status.values = [
                    KeyValue(key="audio_topic", value=inner_self.topic),
                    KeyValue(key="sample_rate", value=str(inner_self.sample_rate)),
                    KeyValue(key="chunk_size", value=str(inner_self.chunk_size)),
                    KeyValue(key="enabled", value=str(inner_self.enabled).lower()),
                ]

                message = DiagnosticArray()
                message.header.stamp = inner_self.get_clock().now().to_msg()
                message.status = [status]
                inner_self.health_publisher.publish(message)

        self.node_cls = AudioNode


def main() -> None:
    runtime = AudioNodeBase()
    rclpy = runtime._rclpy
    rclpy.init()
    node = runtime.node_cls()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
