from __future__ import annotations

import socket
import struct
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from social_robot.config_helpers import nested_ros_params, resolve_ros_config
from social_robot.runtime import ensure_rclpy


class WindowsCameraBridgeNodeBase:
    def __init__(self) -> None:
        self._rclpy, Node = ensure_rclpy()

        class WindowsCameraBridgeNode(Node):
            def __init__(inner_self) -> None:
                super().__init__("windows_camera_bridge_node")
                default_config = Path(__file__).resolve().parents[1] / "config" / "live_sensing.yaml"
                inner_self.declare_parameter("config_path", str(default_config))
                inner_self.declare_parameter("windows_camera_host", "")
                inner_self.declare_parameter("windows_camera_port", 0)
                inner_self.declare_parameter("runtime_type", "")
                config = resolve_ros_config(inner_self.get_parameter("config_path").value, default_config)
                params = nested_ros_params(config, "windows_camera_bridge_node")

                host_override = str(inner_self.get_parameter("windows_camera_host").value or "").strip()
                port_override = int(inner_self.get_parameter("windows_camera_port").value or 0)
                runtime_override = str(inner_self.get_parameter("runtime_type").value or "").strip()

                inner_self.host = str(host_override or params.get("windows_camera_host", "127.0.0.1"))
                inner_self.port = int(port_override or params.get("windows_camera_port", 5001))
                inner_self.topic = str(params.get("camera_topic", "/camera/image_raw"))
                inner_self.health_topic = str(params.get("health_topic", "/system_health"))
                inner_self.retry_delay_seconds = float(params.get("retry_delay_seconds", 2.0))
                inner_self.receive_timeout_seconds = float(params.get("receive_timeout_seconds", 5.0))
                inner_self.runtime_type = str(runtime_override or params.get("runtime_type", "ros2_live_windows_stream_wsl_core"))
                inner_self.stream_available = False
                inner_self.last_warning_time = 0.0
                inner_self.last_frame_shape = "no_frame"
                inner_self._stop_event = threading.Event()

                sensor_msgs = __import__("sensor_msgs.msg", fromlist=["Image"])
                diagnostic_msgs = __import__("diagnostic_msgs.msg", fromlist=["DiagnosticArray"])
                inner_self.image_pub = inner_self.create_publisher(sensor_msgs.Image, inner_self.topic, 10)
                inner_self.health_pub = inner_self.create_publisher(diagnostic_msgs.DiagnosticArray, inner_self.health_topic, 10)

                try:
                    from cv_bridge import CvBridge

                    inner_self.bridge = CvBridge()
                except Exception as exc:
                    raise RuntimeError("cv_bridge is required on the WSL ROS 2 side for windows_camera_bridge_node.") from exc

                inner_self.worker = threading.Thread(target=inner_self.stream_loop, daemon=True)
                inner_self.worker.start()
                inner_self.health_timer = inner_self.create_timer(2.0, inner_self.publish_status)

            def _recv_exact(inner_self, conn: socket.socket, length: int) -> bytes | None:
                data = bytearray()
                while len(data) < length and not inner_self._stop_event.is_set():
                    try:
                        chunk = conn.recv(length - len(data))
                    except socket.timeout:
                        return None
                    if not chunk:
                        return None
                    data.extend(chunk)
                return bytes(data)

            def stream_loop(inner_self) -> None:
                while not inner_self._stop_event.is_set():
                    try:
                        with socket.create_connection((inner_self.host, inner_self.port), timeout=5.0) as conn:
                            conn.settimeout(inner_self.receive_timeout_seconds)
                            inner_self.stream_available = True
                            inner_self.get_logger().info(f"Connected to Windows camera streamer at {inner_self.host}:{inner_self.port}.")
                            while not inner_self._stop_event.is_set():
                                header = inner_self._recv_exact(conn, 4)
                                if not header:
                                    raise ConnectionError("Stream header not received.")
                                payload_length = struct.unpack("!I", header)[0]
                                payload = inner_self._recv_exact(conn, payload_length)
                                if not payload:
                                    raise ConnectionError("Frame payload not received.")

                                encoded = np.frombuffer(payload, dtype=np.uint8)
                                frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                                if frame is None:
                                    inner_self.get_logger().warning("Received an undecodable JPEG frame from the Windows streamer.")
                                    continue

                                try:
                                    message = inner_self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
                                    message.header.stamp = inner_self.get_clock().now().to_msg()
                                    message.header.frame_id = "windows_stream_bridge"
                                    inner_self.image_pub.publish(message)
                                    inner_self.stream_available = True
                                    inner_self.last_frame_shape = f"{frame.shape[1]}x{frame.shape[0]}"
                                except Exception as exc:
                                    inner_self.get_logger().warning(f"Failed to publish bridged camera frame: {exc}")
                    except Exception as exc:
                        inner_self.stream_available = False
                        now = time.monotonic()
                        if now - inner_self.last_warning_time >= inner_self.retry_delay_seconds:
                            inner_self.get_logger().warning(
                                f"Windows camera stream unavailable at {inner_self.host}:{inner_self.port}; retrying: {exc}"
                            )
                            inner_self.last_warning_time = now
                        time.sleep(inner_self.retry_delay_seconds)

            def publish_status(inner_self) -> None:
                DiagnosticArray = __import__("diagnostic_msgs.msg", fromlist=["DiagnosticArray"]).DiagnosticArray
                DiagnosticStatus = __import__("diagnostic_msgs.msg", fromlist=["DiagnosticStatus"]).DiagnosticStatus
                KeyValue = __import__("diagnostic_msgs.msg", fromlist=["KeyValue"]).KeyValue

                status = DiagnosticStatus()
                status.name = "social_robot/windows_camera_bridge"
                status.hardware_id = f"{inner_self.host}:{inner_self.port}"
                status.level = DiagnosticStatus.OK if inner_self.stream_available else DiagnosticStatus.WARN
                status.message = "windows_stream_connected" if inner_self.stream_available else "windows_stream_unavailable"
                status.values = [
                    KeyValue(key="camera_topic", value=inner_self.topic),
                    KeyValue(key="runtime_type", value=inner_self.runtime_type),
                    KeyValue(key="last_frame_shape", value=inner_self.last_frame_shape),
                ]

                message = DiagnosticArray()
                message.header.stamp = inner_self.get_clock().now().to_msg()
                message.status = [status]
                inner_self.health_pub.publish(message)

            def destroy_node(inner_self) -> bool:
                inner_self._stop_event.set()
                if hasattr(inner_self, "worker") and inner_self.worker.is_alive():
                    inner_self.worker.join(timeout=1.0)
                return super().destroy_node()

        self.node_cls = WindowsCameraBridgeNode


def main() -> None:
    runtime = WindowsCameraBridgeNodeBase()
    rclpy = runtime._rclpy
    rclpy.init()
    node = runtime.node_cls()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
