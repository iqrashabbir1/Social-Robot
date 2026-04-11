from __future__ import annotations

import socket
import struct
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from social_robot.config_helpers import nested_ros_params, resolve_ros_config
from social_robot.message_utils import image_to_rosmsg, resize_frame
from social_robot.runtime import ensure_rclpy


class CameraNodeBase:
    def __init__(self) -> None:
        self._rclpy, Node = ensure_rclpy()

        class CameraNode(Node):
            def __init__(inner_self) -> None:
                super().__init__("camera_node")
                default_config = Path(__file__).resolve().parents[1] / "config" / "live_sensing.yaml"
                inner_self.declare_parameter("config_path", str(default_config))
                inner_self.declare_parameter("camera_input_mode", "")
                inner_self.declare_parameter("windows_camera_host", "")
                inner_self.declare_parameter("windows_camera_port", 0)
                config = resolve_ros_config(inner_self.get_parameter("config_path").value, default_config)
                params = nested_ros_params(config, "camera_node")

                camera_input_override = str(inner_self.get_parameter("camera_input_mode").value or "").strip()
                inner_self.camera_input_mode = str(camera_input_override or params.get("camera_input_mode", "ros_topic")).strip().lower()
                inner_self.camera_index = int(params.get("camera_index", 0))
                inner_self.width = int(params.get("width", 640))
                inner_self.height = int(params.get("height", 480))
                inner_self.frame_rate = float(params.get("frame_rate", 10.0))
                inner_self.topic = str(params.get("camera_topic", params.get("image_topic", "/camera/image_raw")))
                inner_self.health_topic = str(params.get("health_topic", "/system_health"))
                host_override = str(inner_self.get_parameter("windows_camera_host").value or "").strip()
                port_override = int(inner_self.get_parameter("windows_camera_port").value or 0)
                inner_self.windows_camera_host = str(host_override or params.get("windows_camera_host", "127.0.0.1"))
                inner_self.windows_camera_port = int(port_override or params.get("windows_camera_port", 5001))
                inner_self.retry_delay_seconds = float(params.get("retry_delay_seconds", 2.0))
                inner_self.receive_timeout_seconds = float(params.get("receive_timeout_seconds", 5.0))
                inner_self.device_available = False
                inner_self.last_frame_time = 0.0
                inner_self.last_warning_time = 0.0
                inner_self.warning_interval_seconds = float(params.get("missing_frame_warning_interval_seconds", 5.0))
                inner_self.cap = None
                inner_self.last_frame_shape = "no_frame"
                inner_self._stop_event = threading.Event()
                inner_self.stream_thread = None

                inner_self.health_publisher = inner_self.create_publisher(
                    __import__("diagnostic_msgs.msg", fromlist=["DiagnosticArray"]).DiagnosticArray,
                    inner_self.health_topic,
                    10,
                )
                sensor_msgs = __import__("sensor_msgs.msg", fromlist=["Image"])
                inner_self.image_cls = sensor_msgs.Image

                if inner_self.camera_input_mode == "local_camera":
                    inner_self.get_logger().info(f"camera_node starting in local_camera mode, camera_index={inner_self.camera_index}")
                    inner_self.publisher = inner_self.create_publisher(inner_self.image_cls, inner_self.topic, 10)
                    inner_self.cap = cv2.VideoCapture(inner_self.camera_index)
                    inner_self.device_available = bool(inner_self.cap.isOpened())
                    if not inner_self.device_available:
                        inner_self.get_logger().warning(f"Webcam index {inner_self.camera_index} could not be opened.")
                    timer_period = max(0.05, 1.0 / max(inner_self.frame_rate, 0.1))
                    inner_self.timer = inner_self.create_timer(timer_period, inner_self.publish_frame)
                elif inner_self.camera_input_mode == "ros_topic":
                    inner_self.get_logger().info(f"camera_node starting in ros_topic mode, topic={inner_self.topic}")
                    inner_self.publisher = None
                    inner_self.create_subscription(inner_self.image_cls, inner_self.topic, inner_self.on_image, 10)
                    inner_self.timer = inner_self.create_timer(1.0, inner_self.monitor_frame_stream)
                elif inner_self.camera_input_mode == "windows_stream_bridge":
                    inner_self.get_logger().info(
                        f"camera_node starting in windows_stream_bridge mode, host={inner_self.windows_camera_host}, port={inner_self.windows_camera_port}"
                    )
                    inner_self.publisher = inner_self.create_publisher(inner_self.image_cls, inner_self.topic, 10)
                    try:
                        from cv_bridge import CvBridge

                        inner_self.bridge = CvBridge()
                    except Exception as exc:
                        raise RuntimeError("cv_bridge is required in windows_stream_bridge mode.") from exc
                    inner_self.stream_thread = threading.Thread(target=inner_self.stream_loop, daemon=True)
                    inner_self.stream_thread.start()
                    inner_self.timer = inner_self.create_timer(1.0, inner_self.monitor_frame_stream)
                else:
                    raise ValueError(
                        f"Unsupported camera_input_mode '{inner_self.camera_input_mode}'. "
                        "Expected one of: local_camera, ros_topic, windows_stream_bridge."
                    )
                inner_self.health_timer = inner_self.create_timer(2.0, inner_self.publish_status)

            def on_image(inner_self, msg) -> None:
                inner_self.last_frame_time = time.monotonic()
                inner_self.device_available = True
                inner_self.last_frame_shape = f"{msg.width}x{msg.height}"

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
                        with socket.create_connection((inner_self.windows_camera_host, inner_self.windows_camera_port), timeout=5.0) as conn:
                            conn.settimeout(inner_self.receive_timeout_seconds)
                            inner_self.device_available = True
                            inner_self.get_logger().info(
                                f"Connected to Windows camera streamer at {inner_self.windows_camera_host}:{inner_self.windows_camera_port}."
                            )
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

                                frame = resize_frame(frame, inner_self.width, inner_self.height)
                                stamp = inner_self.get_clock().now().to_msg()
                                message = inner_self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
                                message.header.stamp = stamp
                                message.header.frame_id = "windows_stream_bridge"
                                inner_self.publisher.publish(message)
                                inner_self.last_frame_time = time.monotonic()
                                inner_self.device_available = True
                                inner_self.last_frame_shape = f"{frame.shape[1]}x{frame.shape[0]}"
                    except Exception as exc:
                        inner_self.device_available = False
                        now = time.monotonic()
                        if now - inner_self.last_warning_time >= inner_self.retry_delay_seconds:
                            inner_self.get_logger().warning(
                                f"Windows camera stream unavailable at {inner_self.windows_camera_host}:{inner_self.windows_camera_port}; retrying: {exc}"
                            )
                            inner_self.last_warning_time = now
                        time.sleep(inner_self.retry_delay_seconds)

            def publish_frame(inner_self) -> None:
                if inner_self.cap is None or not inner_self.cap.isOpened():
                    inner_self.device_available = False
                    return
                ok, frame = inner_self.cap.read()
                if not ok or frame is None:
                    inner_self.device_available = False
                    inner_self.get_logger().warning("Camera frame capture failed.")
                    return
                inner_self.device_available = True
                inner_self.last_frame_time = time.monotonic()
                inner_self.last_frame_shape = f"{frame.shape[1]}x{frame.shape[0]}"
                frame = resize_frame(frame, inner_self.width, inner_self.height)
                stamp = inner_self.get_clock().now().to_msg()
                inner_self.publisher.publish(image_to_rosmsg(frame, stamp))

            def monitor_frame_stream(inner_self) -> None:
                if inner_self.camera_input_mode not in {"ros_topic", "windows_stream_bridge"}:
                    return
                now = time.monotonic()
                if inner_self.last_frame_time == 0.0:
                    inner_self.device_available = False
                    if now - inner_self.last_warning_time >= inner_self.warning_interval_seconds:
                        wait_hint = (
                            "waiting for the Windows stream bridge."
                            if inner_self.camera_input_mode == "windows_stream_bridge"
                            else "waiting for an upstream ROS image publisher."
                        )
                        inner_self.get_logger().warning(f"No frames received yet on {inner_self.topic}; {wait_hint}")
                        inner_self.last_warning_time = now
                    return
                silence_duration = now - inner_self.last_frame_time
                if silence_duration > max(2.0, inner_self.warning_interval_seconds):
                    inner_self.device_available = False
                    if now - inner_self.last_warning_time >= inner_self.warning_interval_seconds:
                        inner_self.get_logger().warning(f"No camera frames received on {inner_self.topic} for {silence_duration:.1f}s.")
                        inner_self.last_warning_time = now

            def publish_status(inner_self) -> None:
                DiagnosticArray = __import__("diagnostic_msgs.msg", fromlist=["DiagnosticArray"]).DiagnosticArray
                DiagnosticStatus = __import__("diagnostic_msgs.msg", fromlist=["DiagnosticStatus"]).DiagnosticStatus
                KeyValue = __import__("diagnostic_msgs.msg", fromlist=["KeyValue"]).KeyValue

                status = DiagnosticStatus()
                status.name = "social_robot/camera"
                status.hardware_id = f"camera_{inner_self.camera_index}" if inner_self.camera_input_mode == "local_camera" else "camera_topic_monitor"
                status.level = DiagnosticStatus.OK if inner_self.device_available else DiagnosticStatus.WARN
                status.message = "camera_available" if inner_self.device_available else "camera_unavailable"
                status.values = [
                    KeyValue(key="image_topic", value=inner_self.topic),
                    KeyValue(key="camera_input_mode", value=inner_self.camera_input_mode),
                    KeyValue(key="camera_index", value=str(inner_self.camera_index)),
                    KeyValue(key="windows_camera_host", value=str(inner_self.windows_camera_host)),
                    KeyValue(key="windows_camera_port", value=str(inner_self.windows_camera_port)),
                    KeyValue(key="last_frame_shape", value=inner_self.last_frame_shape),
                ]

                message = DiagnosticArray()
                message.header.stamp = inner_self.get_clock().now().to_msg()
                message.status = [status]
                inner_self.health_publisher.publish(message)

            def destroy_node(inner_self) -> bool:
                inner_self._stop_event.set()
                if inner_self.stream_thread is not None and inner_self.stream_thread.is_alive():
                    inner_self.stream_thread.join(timeout=1.0)
                if inner_self.cap is not None:
                    inner_self.cap.release()
                return super().destroy_node()

        self.node_cls = CameraNode


def main() -> None:
    runtime = CameraNodeBase()
    rclpy = runtime._rclpy
    rclpy.init()
    node = runtime.node_cls()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
