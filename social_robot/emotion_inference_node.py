from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from social_robot.config_helpers import nested_ros_params, resolve_ros_config
from social_robot.message_utils import decode_json_string, json_string_msg, msg_to_float_audio, now_ms, rosmsg_to_image
from social_robot.runtime import ensure_rclpy


class EmotionInferenceNodeBase:
    def __init__(self) -> None:
        self._rclpy, Node = ensure_rclpy()

        class EmotionInferenceNode(Node):
            def __init__(inner_self) -> None:
                super().__init__("emotion_inference_node")
                default_config = Path(__file__).resolve().parents[1] / "config" / "live_emotion_demo.yaml"
                inner_self.declare_parameter("config_path", str(default_config))
                inner_self.declare_parameter("runtime_type", "")
                config = resolve_ros_config(inner_self.get_parameter("config_path").value, default_config)
                params = nested_ros_params(config, "emotion_inference_node")

                inner_self.enable_audio = bool(params.get("enable_audio", True))
                inner_self.publish_rate = float(params.get("publish_rate", 1.0))
                inner_self.audio_sample_rate = int(params.get("audio_sample_rate", 16000))
                runtime_override = str(inner_self.get_parameter("runtime_type").value or "").strip()
                inner_self.runtime_type = str(runtime_override or params.get("runtime_type", "ros2_live_laptop_sensors"))
                inner_self.latest_frame = None
                inner_self.latest_audio = None
                inner_self.latest_context = {}
                inner_self.face_backend_ok = None
                inner_self.speech_backend_ok = None

                std_msgs = __import__("std_msgs.msg", fromlist=["String"])
                sensor_msgs = __import__("sensor_msgs.msg", fromlist=["Image"])
                geometry_msgs = __import__("geometry_msgs.msg", fromlist=["PoseStamped"])
                inner_self.publisher = inner_self.create_publisher(std_msgs.String, "/emotion_state", 10)
                inner_self.create_subscription(sensor_msgs.Image, "/camera/image_raw", inner_self.on_image, 10)
                inner_self.create_subscription(std_msgs.Float32MultiArray, "/audio/stream", inner_self.on_audio, 10)
                inner_self.create_subscription(geometry_msgs.PoseStamped, "/robot_pose", inner_self.on_pose, 10)
                inner_self.timer = inner_self.create_timer(max(0.2, 1.0 / max(inner_self.publish_rate, 0.1)), inner_self.run_inference)

            def on_image(inner_self, msg) -> None:
                inner_self.latest_frame = rosmsg_to_image(msg)

            def on_audio(inner_self, msg) -> None:
                inner_self.latest_audio = msg_to_float_audio(msg)

            def on_pose(inner_self, msg) -> None:
                inner_self.latest_context = {
                    "x": float(msg.pose.position.x),
                    "y": float(msg.pose.position.y),
                    "z": float(msg.pose.position.z),
                }

            def run_inference(inner_self) -> None:
                from fusion.fusion_logic import fuse_emotions, normalize_face_emotion

                face_label = None
                face_conf = None
                speech_label = None

                if inner_self.latest_frame is not None:
                    try:
                        from perception.face_emotion import detect_face_emotion_from_frame

                        if inner_self.face_backend_ok is not True:
                            inner_self.get_logger().info("Face inference backend available.")
                            inner_self.face_backend_ok = True
                        raw_face, face_conf = detect_face_emotion_from_frame(inner_self.latest_frame, enforce_detection=False)
                        face_label = normalize_face_emotion(raw_face)
                    except Exception as exc:
                        if inner_self.face_backend_ok is not False:
                            inner_self.get_logger().warning(
                                "Face inference backend unavailable or failed. "
                                f"Video emotion output will be disabled until dependencies are installed. Detail: {exc}"
                            )
                            inner_self.face_backend_ok = False

                if inner_self.enable_audio and inner_self.latest_audio is not None and inner_self.latest_audio.size > 0:
                    try:
                        import soundfile as sf
                        from perception.speech_emotion import predict_speech_emotion_robust_3class

                        if inner_self.speech_backend_ok is not True:
                            inner_self.get_logger().info("Speech inference backend available.")
                            inner_self.speech_backend_ok = True
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
                            temp_path = Path(handle.name)
                        sf.write(temp_path, np.asarray(inner_self.latest_audio), inner_self.audio_sample_rate)
                        speech_label = predict_speech_emotion_robust_3class(str(temp_path))
                        temp_path.unlink(missing_ok=True)
                    except Exception as exc:
                        if inner_self.speech_backend_ok is not False:
                            inner_self.get_logger().warning(
                                "Speech inference backend unavailable or failed. "
                                f"Audio emotion output will be disabled until dependencies are installed. Detail: {exc}"
                            )
                            inner_self.speech_backend_ok = False

                fused = fuse_emotions(face_label, speech_label)
                payload = {
                    "timestamp_ms": round(now_ms(), 4),
                    "face_emotion": face_label,
                    "face_confidence": face_conf,
                    "speech_emotion": speech_label,
                    "fused_emotion": fused,
                    "context_state": inner_self.latest_context,
                    "runtime_type": inner_self.runtime_type,
                    "evidence_level": "pilot_demonstration",
                }
                inner_self.publisher.publish(json_string_msg(payload))

        self.node_cls = EmotionInferenceNode


def main() -> None:
    runtime = EmotionInferenceNodeBase()
    rclpy = runtime._rclpy
    rclpy.init()
    node = runtime.node_cls()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
