from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


@dataclass(frozen=True)
class TopicSpec:
    topic: str
    message_type: str
    producer: str
    consumer: str
    description: str


def default_interface_spec() -> list[TopicSpec]:
    return [
        TopicSpec("/camera/image_raw", "sensor_msgs/Image", "camera_node or playback_adapter_node", "digital_twin_node / emotion_inference_node / event_logger_node", "Primary visual sensing stream from webcam, simulator, or playback source."),
        TopicSpec("/audio/stream", "std_msgs/Float32MultiArray", "audio_node or playback_adapter_node", "emotion_inference_node / digital_twin_node / event_logger_node", "Primary audio chunk stream using a standard ROS2 message with float samples."),
        TopicSpec("/robot_pose", "geometry_msgs/PoseStamped", "robot_state_node or simulator publisher", "digital_twin_node / emotion_inference_node / event_logger_node", "Robot pose or placeholder laptop/demo context state."),
        TopicSpec("/head_cmd", "std_msgs/String", "digital_twin_node or future controller", "robot head controller / logger", "Head orientation or gesture command."),
        TopicSpec("/speech_cmd", "std_msgs/String", "digital_twin_node or future dialogue policy", "robot TTS controller / logger", "Speech command text and intent."),
        TopicSpec("/event_log", "std_msgs/String", "digital_twin_node / playback_adapter_node", "event_logger_node / analysis pipeline", "Structured experiment event and synchronization log."),
        TopicSpec("/system_health", "diagnostic_msgs/DiagnosticArray", "robot_state_node", "digital_twin_node / event_logger_node", "CPU, memory, and subsystem health status."),
        TopicSpec("/emotion_state", "std_msgs/String", "emotion_inference_node", "digital_twin_node / event_logger_node", "Emotion inference output encoded as JSON."),
    ]


def interface_spec_dataframe() -> pd.DataFrame:
    return pd.DataFrame(asdict(item) for item in default_interface_spec())
