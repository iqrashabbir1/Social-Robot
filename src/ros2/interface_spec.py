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
        TopicSpec("/camera/image_raw", "sensor_msgs/Image", "camera simulator", "sync pipeline", "Primary visual sensing stream."),
        TopicSpec("/audio/stream", "audio_common_msgs/AudioData", "microphone simulator", "sync pipeline", "Primary audio sensing stream."),
        TopicSpec("/robot_pose", "geometry_msgs/PoseStamped", "robot state estimator", "digital twin", "Robot pose and context stream."),
        TopicSpec("/head_cmd", "std_msgs/String", "control policy", "robot head controller", "Head orientation or gesture command."),
        TopicSpec("/speech_cmd", "std_msgs/String", "dialogue policy", "robot TTS controller", "Speech command text and intent."),
        TopicSpec("/event_log", "std_msgs/String", "event logger", "analysis pipeline", "Structured experiment event log."),
        TopicSpec("/system_health", "diagnostic_msgs/DiagnosticArray", "runtime monitor", "analysis pipeline", "CPU, memory, and subsystem health status."),
    ]


def interface_spec_dataframe() -> pd.DataFrame:
    return pd.DataFrame(asdict(item) for item in default_interface_spec())
