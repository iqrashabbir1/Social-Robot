from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DigitalTwinState:
    last_camera_timestamp_ms: float = 0.0
    last_audio_timestamp_ms: float = 0.0
    last_robot_pose_timestamp_ms: float = 0.0
    latest_head_command: str = "idle"
    latest_speech_command: str = "idle"
    health_state: str = "nominal"
    event_counter: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)

    def update(self, topic: str, timestamp_ms: float, payload: dict[str, Any] | None = None) -> None:
        payload = payload or {}
        self.event_counter += 1
        if topic == "/camera/image_raw":
            self.last_camera_timestamp_ms = timestamp_ms
        elif topic == "/audio/stream":
            self.last_audio_timestamp_ms = timestamp_ms
        elif topic == "/robot_pose":
            self.last_robot_pose_timestamp_ms = timestamp_ms
        elif topic == "/head_cmd":
            self.latest_head_command = str(payload.get("command", self.latest_head_command))
        elif topic == "/speech_cmd":
            self.latest_speech_command = str(payload.get("command", self.latest_speech_command))
        elif topic == "/system_health":
            self.health_state = str(payload.get("health_state", self.health_state))

        self.history.append(
            {
                "event_id": self.event_counter,
                "topic": topic,
                "timestamp_ms": timestamp_ms,
                "health_state": self.health_state,
                "latest_head_command": self.latest_head_command,
                "latest_speech_command": self.latest_speech_command,
            }
        )

    def synchronization_error_ms(self) -> float:
        timestamps = [
            self.last_camera_timestamp_ms,
            self.last_audio_timestamp_ms,
            self.last_robot_pose_timestamp_ms,
        ]
        valid = [value for value in timestamps if value > 0.0]
        if len(valid) < 2:
            return 0.0
        return max(valid) - min(valid)
