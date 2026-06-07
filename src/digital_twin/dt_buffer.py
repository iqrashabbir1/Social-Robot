from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _stable_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _hash_to_unit_interval(payload: dict[str, Any], salt: str) -> float:
    digest = hashlib.sha256(f"{salt}:{json.dumps(payload, sort_keys=True, default=str)}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


@dataclass
class DTStateBuffer:
    max_history: int = 5000
    sync_threshold_ms: float = 500.0
    feature_dim: int = 384
    last_camera_timestamp_ms: float = 0.0
    last_audio_timestamp_ms: float = 0.0
    last_robot_pose_timestamp_ms: float = 0.0
    latest_head_command: str = "idle"
    latest_speech_command: str = "idle"
    health_state: str = "nominal"
    event_counter: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)

    def _topic_to_modality(self, topic: str) -> str:
        if topic == "/camera/image_raw":
            return "visual"
        if topic == "/audio/stream":
            return "audio"
        if topic in {"/robot_pose", "/physio/stream"}:
            return "physio"
        return "context"

    def _payload_to_feature_vector(self, topic: str, timestamp_ms: float, payload: dict[str, Any]) -> np.ndarray:
        feature_vector = payload.get("feature_vector")
        if feature_vector is not None:
            array = np.asarray(feature_vector, dtype=np.float32).reshape(-1)
            if array.size >= self.feature_dim:
                return array[: self.feature_dim]
            padded = np.zeros(self.feature_dim, dtype=np.float32)
            padded[: array.size] = array
            return padded

        vector = np.zeros(self.feature_dim, dtype=np.float32)
        vector[0] = float(timestamp_ms) / 1000.0
        vector[1] = self.last_camera_timestamp_ms / 1000.0
        vector[2] = self.last_audio_timestamp_ms / 1000.0
        vector[3] = self.last_robot_pose_timestamp_ms / 1000.0
        vector[4] = self.synchronization_error_ms()
        vector[5] = 1.0 if self.health_state != "nominal" else 0.0
        vector[6] = 1.0 if self.latest_head_command == "track_face" else 0.0
        vector[7] = 1.0 if self.latest_speech_command == "empathetic_prompt" else 0.0
        vector[8] = _stable_float(payload.get("visual_signal"), _hash_to_unit_interval(payload, "visual"))
        vector[9] = _stable_float(payload.get("audio_signal"), _hash_to_unit_interval(payload, "audio"))
        vector[10] = _stable_float(payload.get("physio_signal"), _hash_to_unit_interval(payload, "physio"))

        basis = np.linspace(0.0, 1.0, self.feature_dim - 11, dtype=np.float32)
        topic_hash = _hash_to_unit_interval({"topic": topic, "payload": payload}, "topic")
        vector[11:] = np.sin((basis + topic_hash) * np.pi).astype(np.float32)
        return vector

    def update(self, topic: str, timestamp_ms: float, payload: dict[str, Any] | None = None) -> None:
        payload = payload or {}
        timestamp_ms = float(timestamp_ms)
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

        sync_error_ms = self.synchronization_error_ms()
        feature_vector = self._payload_to_feature_vector(topic, timestamp_ms, payload)
        modality = self._topic_to_modality(topic)
        record = {
            "event_id": self.event_counter,
            "topic": topic,
            "modality": modality,
            "timestamp_ms": timestamp_ms,
            "health_state": self.health_state,
            "latest_head_command": self.latest_head_command,
            "latest_speech_command": self.latest_speech_command,
            "last_camera_timestamp_ms": self.last_camera_timestamp_ms,
            "last_audio_timestamp_ms": self.last_audio_timestamp_ms,
            "last_robot_pose_timestamp_ms": self.last_robot_pose_timestamp_ms,
            "sync_error_ms": round(float(sync_error_ms), 4),
            "temporal_consistency_ok": bool(sync_error_ms <= self.sync_threshold_ms and self._is_monotonic(timestamp_ms)),
            "visual_signal": _stable_float(payload.get("visual_signal"), np.nan),
            "audio_signal": _stable_float(payload.get("audio_signal"), np.nan),
            "physio_signal": _stable_float(payload.get("physio_signal"), np.nan),
            "feature_vector": feature_vector.tolist(),
            "payload": payload,
        }
        self.history.append(record)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history :]

    def _is_monotonic(self, timestamp_ms: float) -> bool:
        if not self.history:
            return True
        return float(timestamp_ms) >= float(self.history[-1]["timestamp_ms"])

    def synchronization_error_ms(self) -> float:
        timestamps = [
            self.last_camera_timestamp_ms,
            self.last_audio_timestamp_ms,
            self.last_robot_pose_timestamp_ms,
        ]
        valid = [value for value in timestamps if value > 0.0]
        if len(valid) < 2:
            return 0.0
        return float(max(valid) - min(valid))

    def temporal_consistency_check(self) -> dict[str, float | bool]:
        if not self.history:
            return {
                "num_updates": 0,
                "mean_sync_error_ms": 0.0,
                "p99_sync_error_ms": 0.0,
                "within_threshold_ratio": 1.0,
                "monotonic_timestamps": True,
            }
        df = self.to_dataframe()
        monotonic = bool(df["timestamp_ms"].is_monotonic_increasing)
        within_threshold_ratio = float((df["sync_error_ms"] <= self.sync_threshold_ms).mean())
        return {
            "num_updates": int(len(df)),
            "mean_sync_error_ms": float(df["sync_error_ms"].mean()),
            "p99_sync_error_ms": float(df["sync_error_ms"].quantile(0.99)),
            "within_threshold_ratio": within_threshold_ratio,
            "monotonic_timestamps": monotonic,
        }

    def impute_missing_modalities(self) -> pd.DataFrame:
        df = self.to_dataframe()
        if df.empty:
            return df
        numeric_columns = ["visual_signal", "audio_signal", "physio_signal"]
        interpolated = df.copy()
        interpolated[numeric_columns] = interpolated[numeric_columns].interpolate(method="linear", limit_direction="both")
        return interpolated

    def get_state_at_time(self, timestamp_ms: float) -> dict[str, Any]:
        if not self.history:
            raise ValueError("Digital-twin buffer is empty.")
        df = self.impute_missing_modalities().sort_values("timestamp_ms").reset_index(drop=True)
        target = float(timestamp_ms)
        if target <= float(df.loc[0, "timestamp_ms"]):
            return df.iloc[0].to_dict()
        if target >= float(df.loc[len(df) - 1, "timestamp_ms"]):
            return df.iloc[len(df) - 1].to_dict()

        earlier = df.loc[df["timestamp_ms"] <= target].iloc[-1]
        later = df.loc[df["timestamp_ms"] >= target].iloc[0]
        if float(later["timestamp_ms"]) == float(earlier["timestamp_ms"]):
            return earlier.to_dict()
        alpha = (target - float(earlier["timestamp_ms"])) / (float(later["timestamp_ms"]) - float(earlier["timestamp_ms"]))
        state = earlier.to_dict()
        for column in [
            "timestamp_ms",
            "last_camera_timestamp_ms",
            "last_audio_timestamp_ms",
            "last_robot_pose_timestamp_ms",
            "sync_error_ms",
            "visual_signal",
            "audio_signal",
            "physio_signal",
        ]:
            state[column] = float(earlier[column]) + alpha * (float(later[column]) - float(earlier[column]))
        state["interpolated"] = True
        return state

    def latest_feature_history(self, sequence_length: int) -> np.ndarray:
        if not self.history:
            return np.zeros((sequence_length, self.feature_dim), dtype=np.float32)
        features = [np.asarray(row["feature_vector"], dtype=np.float32) for row in self.history[-sequence_length:]]
        if len(features) < sequence_length:
            pad_count = sequence_length - len(features)
            padding = [np.zeros(self.feature_dim, dtype=np.float32) for _ in range(pad_count)]
            features = padding + features
        return np.stack(features, axis=0)

    def to_dataframe(self) -> pd.DataFrame:
        if not self.history:
            return pd.DataFrame(
                columns=[
                    "event_id",
                    "topic",
                    "modality",
                    "timestamp_ms",
                    "health_state",
                    "latest_head_command",
                    "latest_speech_command",
                    "last_camera_timestamp_ms",
                    "last_audio_timestamp_ms",
                    "last_robot_pose_timestamp_ms",
                    "sync_error_ms",
                    "temporal_consistency_ok",
                    "visual_signal",
                    "audio_signal",
                    "physio_signal",
                    "feature_vector",
                    "payload",
                ]
            )
        return pd.DataFrame(self.history)
