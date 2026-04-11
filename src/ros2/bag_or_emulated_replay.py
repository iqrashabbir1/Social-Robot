from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.real_anchor_loader import resolve_latest_session
from src.ros2.interface_spec import default_interface_spec
from src.ros2.replay_runner import load_replay_events


def ros2_available() -> bool:
    return shutil.which("ros2") is not None


def _mock_topic_stream(steps: int, interval_ms: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    topics = [item.topic for item in default_interface_spec()]
    for step in range(steps):
        base_time = step * interval_ms
        for topic_index, topic in enumerate(topics):
            rows.append(
                {
                    "step": step,
                    "topic": topic,
                    "source_timestamp_ms": round(float(base_time + topic_index * 2.0 + rng.normal(0.0, 3.0)), 4),
                    "payload_note": f"mock_{topic.replace('/', '_')}",
                    "payload_size": int(abs(rng.normal(256.0, 40.0))),
                    "source_type": "mock_topic_stream",
                }
            )
    return pd.DataFrame(rows)


def _pilot_topic_stream(project_root: Path) -> pd.DataFrame:
    session_dir = resolve_latest_session(project_root)
    video_df = pd.read_csv(session_dir / "video_frames.csv")
    audio_df = pd.read_csv(session_dir / "audio_chunks.csv")
    context_df = pd.read_csv(session_dir / "robot_state_log.csv")

    rows: list[dict[str, object]] = []
    for step, row in enumerate(video_df.head(60).to_dict(orient="records")):
        base_time = float(row["timestamp_ms"])
        rows.extend(
            [
                {"step": step, "topic": "/camera/image_raw", "source_timestamp_ms": base_time, "payload_note": "pilot_frame", "payload_size": 512, "source_type": "pilot_real_anchor"},
                {"step": step, "topic": "/head_cmd", "source_timestamp_ms": base_time + 12.0, "payload_note": "head_follow", "payload_size": 64, "source_type": "pilot_real_anchor"},
                {"step": step, "topic": "/speech_cmd", "source_timestamp_ms": base_time + 18.0, "payload_note": "speech_prompt", "payload_size": 96, "source_type": "pilot_real_anchor"},
                {"step": step, "topic": "/event_log", "source_timestamp_ms": base_time + 22.0, "payload_note": "playback_tick", "payload_size": 80, "source_type": "pilot_real_anchor"},
                {"step": step, "topic": "/system_health", "source_timestamp_ms": base_time + 24.0, "payload_note": "runtime_ok", "payload_size": 72, "source_type": "pilot_real_anchor"},
            ]
        )
    for step, row in enumerate(audio_df.head(60).to_dict(orient="records")):
        rows.append(
            {
                "step": step,
                "topic": "/audio/stream",
                "source_timestamp_ms": float(row["timestamp_ms"]),
                "payload_note": "pilot_audio",
                "payload_size": 320,
                "source_type": "pilot_real_anchor",
            }
        )
    for step, row in enumerate(context_df.head(60).to_dict(orient="records")):
        rows.append(
            {
                "step": step,
                "topic": "/robot_pose",
                "source_timestamp_ms": float(row["timestamp_ms"]),
                "payload_note": str(row["robot_state"]),
                "payload_size": 128,
                "source_type": "pilot_real_anchor",
            }
        )
    return pd.DataFrame(rows).sort_values(["step", "source_timestamp_ms", "topic"]).reset_index(drop=True)


def load_or_emulate_topic_stream(project_root: Path, replay_source: str, seed: int, steps: int) -> tuple[pd.DataFrame, dict[str, str]]:
    source_path = Path(replay_source).resolve() if replay_source else Path()
    runtime_kind = "ros2_live" if ros2_available() else "ros2_playback_grounded"
    if source_path and source_path.exists():
        events = load_replay_events(source_path)
        normalized = pd.DataFrame(
            {
                "step": events.get("step", range(len(events))),
                "topic": events.get("topic", "/event_log"),
                "source_timestamp_ms": events.get("source_timestamp_ms", events.get("timestamp_ms", 0.0)),
                "payload_note": events.get("payload_note", "loaded_replay"),
                "payload_size": events.get("payload_size", 256),
                "source_type": "recorded_topic_stream",
            }
        )
        return normalized, {"runtime_type": runtime_kind, "replay_mode": "recorded_source"}

    try:
        pilot_events = _pilot_topic_stream(project_root)
        return pilot_events, {"runtime_type": runtime_kind, "replay_mode": "pilot_real_anchor_emulation"}
    except Exception:
        return _mock_topic_stream(steps=steps, interval_ms=100.0, seed=seed), {"runtime_type": runtime_kind, "replay_mode": "mock_emulated_topics"}
