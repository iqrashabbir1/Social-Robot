from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import sounddevice as sd
import soundfile as sf

from src.common.io_utils import write_dataframe, write_json
from src.data.session_manifest_builder import build_session_manifest


def _default_session_name() -> str:
    return datetime.now().strftime("pilot_%Y%m%d_%H%M%S")


def collect_pilot_session(
    project_root: Path,
    session_name: str | None = None,
    duration_seconds: int = 5,
    frame_interval_ms: int = 200,
    audio_samplerate: int = 16000,
) -> dict[str, str]:
    session_id = session_name or _default_session_name()
    session_dir = project_root / "data" / "pilot" / "sessions" / session_id
    frames_dir = session_dir / "frames"
    session_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    total_samples = int(duration_seconds * audio_samplerate)
    audio_recording = sd.rec(total_samples, samplerate=audio_samplerate, channels=1, dtype="float32")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam for pilot-session collection.")

    start = time.perf_counter()
    next_frame_deadline = 0.0
    video_rows: list[dict[str, object]] = []
    context_rows: list[dict[str, object]] = []
    frame_index = 0
    context_tick = 0

    while True:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if elapsed_ms >= duration_seconds * 1000.0:
            break

        if elapsed_ms >= next_frame_deadline:
            success, frame = cap.read()
            if success and frame is not None:
                frame_path = frames_dir / f"frame_{frame_index:05d}.jpg"
                cv2.imwrite(str(frame_path), frame)
                video_rows.append(
                    {
                        "session_id": session_id,
                        "frame_index": frame_index,
                        "timestamp_ms": round(elapsed_ms, 4),
                        "video_timestamp_ms": round(elapsed_ms, 4),
                        "video_available": 1,
                        "frame_path": str(frame_path),
                    }
                )
                frame_index += 1
            next_frame_deadline += frame_interval_ms

        if not context_rows or elapsed_ms - float(context_rows[-1]["timestamp_ms"]) >= 250.0:
            context_rows.append(
                {
                    "session_id": session_id,
                    "timestamp_ms": round(elapsed_ms, 4),
                    "context_timestamp_ms": round(elapsed_ms, 4),
                    "context_available": 1,
                    "robot_state": f"idle_tick_{context_tick}",
                }
            )
            context_tick += 1

        time.sleep(0.01)

    cap.release()
    sd.wait()

    audio_path = session_dir / "audio.wav"
    sf.write(audio_path, audio_recording, audio_samplerate)

    chunk_size = max(1, int(audio_samplerate * (frame_interval_ms / 1000.0)))
    audio_rows: list[dict[str, object]] = []
    mono = np.asarray(audio_recording).reshape(-1)
    for index, start_sample in enumerate(range(0, len(mono), chunk_size)):
        chunk = mono[start_sample : start_sample + chunk_size]
        if chunk.size == 0:
            continue
        timestamp_ms = (start_sample / audio_samplerate) * 1000.0
        audio_rows.append(
            {
                "session_id": session_id,
                "chunk_index": index,
                "timestamp_ms": round(timestamp_ms, 4),
                "audio_timestamp_ms": round(timestamp_ms, 4),
                "audio_available": 1,
                "rms_energy": round(float(np.sqrt(np.mean(np.square(chunk)))), 6),
                "audio_path": str(audio_path),
            }
        )

    write_dataframe(session_dir / "video_frames.csv", pd.DataFrame(video_rows))
    write_dataframe(session_dir / "audio_chunks.csv", pd.DataFrame(audio_rows))
    write_dataframe(session_dir / "robot_state_log.csv", pd.DataFrame(context_rows))
    payload = build_session_manifest(session_dir)
    write_json(session_dir / "collection_summary.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect a small pilot real-anchor session with webcam and microphone.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--session-name", default=None)
    parser.add_argument("--duration-seconds", type=int, default=5)
    parser.add_argument("--frame-interval-ms", type=int, default=200)
    args = parser.parse_args()
    payload = collect_pilot_session(
        Path(args.project_root).resolve(),
        session_name=args.session_name,
        duration_seconds=args.duration_seconds,
        frame_interval_ms=args.frame_interval_ms,
    )
    import json

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
