from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import soundfile as sf

from src.features.audio_features import generate_audio_features
from src.features.context_features import generate_context_features
from src.features.video_features import generate_video_features


def resolve_latest_session(project_root: Path) -> Path:
    sessions_root = project_root / "data" / "pilot" / "sessions"
    candidates = sorted((path for path in sessions_root.glob("*") if path.is_dir()), key=lambda item: item.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError("No pilot sessions found under data/pilot/sessions.")
    return candidates[-1]


def load_anchor_session(session_dir: Path) -> dict[str, pd.DataFrame]:
    return {
        "video": pd.read_csv(session_dir / "video_frames.csv"),
        "audio": pd.read_csv(session_dir / "audio_chunks.csv"),
        "context": pd.read_csv(session_dir / "robot_state_log.csv"),
        "metadata": pd.read_csv(session_dir / "session_metadata.csv"),
    }


def build_real_anchor_streams(session_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    loaded = load_anchor_session(session_dir)
    video_df = loaded["video"].copy()
    video_df["label"] = "pilot_unlabeled"
    video_df["video_feature_0"] = np.linspace(0.0, 1.0, len(video_df))
    video_df["video_feature_1"] = np.linspace(1.0, 0.0, len(video_df))

    audio_df = loaded["audio"].copy()
    audio_df["label"] = "pilot_unlabeled"
    audio_df["audio_feature_0"] = audio_df["rms_energy"].fillna(0.0)
    audio_df["audio_feature_1"] = audio_df["rms_energy"].rolling(3, min_periods=1).mean().fillna(0.0)

    context_df = loaded["context"].copy()
    context_df["label"] = "pilot_unlabeled"
    context_df["context_feature_0"] = np.linspace(0.0, 0.5, len(context_df))
    context_df["context_feature_1"] = np.linspace(0.5, 0.0, len(context_df))

    physiology_df = pd.DataFrame(
        {
            "session_id": [session_dir.name],
            "timestamp_ms": [0.0],
            "phys_timestamp_ms": [0.0],
            "phys_available": [0],
            "label": ["pilot_unlabeled"],
            "phys_feature_0": [0.0],
            "phys_feature_1": [0.0],
        }
    )
    return video_df, audio_df, context_df, physiology_df


def sample_real_anchor_for_baseline(session_dir: Path, max_frames: int = 20) -> pd.DataFrame:
    video_df = pd.read_csv(session_dir / "video_frames.csv")
    if len(video_df) > max_frames:
        video_df = video_df.iloc[:: max(1, len(video_df) // max_frames)].head(max_frames).copy()
    return video_df.reset_index(drop=True)


def build_real_anchor_demo_bundle(session_dir: Path, seed: int = 42) -> dict[str, object]:
    loaded = load_anchor_session(session_dir)
    video_df = loaded["video"]
    audio_df = loaded["audio"]
    context_df = loaded["context"]
    sample_count = min(len(video_df), len(audio_df), max(len(context_df), 1), 32)
    labels = ["neutral"] * sample_count
    rng = np.random.default_rng(seed)
    return {
        "video": generate_video_features(labels, rng),
        "audio": generate_audio_features(labels, rng),
        "context": generate_context_features(labels, rng),
        "labels": labels,
    }


def baseline_predictions_from_frames(session_dir: Path, max_frames: int = 20) -> pd.DataFrame:
    from perception.face_emotion import detect_face_emotion_from_frame

    sampled = sample_real_anchor_for_baseline(session_dir, max_frames=max_frames)
    rows: list[dict[str, object]] = []
    for row in sampled.to_dict(orient="records"):
        frame = cv2.imread(str(row["frame_path"]))
        emotion, confidence = detect_face_emotion_from_frame(frame, enforce_detection=False)
        rows.append(
            {
                "frame_index": row["frame_index"],
                "timestamp_ms": row["timestamp_ms"],
                "predicted_emotion": emotion or "undetected",
                "confidence": confidence,
                "data_source_type": "pilot_real_anchor",
                "runtime_type": "software_only",
                "evidence_level": "pilot_demonstration",
            }
        )
    return pd.DataFrame(rows)


def summarize_anchor_audio(session_dir: Path) -> dict[str, float]:
    waveform, samplerate = sf.read(session_dir / "audio.wav")
    mono = np.asarray(waveform).reshape(-1)
    return {
        "samplerate": float(samplerate),
        "duration_seconds": float(len(mono) / samplerate),
        "rms_energy": float(np.sqrt(np.mean(np.square(mono)))) if mono.size else 0.0,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Inspect a pilot real-anchor session.")
    parser.add_argument("--session-dir", default=None)
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    session_dir = Path(args.session_dir).resolve() if args.session_dir else resolve_latest_session(Path(args.project_root).resolve())
    payload = {
        "session_dir": str(session_dir),
        "audio_summary": summarize_anchor_audio(session_dir),
        "video_rows": int(len(pd.read_csv(session_dir / "video_frames.csv"))),
        "audio_rows": int(len(pd.read_csv(session_dir / "audio_chunks.csv"))),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
