from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.common.io_utils import write_dataframe, write_json


def build_session_manifest(session_dir: Path) -> dict[str, str]:
    video_manifest_path = session_dir / "video_frames.csv"
    audio_manifest_path = session_dir / "audio_chunks.csv"
    context_manifest_path = session_dir / "robot_state_log.csv"

    video_df = pd.read_csv(video_manifest_path) if video_manifest_path.exists() else pd.DataFrame()
    audio_df = pd.read_csv(audio_manifest_path) if audio_manifest_path.exists() else pd.DataFrame()
    context_df = pd.read_csv(context_manifest_path) if context_manifest_path.exists() else pd.DataFrame()

    start_candidates = []
    end_candidates = []
    for df, column in ((video_df, "timestamp_ms"), (audio_df, "timestamp_ms"), (context_df, "timestamp_ms")):
        if not df.empty and column in df.columns:
            start_candidates.append(float(df[column].min()))
            end_candidates.append(float(df[column].max()))

    duration_ms = max(end_candidates) - min(start_candidates) if start_candidates and end_candidates else 0.0
    session_metadata = pd.DataFrame(
        [
            {
                "session_id": session_dir.name,
                "duration_ms": round(duration_ms, 4),
                "video_frames": int(len(video_df)),
                "audio_chunks": int(len(audio_df)),
                "context_events": int(len(context_df)),
                "data_source_type": "pilot_real_anchor",
                "runtime_type": "software_only",
                "evidence_level": "pilot_demonstration",
            }
        ]
    )
    session_metadata_path = session_dir / "session_metadata.csv"
    write_dataframe(session_metadata_path, session_metadata)

    manifest_payload = {
        "session_dir": str(session_dir),
        "session_metadata": str(session_metadata_path),
        "video_frames_csv": str(video_manifest_path),
        "audio_chunks_csv": str(audio_manifest_path),
        "robot_state_csv": str(context_manifest_path),
    }
    manifest_json_path = session_dir / "session_manifest.json"
    write_json(manifest_json_path, manifest_payload)
    return {key: str(value) for key, value in manifest_payload.items()}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build a pilot-session manifest from captured real-anchor files.")
    parser.add_argument("--session-dir", required=True)
    args = parser.parse_args()
    payload = build_session_manifest(Path(args.session_dir).resolve())
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
