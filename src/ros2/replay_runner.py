from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_replay_events(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return pd.DataFrame(rows)
    return pd.read_csv(path)


def replay_with_speed(events: pd.DataFrame, playback_speed: float = 1.0) -> pd.DataFrame:
    replayed = events.copy()
    if "received_timestamp_ms" in replayed.columns:
        replayed["received_timestamp_ms"] = replayed["received_timestamp_ms"] / playback_speed
    replayed["replay_speed"] = playback_speed
    return replayed
