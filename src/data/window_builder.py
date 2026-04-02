from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class WindowConfig:
    window_ms: int = 2000
    hop_ms: int = 1000


def build_windows(aligned_df: pd.DataFrame, config: WindowConfig | None = None) -> pd.DataFrame:
    config = config or WindowConfig()
    start = int(aligned_df["timestamp_ms"].min())
    end = int(aligned_df["timestamp_ms"].max())
    rows: list[dict[str, object]] = []
    window_id = 0
    for window_start in range(start, end - config.window_ms + 1, config.hop_ms):
        window_end = window_start + config.window_ms
        subset = aligned_df.loc[(aligned_df["timestamp_ms"] >= window_start) & (aligned_df["timestamp_ms"] < window_end)]
        if subset.empty:
            continue
        majority_label = subset["label"].mode().iloc[0]
        rows.append(
            {
                "window_id": window_id,
                "start_ms": window_start,
                "end_ms": window_end,
                "sample_count": int(len(subset)),
                "majority_label": majority_label,
                "video_frames": int(subset["video_available"].sum()),
                "audio_frames": int(subset["audio_available"].sum()),
                "context_frames": int(subset["context_available"].sum()),
                "physiology_frames": int(subset["phys_available"].sum()),
            }
        )
        window_id += 1
    return pd.DataFrame(rows)
