from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class AlignmentConfig:
    tolerance_ms: int = 120


def align_modalities(
    video_df: pd.DataFrame,
    audio_df: pd.DataFrame,
    context_df: pd.DataFrame,
    physiology_df: pd.DataFrame,
    config: AlignmentConfig | None = None,
) -> pd.DataFrame:
    config = config or AlignmentConfig()
    merged = video_df.sort_values("timestamp_ms").copy()
    for source_df, prefix in (
        (audio_df.sort_values("timestamp_ms"), "audio"),
        (context_df.sort_values("timestamp_ms"), "context"),
        (physiology_df.sort_values("timestamp_ms"), "phys"),
    ):
        merged = pd.merge_asof(
            merged,
            source_df,
            on="timestamp_ms",
            direction="nearest",
            tolerance=config.tolerance_ms,
            suffixes=("", f"_{prefix}"),
        )
    merged["alignment_error_audio_ms"] = (merged["timestamp_ms"] - merged["audio_timestamp_ms"]).abs()
    merged["alignment_error_context_ms"] = (merged["timestamp_ms"] - merged["context_timestamp_ms"]).abs()
    merged["alignment_error_phys_ms"] = (merged["timestamp_ms"] - merged["phys_timestamp_ms"]).abs()
    return merged
