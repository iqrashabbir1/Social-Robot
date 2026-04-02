from __future__ import annotations

import pandas as pd

from src.data.modality_tracker import compute_modality_availability
from src.data.sync_pipeline import AlignmentConfig, align_modalities
from src.data.window_builder import WindowConfig, build_windows


def test_sync_pipeline_builds_windows_and_availability() -> None:
    video = pd.DataFrame(
        {
            "timestamp_ms": [0, 1000, 2000, 3000],
            "video_timestamp_ms": [0, 1000, 2000, 3000],
            "video_available": [1, 1, 1, 1],
            "label": ["happy", "happy", "sad", "sad"],
        }
    )
    audio = pd.DataFrame(
        {
            "timestamp_ms": [10, 1010, 2010, 3010],
            "audio_timestamp_ms": [10, 1010, 2010, 3010],
            "audio_available": [1, 1, 1, 0],
        }
    )
    context = pd.DataFrame(
        {
            "timestamp_ms": [20, 1020, 2020, 3020],
            "context_timestamp_ms": [20, 1020, 2020, 3020],
            "context_available": [1, 1, 1, 1],
        }
    )
    phys = pd.DataFrame(
        {
            "timestamp_ms": [30, 1030, 2030, 3030],
            "phys_timestamp_ms": [30, 1030, 2030, 3030],
            "phys_available": [1, 0, 1, 1],
        }
    )

    aligned = align_modalities(video, audio, context, phys, AlignmentConfig(tolerance_ms=80))
    windows = build_windows(aligned, WindowConfig(window_ms=2000, hop_ms=1000))
    availability = compute_modality_availability(aligned, windows)

    assert not aligned.empty
    assert len(windows) >= 2
    assert {"video_available_ratio", "audio_available_ratio", "physiology_available_ratio"}.issubset(availability.columns)
