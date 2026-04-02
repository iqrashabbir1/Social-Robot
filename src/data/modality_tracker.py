from __future__ import annotations

import pandas as pd


def compute_modality_availability(aligned_df: pd.DataFrame, windows_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for window in windows_df.to_dict(orient="records"):
        subset = aligned_df.loc[
            (aligned_df["timestamp_ms"] >= window["start_ms"]) & (aligned_df["timestamp_ms"] < window["end_ms"])
        ]
        rows.append(
            {
                "window_id": window["window_id"],
                "video_available_ratio": round(float(subset["video_available"].mean()), 4),
                "audio_available_ratio": round(float(subset["audio_available"].mean()), 4),
                "context_available_ratio": round(float(subset["context_available"].mean()), 4),
                "physiology_available_ratio": round(float(subset["phys_available"].mean()), 4),
                "all_modalities_present_ratio": round(
                    float(
                        (
                            subset["video_available"]
                            * subset["audio_available"]
                            * subset["context_available"]
                            * subset["phys_available"]
                        ).mean()
                    ),
                    4,
                ),
            }
        )
    return pd.DataFrame(rows)
