from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FaultProfile:
    fault_type: str
    severity: str
    delay_ms: float = 0.0
    drop_probability: float = 0.0
    sync_noise_ms: float = 0.0


def default_fault_profiles() -> list[FaultProfile]:
    return [
        FaultProfile("delay", "mild", delay_ms=28.0, sync_noise_ms=6.0),
        FaultProfile("delay", "severe", delay_ms=65.0, sync_noise_ms=14.0),
        FaultProfile("dropout", "mild", drop_probability=0.08, sync_noise_ms=8.0),
        FaultProfile("dropout", "severe", drop_probability=0.18, sync_noise_ms=15.0),
        FaultProfile("sensor_noise", "moderate", delay_ms=12.0, sync_noise_ms=18.0),
    ]


def apply_fault_profile(events: pd.DataFrame, profile: FaultProfile, rng: np.random.Generator) -> pd.DataFrame:
    faulty = events.copy()
    if "received_timestamp_ms" in faulty.columns:
        faulty["received_timestamp_ms"] = faulty["received_timestamp_ms"] + profile.delay_ms + rng.normal(
            0.0,
            max(profile.sync_noise_ms, 1.0),
            len(faulty),
        )
    if "mirrored_timestamp_ms" in faulty.columns:
        faulty["mirrored_timestamp_ms"] = faulty["mirrored_timestamp_ms"] + profile.delay_ms
    if profile.drop_probability > 0.0:
        dropped = rng.random(len(faulty)) < profile.drop_probability
        faulty.loc[dropped, "dropped"] = 1
        faulty.loc[dropped, "success_flag"] = 0
    faulty["fault_type"] = profile.fault_type
    faulty["severity"] = profile.severity
    return faulty
