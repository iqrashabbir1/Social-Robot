from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from src.digital_twin.twin_state import DigitalTwinState
from src.ros2.topic_logger import TopicEvent, TopicLogger


def replay_topic_stream(events: pd.DataFrame, seed: int, mode: str = "PLAYBACK") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    logger = TopicLogger()
    twin = DigitalTwinState()
    timing_rows: list[dict[str, object]] = []
    sync_rows: list[dict[str, object]] = []

    for step, step_df in events.groupby("step", sort=True):
        sync_markers: list[float] = []
        failure_streak = 0
        for row in step_df.to_dict(orient="records"):
            source_ts = float(row["source_timestamp_ms"])
            latency_ms = abs(rng.normal(24.0, 5.0))
            received_ts = source_ts + latency_ms
            mirrored_ts = received_ts + abs(rng.normal(3.5, 1.2))
            cpu_percent = 24.0 + rng.normal(0.0, 1.2)
            memory_mb = 212.0 + rng.normal(0.0, 5.0)
            dropped = int(rng.random() < 0.01)
            success_flag = int(dropped == 0 and latency_ms < 40.0)
            recovered_flag = int(failure_streak > 0 and success_flag == 1)
            failure_streak = 0 if success_flag == 1 else failure_streak + 1
            twin.update(str(row["topic"]), mirrored_ts, {"payload_note": row["payload_note"]})
            sync_markers.append(mirrored_ts)

            event = TopicEvent(
                mode=mode,
                step=int(step),
                topic=str(row["topic"]),
                source_timestamp_ms=round(source_ts, 4),
                received_timestamp_ms=round(received_ts, 4),
                mirrored_timestamp_ms=round(mirrored_ts, 4),
                latency_ms=round(latency_ms, 4),
                payload_size=int(row.get("payload_size", 256)),
                dropped=dropped,
                success_flag=success_flag,
                recovered_flag=recovered_flag,
                cpu_percent=round(cpu_percent, 4),
                memory_mb=round(memory_mb, 4),
                payload_note=str(row.get("payload_note", "")),
            )
            logger.log(event)
            timing_rows.append({**asdict(event), "source_type": row.get("source_type", "unknown")})

        if sync_markers:
            sync_rows.append(
                {
                    "mode": mode,
                    "step": int(step),
                    "sync_error_ms": round(max(sync_markers) - min(sync_markers), 4),
                    "event_count": int(len(sync_markers)),
                }
            )

    return logger.to_dataframe(), pd.DataFrame(sync_rows), pd.DataFrame(timing_rows)
