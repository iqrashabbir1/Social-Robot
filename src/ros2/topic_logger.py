from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.common.io_utils import write_dataframe


@dataclass
class TopicEvent:
    mode: str
    step: int
    topic: str
    source_timestamp_ms: float
    received_timestamp_ms: float
    mirrored_timestamp_ms: float
    latency_ms: float
    payload_size: int
    dropped: int
    success_flag: int
    recovered_flag: int
    cpu_percent: float
    memory_mb: float
    payload_note: str


class TopicLogger:
    def __init__(self) -> None:
        self._events: list[TopicEvent] = []

    def log(self, event: TopicEvent) -> None:
        self._events.append(event)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(asdict(event) for event in self._events)

    def write_csv(self, path: Path) -> None:
        write_dataframe(path, self.to_dataframe())

    def payload_overview(self) -> list[dict[str, Any]]:
        return [
            {
                "mode": event.mode,
                "step": event.step,
                "topic": event.topic,
                "payload_note": event.payload_note,
                "payload_size": event.payload_size,
            }
            for event in self._events
        ]
