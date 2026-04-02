from __future__ import annotations

from typing import Iterable

import numpy as np


def summarize_latency(values_ms: Iterable[float]) -> dict[str, float]:
    values = np.asarray(list(values_ms), dtype=float)
    if values.size == 0:
        return {"mean_latency_ms": 0.0, "p95_latency_ms": 0.0, "max_latency_ms": 0.0}
    return {
        "mean_latency_ms": float(values.mean()),
        "p95_latency_ms": float(np.percentile(values, 95)),
        "max_latency_ms": float(values.max()),
    }


def rate_from_binary(values: Iterable[int | bool]) -> float:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return 0.0
    return float(arr.mean())
