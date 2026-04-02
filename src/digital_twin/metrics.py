from __future__ import annotations

import pandas as pd

from src.evaluation.metrics_system import rate_from_binary, summarize_latency


def summarize_cs1_latency(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for mode, subset in events.groupby("mode"):
        summary = summarize_latency(subset["latency_ms"].tolist())
        rows.append(
            {
                "mode": mode,
                "mean_latency_ms": round(summary["mean_latency_ms"], 4),
                "p95_latency_ms": round(summary["p95_latency_ms"], 4),
                "max_latency_ms": round(summary["max_latency_ms"], 4),
                "message_drop_rate": round(rate_from_binary(subset["dropped"]), 4),
                "task_success_rate": round(rate_from_binary(subset["success_flag"]), 4),
                "recovery_rate": round(rate_from_binary(subset["recovered_flag"]), 4),
                "cpu_percent_mean": round(float(subset["cpu_percent"].mean()), 4),
                "memory_mb_mean": round(float(subset["memory_mb"].mean()), 4),
            }
        )
    return pd.DataFrame(rows)


def summarize_fault_results(events: pd.DataFrame) -> pd.DataFrame:
    faulty = events.loc[events["mode"] == "M4"].copy()
    if faulty.empty:
        return pd.DataFrame(
            columns=[
                "fault_type",
                "severity",
                "task_success_rate",
                "recovery_rate",
                "message_drop_rate",
                "mean_latency_ms",
            ]
        )
    rows: list[dict[str, object]] = []
    for (fault_type, severity), subset in faulty.groupby(["fault_type", "severity"]):
        summary = summarize_latency(subset["latency_ms"].tolist())
        rows.append(
            {
                "fault_type": fault_type,
                "severity": severity,
                "task_success_rate": round(rate_from_binary(subset["success_flag"]), 4),
                "recovery_rate": round(rate_from_binary(subset["recovered_flag"]), 4),
                "message_drop_rate": round(rate_from_binary(subset["dropped"]), 4),
                "mean_latency_ms": round(summary["mean_latency_ms"], 4),
            }
        )
    return pd.DataFrame(rows)
