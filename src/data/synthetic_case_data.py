from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


SEED = 42


@dataclass(frozen=True)
class Paths:
    project_root: Path

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def output_csv_dir(self) -> Path:
        return self.project_root / "outputs" / "csv"

    @property
    def output_tables_dir(self) -> Path:
        return self.project_root / "outputs" / "tables"

    @property
    def output_logs_dir(self) -> Path:
        return self.project_root / "outputs" / "logs"


def ensure_dirs(paths: Paths) -> None:
    for directory in (paths.output_csv_dir, paths.output_tables_dir, paths.output_logs_dir):
        directory.mkdir(parents=True, exist_ok=True)


def generate_physiology_timeseries(length: int = 240, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    timeline = pd.date_range("2026-01-01 08:00:00", periods=length, freq="5min")
    hr = 72 + rng.normal(0, 4, length).cumsum() * 0.08
    sbp = 124 + rng.normal(0, 3, length)
    dbp = 78 + rng.normal(0, 2, length)
    spo2 = 97.6 + rng.normal(0, 0.4, length)
    gait_var = np.clip(0.25 + rng.normal(0, 0.05, length), 0.05, 0.7)
    activity = np.clip(0.55 + rng.normal(0, 0.12, length), 0.05, 1.0)
    speech_valence = np.clip(0.55 + rng.normal(0, 0.16, length), 0.0, 1.0)

    deterioration_zone = np.arange(length) > int(length * 0.65)
    hr[deterioration_zone] += np.linspace(2, 18, deterioration_zone.sum())
    sbp[deterioration_zone] += np.linspace(0, 14, deterioration_zone.sum())
    spo2[deterioration_zone] -= np.linspace(0.0, 4.5, deterioration_zone.sum())
    gait_var[deterioration_zone] += np.linspace(0.0, 0.18, deterioration_zone.sum())
    activity[deterioration_zone] -= np.linspace(0.0, 0.28, deterioration_zone.sum())
    speech_valence[deterioration_zone] -= np.linspace(0.0, 0.22, deterioration_zone.sum())

    risk_score = (
        0.18 * np.clip((hr - 78) / 25, 0, None)
        + 0.16 * np.clip((sbp - 130) / 20, 0, None)
        + 0.24 * np.clip((96.0 - spo2) / 4, 0, None)
        + 0.22 * np.clip((gait_var - 0.28) / 0.22, 0, None)
        + 0.12 * np.clip((0.52 - activity) / 0.35, 0, None)
        + 0.08 * np.clip((0.5 - speech_valence) / 0.4, 0, None)
    )
    risk_score = np.clip(risk_score, 0, 1)
    anomaly_score = np.clip(0.35 * risk_score + rng.normal(0.06, 0.05, length), 0, 1)

    risk_level = np.where(
        risk_score >= 0.62,
        "high",
        np.where(risk_score >= 0.42, "moderate", "low"),
    )
    anomaly_label = np.where(anomaly_score >= 0.52, 1, 0)
    event_label = np.where((risk_score >= 0.62) | ((spo2 < 94.5) & (hr > 84)), 1, 0)

    df = pd.DataFrame(
        {
            "timestamp": timeline,
            "heart_rate_bpm": np.round(hr, 2),
            "systolic_bp_mmHg": np.round(sbp, 2),
            "diastolic_bp_mmHg": np.round(dbp, 2),
            "spo2_percent": np.round(spo2, 2),
            "gait_variability": np.round(gait_var, 4),
            "activity_index": np.round(activity, 4),
            "speech_valence_index": np.round(speech_valence, 4),
            "risk_score": np.round(risk_score, 4),
            "anomaly_score": np.round(anomaly_score, 4),
            "risk_level": risk_level,
            "anomaly_label": anomaly_label,
            "event_label": event_label,
        }
    )
    return df


def generate_medication_log(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    schedule = pd.date_range("2026-01-01 08:00:00", periods=24, freq="12h")
    statuses = []
    reasons = []
    delays = []
    for idx, timestamp in enumerate(schedule):
        if idx in {5, 8, 14, 19}:
            status = "missed"
            reason = ["forgetfulness", "side_effects", "refusal", "access_issue"][idx % 4]
            delay = np.nan
        elif idx in {2, 11, 15, 21}:
            status = "delayed"
            reason = ["sleeping", "caregiver_unavailable", "confusion", "telepresence_pending"][idx % 4]
            delay = int(rng.integers(18, 95))
        else:
            status = "taken"
            reason = "on_time"
            delay = int(rng.integers(0, 15))
        statuses.append(status)
        reasons.append(reason)
        delays.append(delay)

    return pd.DataFrame(
        {
            "scheduled_time": schedule,
            "medication_name": ["CardioSafe-10mg"] * len(schedule),
            "adherence_status": statuses,
            "missed_or_delayed_reason": reasons,
            "delay_minutes": delays,
            "escalation_required": [status in {"missed", "delayed"} for status in statuses],
        }
    )


def generate_hitl_alerts(physiology: pd.DataFrame, medication: pd.DataFrame) -> pd.DataFrame:
    alert_rows = []
    for _, row in physiology.iloc[::30].iterrows():
        alert_rows.append(
            {
                "timestamp": row["timestamp"],
                "alert_type": "health_risk",
                "severity": row["risk_level"],
                "confidence": float(row["risk_score"]),
                "requires_override": row["risk_level"] == "high",
                "ack_latency_sec": 24 if row["risk_level"] == "low" else 68 if row["risk_level"] == "moderate" else 41,
                "override_action": "reviewed" if row["risk_level"] != "high" else "telepresence_escalation",
            }
        )
    for _, row in medication[medication["escalation_required"]].iterrows():
        severity = "moderate" if row["adherence_status"] == "delayed" else "high"
        alert_rows.append(
            {
                "timestamp": row["scheduled_time"],
                "alert_type": "medication_adherence",
                "severity": severity,
                "confidence": 0.71 if severity == "moderate" else 0.85,
                "requires_override": severity == "high",
                "ack_latency_sec": 55 if severity == "moderate" else 38,
                "override_action": "reminder_confirmed" if severity == "moderate" else "clinician_review",
            }
        )
    df = pd.DataFrame(alert_rows).sort_values("timestamp").reset_index(drop=True)
    return df


def generate_privacy_tradeoff() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "profile": ["minimal", "balanced", "clinical", "rich_multimodal"],
            "privacy_score": [0.96, 0.82, 0.63, 0.44],
            "utility_score": [0.52, 0.74, 0.86, 0.93],
            "edge_latency_ms": [88, 103, 124, 162],
            "cloud_dependency_ratio": [0.08, 0.18, 0.29, 0.42],
        }
    )


def generate_latency_resource_tradeoff() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "mode": ["baseline_mer", "late_fusion_mer", "transformer_mer", "full_stack_edge", "full_stack_hybrid"],
            "latency_ms": [72, 94, 132, 156, 188],
            "cpu_percent": [21, 28, 35, 48, 42],
            "memory_mb": [230, 315, 460, 710, 640],
            "clinical_utility_score": [0.46, 0.61, 0.73, 0.84, 0.89],
            "evidence_level": [
                "implemented_real_baseline",
                "simulation_based_evaluation",
                "simulation_based_evaluation",
                "simulation_based_evaluation",
                "simulation_based_evaluation",
            ],
        }
    )


def generate_module_contribution() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "module": [
                "Speech baseline",
                "Vision baseline",
                "Physiology risk head",
                "Medication reasoner",
                "Digital twin",
                "KG plus LLM explainer",
                "HITL dashboard",
                "Privacy controller",
            ],
            "contribution_score": [0.11, 0.12, 0.18, 0.14, 0.15, 0.1, 0.11, 0.09],
            "category": [
                "perception",
                "perception",
                "prediction",
                "prediction",
                "reasoning",
                "reasoning",
                "oversight",
                "deployment",
            ],
        }
    )


def generate_end_to_end_workflow() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "step_order": list(range(1, 10)),
            "step_name": [
                "Acquire multimodal observations",
                "Synchronize windows",
                "Extract embeddings",
                "Predict emotion and risk",
                "Update digital twin",
                "Query knowledge graph",
                "Generate explanation",
                "Dashboard review and telepresence",
                "Log decision and feedback",
            ],
            "median_latency_ms": [18, 11, 27, 44, 9, 15, 38, 21, 6],
        }
    )


def generate_pilot_readiness() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "check": [
                "Reproducible benchmark pipeline",
                "Baseline MER execution",
                "Simulation-backed risk evaluation",
                "Medication adherence workflow",
                "Dashboard prototype",
                "Knowledge graph population",
                "Telepresence escalation path",
                "Privacy policy configuration",
                "Ethics and consent package",
                "Clinical site agreement",
            ],
            "status": [
                "complete",
                "complete",
                "complete",
                "complete",
                "complete",
                "complete",
                "complete",
                "complete",
                "pending_external",
                "pending_external",
            ],
            "owner": [
                "repo",
                "repo",
                "repo",
                "repo",
                "repo",
                "repo",
                "repo",
                "repo",
                "deployment_team",
                "deployment_team",
            ],
        }
    )


def update_case_study_metrics(metric_path: Path, values: dict[str, object]) -> None:
    df = pd.read_csv(metric_path)
    for metric_id, current_value in values.items():
        df.loc[df["metric_id"] == metric_id, "current_value"] = current_value
    df.to_csv(metric_path, index=False)


def write_df(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
