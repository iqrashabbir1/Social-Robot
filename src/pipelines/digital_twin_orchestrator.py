from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class TwinState:
    patient_id: str
    risk_level: str
    adherence_state: str
    emotion_state: str
    anomaly_state: str
    oversight_required: bool


def update_twin_state(observations: Dict[str, str]) -> TwinState:
    risk_level = observations.get("risk_level", "unknown")
    adherence_state = observations.get("adherence_state", "unknown")
    emotion_state = observations.get("emotion_state", "unknown")
    anomaly_state = observations.get("anomaly_state", "unknown")
    oversight_required = risk_level in {"high", "critical"} or anomaly_state == "present"

    return TwinState(
        patient_id=observations.get("patient_id", "sim_patient"),
        risk_level=risk_level,
        adherence_state=adherence_state,
        emotion_state=emotion_state,
        anomaly_state=anomaly_state,
        oversight_required=oversight_required,
    )


def ros2_topics() -> List[str]:
    return [
        "/caregiver/sensors/visual",
        "/caregiver/sensors/audio",
        "/caregiver/sensors/physiology",
        "/caregiver/digital_twin/state",
        "/caregiver/alerts/risk",
        "/caregiver/alerts/adherence",
        "/caregiver/dashboard/override",
        "/caregiver/telepresence/session",
    ]
