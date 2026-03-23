from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class ModuleSpec:
    module_id: str
    layer: str
    name: str
    inputs: List[str]
    outputs: List[str]
    evidence_level: str


def get_module_specs() -> List[ModuleSpec]:
    return [
        ModuleSpec(
            module_id="M1",
            layer="Layer 1",
            name="Sensor adapters",
            inputs=["camera", "microphone", "wearable", "iot", "dashboard events"],
            outputs=["timestamped packets"],
            evidence_level="partial_existing_assets",
        ),
        ModuleSpec(
            module_id="M2",
            layer="Layer 1",
            name="Synchronization engine",
            inputs=["timestamped packets"],
            outputs=["aligned windows", "quality flags"],
            evidence_level="planned_experiment",
        ),
        ModuleSpec(
            module_id="M3",
            layer="Layer 2",
            name="Feature and embedding extractors",
            inputs=["aligned windows"],
            outputs=["modality embeddings"],
            evidence_level="implemented_real_baseline plus planned",
        ),
        ModuleSpec(
            module_id="M4",
            layer="Layer 3",
            name="Emotion and health task heads",
            inputs=["modality embeddings", "digital twin context"],
            outputs=["emotion", "risk", "anomaly", "adherence"],
            evidence_level="baseline implemented for MER only",
        ),
        ModuleSpec(
            module_id="M5",
            layer="Layer 3",
            name="Medication adherence reasoner",
            inputs=["medication logs", "behavior", "dialogue context"],
            outputs=["adherence state", "missed-dose reason"],
            evidence_level="planned_experiment",
        ),
        ModuleSpec(
            module_id="M6",
            layer="Layer 4",
            name="Digital twin state manager",
            inputs=["predictions", "ros2 state", "patient profile"],
            outputs=["patient twin state"],
            evidence_level="simulation_based_evaluation",
        ),
        ModuleSpec(
            module_id="M7",
            layer="Layer 4",
            name="Knowledge graph plus LLM explainer",
            inputs=["patient twin state", "predictions", "care graph"],
            outputs=["evidence-linked explanation"],
            evidence_level="planned_experiment",
        ),
        ModuleSpec(
            module_id="M8",
            layer="Layer 5",
            name="HITL dashboard and telepresence",
            inputs=["explanation bundles", "alerts"],
            outputs=["override actions", "telepresence sessions"],
            evidence_level="planned_experiment",
        ),
        ModuleSpec(
            module_id="M9",
            layer="Layer 5",
            name="Privacy and deployment controller",
            inputs=["policy", "device state", "consent state"],
            outputs=["routing decisions", "audit records"],
            evidence_level="simulation_based_evaluation",
        ),
    ]
