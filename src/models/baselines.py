from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class ApproachSpec:
    approach_id: str
    name: str
    archetype: str
    evidence_level: str
    strengths: List[str]
    limitations: List[str]


def get_approach_registry() -> List[ApproachSpec]:
    return [
        ApproachSpec(
            approach_id="A1",
            name="Classical wearable monitoring plus shallow ML risk prediction",
            archetype="physiology-first baseline",
            evidence_level="literature_aligned_baseline",
            strengths=["simple", "interpretable", "low compute"],
            limitations=["limited affect awareness", "limited interaction reasoning"],
        ),
        ApproachSpec(
            approach_id="A2",
            name="Medication-management robot only",
            archetype="adherence-only assistant",
            evidence_level="literature_aligned_baseline",
            strengths=["strong reminder workflow", "clear user value"],
            limitations=["weak health prediction", "weak multimodal cognition"],
        ),
        ApproachSpec(
            approach_id="A3",
            name="ROS2 or digital-twin robot without full cognition stack",
            archetype="simulation-first assistant",
            evidence_level="literature_aligned_baseline",
            strengths=["strong reproducibility", "good integration path"],
            limitations=["limited end-to-end clinical reasoning"],
        ),
        ApproachSpec(
            approach_id="A4",
            name="IoT-edge healthcare monitoring without social reasoning",
            archetype="deployment-first monitoring",
            evidence_level="literature_aligned_baseline",
            strengths=["good edge fit", "good monitoring coverage"],
            limitations=["limited explanation", "limited human-robot interaction"],
        ),
        ApproachSpec(
            approach_id="A5",
            name="Multimodal emotion recognition system only",
            archetype="affect-only benchmark",
            evidence_level="implemented_real_baseline_reference",
            strengths=["good interaction signal", "strong benchmark value"],
            limitations=["no care loop", "no risk prediction"],
        ),
        ApproachSpec(
            approach_id="A6",
            name="LLM-enabled socially assistive robot dialogue system",
            archetype="dialogue-first assistant",
            evidence_level="literature_aligned_baseline",
            strengths=["natural interaction", "telepresence-friendly"],
            limitations=["weak grounding", "privacy and safety concerns"],
        ),
        ApproachSpec(
            approach_id="A7",
            name="Explainable KG and HITL healthcare robot",
            archetype="reasoning-first assistant",
            evidence_level="literature_aligned_baseline",
            strengths=["strong explainability", "good safety governance"],
            limitations=["partial sensing integration", "partial deployment realism"],
        ),
        ApproachSpec(
            approach_id="A8",
            name="Proposed integrated caregiving system",
            archetype="full-stack cognitive caregiving robot",
            evidence_level="research_target",
            strengths=["closed-loop care support", "digital twin", "privacy and HITL aware"],
            limitations=["requires staged validation", "higher system complexity"],
        ),
    ]
