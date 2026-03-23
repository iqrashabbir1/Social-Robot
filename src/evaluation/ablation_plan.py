from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class AblationSpec:
    ablation_id: str
    description: str
    comparison_target: str
    evidence_level: str


def default_ablation_plan() -> List[AblationSpec]:
    return [
        AblationSpec(
            ablation_id="ABL1",
            description="Face-only baseline versus speech-only baseline versus existing rule fusion",
            comparison_target="CS3",
            evidence_level="implemented_real_baseline",
        ),
        AblationSpec(
            ablation_id="ABL2",
            description="Late fusion versus cross-attention transformer for MER",
            comparison_target="CS3",
            evidence_level="planned_experiment",
        ),
        AblationSpec(
            ablation_id="ABL3",
            description="Health-risk prediction with and without digital-twin context",
            comparison_target="CS4",
            evidence_level="simulation_based_evaluation",
        ),
        AblationSpec(
            ablation_id="ABL4",
            description="Adherence reasoning with and without missed-dose reason modeling",
            comparison_target="CS5",
            evidence_level="planned_experiment",
        ),
        AblationSpec(
            ablation_id="ABL5",
            description="LLM explanations with and without KG grounding",
            comparison_target="CS7",
            evidence_level="planned_experiment",
        ),
        AblationSpec(
            ablation_id="ABL6",
            description="Edge-only versus hybrid routing under privacy constraints",
            comparison_target="CS8",
            evidence_level="simulation_based_evaluation",
        ),
    ]
