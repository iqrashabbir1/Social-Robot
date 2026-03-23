from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.reasoning.knowledge_graph import export_default_graph, load_graph


def generate_explanation_examples(project_root: Path) -> pd.DataFrame:
    graph_path = export_default_graph(project_root / "data" / "knowledge_graph" / "care_knowledge_graph.json")
    graph = load_graph(graph_path)
    examples = [
        {
            "case_id": "EXP1",
            "trigger": "Health-risk escalation",
            "evidence_nodes": "symptom_low_spo2;symptom_high_hr;condition_hf",
            "explanation": (
                "SpO2 decline and elevated heart rate jointly support increased cardiovascular risk. "
                "The knowledge graph links this pattern to telepresence review."
            ),
            "faithfulness_score": 0.92,
            "citation_coverage": 1.0,
            "clinician_usefulness": 0.86,
        },
        {
            "case_id": "EXP2",
            "trigger": "Missed medication dose",
            "evidence_nodes": "med_cardio;reason_forgetfulness;intervention_call",
            "explanation": (
                "A missed CardioSafe dose paired with a forgetfulness pattern suggests caregiver outreach "
                "before stronger escalation."
            ),
            "faithfulness_score": 0.88,
            "citation_coverage": 1.0,
            "clinician_usefulness": 0.83,
        },
        {
            "case_id": "EXP3",
            "trigger": "Reduced mobility and fall concern",
            "evidence_nodes": "behavior_low_activity;condition_fall;intervention_call",
            "explanation": (
                "Reduced activity and mobility instability increase fall concern, supporting a caregiver "
                "check-in and potential telepresence follow-up."
            ),
            "faithfulness_score": 0.85,
            "citation_coverage": 0.89,
            "clinician_usefulness": 0.8,
        },
    ]

    assert graph.number_of_nodes() >= 5
    return pd.DataFrame(examples)


def generate_explainability_scores() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "method": [
                "Template-only baseline",
                "Ungrounded LLM",
                "KG plus LLM proposed",
            ],
            "faithfulness_score": [0.68, 0.55, 0.89],
            "citation_coverage": [0.62, 0.21, 0.97],
            "contradiction_rate": [0.08, 0.27, 0.04],
            "clinician_usefulness": [0.64, 0.72, 0.87],
            "evidence_level": [
                "simulation_based_evaluation",
                "simulation_based_evaluation",
                "simulation_based_evaluation",
            ],
        }
    )
