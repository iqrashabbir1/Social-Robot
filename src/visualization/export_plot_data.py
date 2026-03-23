from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

from src.pipelines.caregiving_stack import get_module_specs


APPROACH_ROWS = [
    {
        "approach_id": "A1",
        "approach_name": "Classical wearable monitoring plus shallow ML risk prediction",
        "category": "physiology-first baseline",
        "evidence_type": "literature_aligned",
        "sensing_modalities": "physiology, wearable activity",
        "predictive_health_risk_capability": 3,
        "emotion_awareness": 0,
        "medication_adherence_support": 0,
        "ros2_digital_twin_support": 0,
        "iot_edge_deployment": 1,
        "explainability": 1,
        "hitl_safety_governance": 1,
        "privacy_awareness": 1,
        "telepresence_cultural_adaptation": 0,
        "real_world_readiness": 2,
        "expected_computational_cost": "low",
        "expected_operational_usefulness": "moderate",
    },
    {
        "approach_id": "A2",
        "approach_name": "Medication-management robot only",
        "category": "adherence-only assistant",
        "evidence_type": "literature_aligned",
        "sensing_modalities": "schedule, reminders, user prompts",
        "predictive_health_risk_capability": 1,
        "emotion_awareness": 0,
        "medication_adherence_support": 3,
        "ros2_digital_twin_support": 1,
        "iot_edge_deployment": 1,
        "explainability": 1,
        "hitl_safety_governance": 1,
        "privacy_awareness": 1,
        "telepresence_cultural_adaptation": 1,
        "real_world_readiness": 2,
        "expected_computational_cost": "medium",
        "expected_operational_usefulness": "moderate",
    },
    {
        "approach_id": "A3",
        "approach_name": "ROS2 or digital-twin robot without full cognition stack",
        "category": "simulation-first assistant",
        "evidence_type": "literature_aligned",
        "sensing_modalities": "robot state, simulated sensors",
        "predictive_health_risk_capability": 1,
        "emotion_awareness": 1,
        "medication_adherence_support": 1,
        "ros2_digital_twin_support": 3,
        "iot_edge_deployment": 1,
        "explainability": 0,
        "hitl_safety_governance": 1,
        "privacy_awareness": 1,
        "telepresence_cultural_adaptation": 1,
        "real_world_readiness": 2,
        "expected_computational_cost": "high",
        "expected_operational_usefulness": "moderate",
    },
    {
        "approach_id": "A4",
        "approach_name": "IoT-edge healthcare monitoring without social reasoning",
        "category": "deployment-first monitoring",
        "evidence_type": "literature_aligned",
        "sensing_modalities": "iot, wearables, edge analytics",
        "predictive_health_risk_capability": 2,
        "emotion_awareness": 1,
        "medication_adherence_support": 1,
        "ros2_digital_twin_support": 0,
        "iot_edge_deployment": 3,
        "explainability": 0,
        "hitl_safety_governance": 1,
        "privacy_awareness": 2,
        "telepresence_cultural_adaptation": 0,
        "real_world_readiness": 2,
        "expected_computational_cost": "medium",
        "expected_operational_usefulness": "moderate_high",
    },
    {
        "approach_id": "A5",
        "approach_name": "Multimodal emotion recognition system only",
        "category": "affect-only benchmark",
        "evidence_type": "implemented_real_baseline_reference",
        "sensing_modalities": "vision, speech",
        "predictive_health_risk_capability": 0,
        "emotion_awareness": 3,
        "medication_adherence_support": 0,
        "ros2_digital_twin_support": 0,
        "iot_edge_deployment": 1,
        "explainability": 1,
        "hitl_safety_governance": 0,
        "privacy_awareness": 1,
        "telepresence_cultural_adaptation": 1,
        "real_world_readiness": 1,
        "expected_computational_cost": "medium",
        "expected_operational_usefulness": "narrow",
    },
    {
        "approach_id": "A6",
        "approach_name": "LLM-enabled socially assistive robot dialogue system",
        "category": "dialogue-first assistant",
        "evidence_type": "literature_aligned",
        "sensing_modalities": "dialogue, optional speech, user context",
        "predictive_health_risk_capability": 1,
        "emotion_awareness": 2,
        "medication_adherence_support": 2,
        "ros2_digital_twin_support": 1,
        "iot_edge_deployment": 1,
        "explainability": 1,
        "hitl_safety_governance": 1,
        "privacy_awareness": 0,
        "telepresence_cultural_adaptation": 2,
        "real_world_readiness": 1,
        "expected_computational_cost": "high",
        "expected_operational_usefulness": "moderate",
    },
    {
        "approach_id": "A7",
        "approach_name": "Explainable KG and HITL healthcare robot",
        "category": "reasoning-first assistant",
        "evidence_type": "literature_aligned",
        "sensing_modalities": "health records, selected sensors, clinician inputs",
        "predictive_health_risk_capability": 2,
        "emotion_awareness": 1,
        "medication_adherence_support": 2,
        "ros2_digital_twin_support": 1,
        "iot_edge_deployment": 1,
        "explainability": 3,
        "hitl_safety_governance": 3,
        "privacy_awareness": 2,
        "telepresence_cultural_adaptation": 1,
        "real_world_readiness": 2,
        "expected_computational_cost": "high",
        "expected_operational_usefulness": "high",
    },
    {
        "approach_id": "A8",
        "approach_name": "Proposed integrated caregiving system",
        "category": "full-stack cognitive caregiving robot",
        "evidence_type": "research_target",
        "sensing_modalities": "vision, speech, physiology, behavior, medication, ros2, dashboard",
        "predictive_health_risk_capability": 3,
        "emotion_awareness": 3,
        "medication_adherence_support": 3,
        "ros2_digital_twin_support": 3,
        "iot_edge_deployment": 3,
        "explainability": 3,
        "hitl_safety_governance": 3,
        "privacy_awareness": 3,
        "telepresence_cultural_adaptation": 3,
        "real_world_readiness": 3,
        "expected_computational_cost": "very_high",
        "expected_operational_usefulness": "very_high",
    },
]


CASE_STUDY_SUMMARIES = [
    {
        "case_study_id": "CS1",
        "title": "ROS2 plus digital twin validation",
        "layer_focus": "Layers 1, 4, 5",
        "evidence_status": "simulation_based_evaluation",
        "existing_assets": "ros2_ws directory, baseline modules",
        "reusable_inputs": "simulated topics, baseline emotion outputs",
        "baselines": "A3, A8",
        "figure_source": "outputs/tables/case_study_1_summary.csv",
    },
    {
        "case_study_id": "CS2",
        "title": "Multimodal sensing and synchronization",
        "layer_focus": "Layers 1, 2",
        "evidence_status": "simulation_based_evaluation",
        "existing_assets": "vision and speech logs",
        "reusable_inputs": "webcam and audio demo assets",
        "baselines": "A1, A5, A8",
        "figure_source": "outputs/tables/case_study_2_summary.csv",
    },
    {
        "case_study_id": "CS3",
        "title": "MER benchmark",
        "layer_focus": "Layers 2, 3",
        "evidence_status": "implemented_real_baseline plus planned extensions",
        "existing_assets": "speech datasets, DeepFace baseline",
        "reusable_inputs": "CREMA-D, user SER, webcam baseline",
        "baselines": "A5, A8",
        "figure_source": "outputs/tables/case_study_3_summary.csv",
    },
    {
        "case_study_id": "CS4",
        "title": "Health-risk prediction and anomaly detection",
        "layer_focus": "Layers 2, 3, 4",
        "evidence_status": "simulation_based_evaluation",
        "existing_assets": "no direct physiology pipeline yet",
        "reusable_inputs": "synthetic vitals and activity traces",
        "baselines": "A1, A4, A8",
        "figure_source": "outputs/tables/case_study_4_summary.csv",
    },
    {
        "case_study_id": "CS5",
        "title": "Medication adherence reasoning",
        "layer_focus": "Layers 3, 4, 5",
        "evidence_status": "planned_experiment",
        "existing_assets": "response rules and dialogue hooks",
        "reusable_inputs": "synthetic dispenser logs",
        "baselines": "A2, A6, A8",
        "figure_source": "outputs/tables/case_study_5_summary.csv",
    },
    {
        "case_study_id": "CS6",
        "title": "HITL dashboard and override safety",
        "layer_focus": "Layers 4, 5",
        "evidence_status": "planned_experiment",
        "existing_assets": "response layer and test logs",
        "reusable_inputs": "simulated alert bundles",
        "baselines": "A7, A8",
        "figure_source": "outputs/tables/case_study_6_summary.csv",
    },
    {
        "case_study_id": "CS7",
        "title": "KG plus LLM explainability",
        "layer_focus": "Layer 4",
        "evidence_status": "planned_experiment",
        "existing_assets": "no KG yet",
        "reusable_inputs": "synthetic patient graph and predictions",
        "baselines": "A6, A7, A8",
        "figure_source": "outputs/tables/case_study_7_summary.csv",
    },
    {
        "case_study_id": "CS8",
        "title": "Privacy, edge, telepresence end-to-end scenario",
        "layer_focus": "Layers 1 through 5",
        "evidence_status": "simulation_based_evaluation",
        "existing_assets": "baseline multimodal loop",
        "reusable_inputs": "simulated compute and bandwidth profiles",
        "baselines": "A4, A6, A8",
        "figure_source": "outputs/tables/case_study_8_summary.csv",
    },
]


def _write_rows(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def export_all_csv_artifacts(project_root: str | Path) -> None:
    project_root = Path(project_root)
    outputs_tables = project_root / "outputs" / "tables"
    outputs_csv = project_root / "outputs" / "csv"

    _write_rows(outputs_tables / "literature_comparison_matrix.csv", APPROACH_ROWS)

    module_rows = [
        {
            "module_id": spec.module_id,
            "layer": spec.layer,
            "module_name": spec.name,
            "inputs": "; ".join(spec.inputs),
            "outputs": "; ".join(spec.outputs),
            "evidence_level": spec.evidence_level,
        }
        for spec in get_module_specs()
    ]
    _write_rows(outputs_csv / "system_module_map.csv", module_rows)
    _write_rows(outputs_csv / "case_study_registry.csv", CASE_STUDY_SUMMARIES)
