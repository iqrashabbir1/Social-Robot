from __future__ import annotations

import json
from pathlib import Path

import networkx as nx


DEFAULT_GRAPH = {
    "nodes": [
        {"id": "patient_001", "type": "patient", "label": "Elderly homecare resident"},
        {"id": "condition_hf", "type": "condition", "label": "Heart failure risk"},
        {"id": "condition_fall", "type": "condition", "label": "Fall risk"},
        {"id": "med_cardio", "type": "medication", "label": "CardioSafe-10mg"},
        {"id": "symptom_low_spo2", "type": "signal", "label": "SpO2 decline"},
        {"id": "symptom_high_hr", "type": "signal", "label": "Heart rate elevation"},
        {"id": "behavior_low_activity", "type": "behavior", "label": "Reduced activity"},
        {"id": "reason_forgetfulness", "type": "adherence_reason", "label": "Forgetfulness"},
        {"id": "reason_side_effects", "type": "adherence_reason", "label": "Side effects"},
        {"id": "intervention_call", "type": "intervention", "label": "Caregiver call"},
        {"id": "intervention_telepresence", "type": "intervention", "label": "Telepresence review"},
    ],
    "edges": [
        {"source": "patient_001", "target": "condition_hf", "relation": "monitored_for"},
        {"source": "patient_001", "target": "condition_fall", "relation": "monitored_for"},
        {"source": "patient_001", "target": "med_cardio", "relation": "prescribed"},
        {"source": "symptom_low_spo2", "target": "condition_hf", "relation": "supports_risk"},
        {"source": "symptom_high_hr", "target": "condition_hf", "relation": "supports_risk"},
        {"source": "behavior_low_activity", "target": "condition_fall", "relation": "supports_risk"},
        {"source": "reason_forgetfulness", "target": "intervention_call", "relation": "suggests"},
        {"source": "reason_side_effects", "target": "intervention_telepresence", "relation": "suggests"},
        {"source": "condition_hf", "target": "intervention_telepresence", "relation": "requires_review"},
        {"source": "condition_fall", "target": "intervention_call", "relation": "requires_review"},
    ],
}


def export_default_graph(graph_path: Path) -> Path:
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    graph_path.write_text(json.dumps(DEFAULT_GRAPH, indent=2), encoding="utf-8")
    return graph_path


def load_graph(graph_path: Path) -> nx.DiGraph:
    data = json.loads(graph_path.read_text(encoding="utf-8"))
    graph = nx.DiGraph()
    for node in data["nodes"]:
        graph.add_node(node["id"], **node)
    for edge in data["edges"]:
        graph.add_edge(edge["source"], edge["target"], relation=edge["relation"])
    return graph
