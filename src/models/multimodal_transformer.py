from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List


@dataclass(frozen=True)
class MultimodalTransformerConfig:
    visual_dim: int = 256
    audio_dim: int = 128
    physiological_dim: int = 128
    behavior_dim: int = 64
    medication_dim: int = 32
    hidden_dim: int = 384
    num_heads: int = 6
    num_layers: int = 4
    dropout: float = 0.1
    tasks: tuple[str, ...] = (
        "emotion",
        "health_risk",
        "anomaly",
        "adherence",
    )


def build_architecture_summary(
    config: MultimodalTransformerConfig | None = None,
) -> Dict[str, object]:
    config = config or MultimodalTransformerConfig()
    return {
        "encoder_blocks": [
            "visual_encoder",
            "audio_encoder",
            "physiology_encoder",
            "behavior_encoder",
            "medication_encoder",
        ],
        "fusion_strategy": "cross_attention_transformer",
        "task_heads": list(config.tasks),
        "explainability_hooks": [
            "attention_rollup",
            "knowledge_graph_retrieval",
            "llm_explanation_conditioning",
        ],
        "deployment_modes": ["edge", "hybrid", "cloud_assisted"],
        "config": asdict(config),
    }


def describe_training_plan() -> List[str]:
    return [
        "Pretrain or warm start unimodal encoders where public data exist.",
        "Align modalities with synchronized windows and contrastive consistency.",
        "Train task heads with uncertainty and calibration objectives.",
        "Evaluate missing-modality robustness and latency-aware inference routing.",
    ]
