from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Optional


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

try:
    from fusion.fusion_logic import fuse_emotions
    from response.response_rules import get_response_for_emotion
except Exception:
    fuse_emotions = None
    get_response_for_emotion = None


@dataclass
class BaselineMerResult:
    face_emotion: Optional[str]
    speech_emotion: Optional[str]
    fused_emotion: str
    response_text: Optional[str]
    evidence_level: str = "implemented_real_baseline"


def describe_baseline() -> dict[str, object]:
    return {
        "name": "DeepFace plus speech SVM baseline",
        "inputs": ["webcam_frame", "speech_clip"],
        "components": [
            "perception.face_emotion.detect_face_emotion_from_frame",
            "perception.speech_emotion.predict_speech_emotion_robust_3class",
            "fusion.fusion_logic.fuse_emotions",
            "response.response_rules.get_response_for_emotion",
        ],
        "role": "baseline MER model and ablation reference",
        "limitations": [
            "no physiology",
            "no digital twin",
            "no medication reasoning",
            "no explainability layer",
        ],
    }


def fuse_precomputed_predictions(
    face_emotion: Optional[str],
    speech_emotion: Optional[str],
) -> BaselineMerResult:
    if fuse_emotions is None:
        fused = "unknown"
    else:
        fused = fuse_emotions(face_emotion, speech_emotion)

    if get_response_for_emotion is None:
        response_text = None
    else:
        response_text = get_response_for_emotion(fused)

    return BaselineMerResult(
        face_emotion=face_emotion,
        speech_emotion=speech_emotion,
        fused_emotion=fused,
        response_text=response_text,
    )
