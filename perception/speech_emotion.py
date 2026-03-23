import os
import sys
import numpy as np
import joblib
from typing import Tuple, Dict

CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from perception.speech_features import extract_mfcc_features

# New SVM model trained on balanced CREMA-D with 26-dim MFCC stats
MODEL_PATH = os.path.join(PROJECT_ROOT, "data", "speech", "speech_svm_crema_balanced.joblib")

_model = None

def load_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
        _model = joblib.load(MODEL_PATH)
    return _model


def predict_speech_emotion_raw(wav_path: str) -> str:
    """
    Raw 3-class prediction via SVM (argmax of probabilities).
    Returns: 'positive', 'neutral', or 'negative'
    """
    model = load_model()
    feat_vec = extract_mfcc_features(wav_path)
    X = np.expand_dims(feat_vec, axis=0)
    pred = model.predict(X)[0]
    return str(pred)


def predict_speech_emotion_with_proba(wav_path: str) -> Tuple[str, Dict[str, float]]:
    """
    Return (label, proba_dict) where proba_dict[label] = probability.
    """
    model = load_model()
    feat_vec = extract_mfcc_features(wav_path)
    X = np.expand_dims(feat_vec, axis=0)

    proba = model.predict_proba(X)[0]
    classes = model.classes_

    label_idx = int(np.argmax(proba))
    label = str(classes[label_idx])
    proba_dict = {str(cls): float(p) for cls, p in zip(classes, proba)}

    return label, proba_dict


def predict_speech_emotion_robust_3class(
    wav_path: str,
    pos_threshold: float = 0.45,
    neg_threshold: float = 0.45
) -> str:
    """
    Robust 3-class decision:
      - If P(positive) >= pos_threshold and >= P(negative), P(neutral) -> 'positive'
      - Elif P(negative) >= neg_threshold and >= P(positive), P(neutral) -> 'negative'
      - Else -> 'neutral'
    """
    label, proba = predict_speech_emotion_with_proba(wav_path)

    p_pos = proba.get("positive", 0.0)
    p_neu = proba.get("neutral", 0.0)
    p_neg = proba.get("negative", 0.0)

    if p_pos >= pos_threshold and p_pos >= p_neg and p_pos >= p_neu:
        return "positive"

    if p_neg >= neg_threshold and p_neg >= p_pos and p_neg >= p_neu:
        return "negative"

    return "neutral"
