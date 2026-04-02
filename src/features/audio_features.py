from __future__ import annotations

import numpy as np


AUDIO_PROTOTYPES = {
    "happy": np.array([1.1, 0.8, 0.7, 0.6, 1.0, 0.9]),
    "sad": np.array([-1.0, -0.7, 0.4, -0.8, -0.6, 0.3]),
    "neutral": np.array([0.0, 0.1, 0.1, 0.0, 0.1, 0.1]),
    "fear": np.array([0.7, -0.2, 1.1, 0.6, -0.1, 0.9]),
}


def generate_audio_features(labels: list[str], rng: np.random.Generator) -> np.ndarray:
    return np.vstack([AUDIO_PROTOTYPES[label] + rng.normal(0.0, 0.4, 6) for label in labels])
