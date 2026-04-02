from __future__ import annotations

import numpy as np


CONTEXT_PROTOTYPES = {
    "happy": np.array([0.9, 0.2, 0.8, 0.4]),
    "sad": np.array([-0.7, 0.6, -0.5, 0.8]),
    "neutral": np.array([0.1, 0.2, 0.1, 0.1]),
    "fear": np.array([0.5, 0.9, -0.1, 0.7]),
}


def generate_context_features(labels: list[str], rng: np.random.Generator) -> np.ndarray:
    return np.vstack([CONTEXT_PROTOTYPES[label] + rng.normal(0.0, 0.28, 4) for label in labels])
