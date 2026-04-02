from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


VIDEO_PROTOTYPES = {
    "happy": np.array([1.5, 1.2, 0.4, 0.9, 1.0, 0.8, 0.5, 1.1]),
    "sad": np.array([-1.2, -0.8, 0.7, -0.9, -1.0, -0.6, 0.3, -0.7]),
    "neutral": np.array([0.1, 0.0, 0.2, 0.1, 0.1, 0.0, 0.2, 0.1]),
    "fear": np.array([0.8, -0.2, 1.3, 0.7, -0.4, 1.0, 0.6, -0.2]),
}


def load_visual_baseline_log(project_root: Path) -> pd.DataFrame:
    return pd.read_csv(project_root / "tests" / "emotion_log_labeled.csv")


def generate_video_features(labels: list[str], rng: np.random.Generator) -> np.ndarray:
    return np.vstack([VIDEO_PROTOTYPES[label] + rng.normal(0.0, 0.45, 8) for label in labels])
