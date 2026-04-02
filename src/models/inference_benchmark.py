from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from src.features.audio_features import generate_audio_features
from src.features.context_features import generate_context_features
from src.features.video_features import load_visual_baseline_log, generate_video_features


LABELS = ["happy", "sad", "neutral", "fear"]


def build_synthetic_multimodal_dataset(
    project_root: Path,
    seed: int,
    n_samples: int,
) -> dict[str, np.ndarray | list[str]]:
    rng = np.random.default_rng(seed)
    baseline = load_visual_baseline_log(project_root)
    sampled = baseline.sample(n=n_samples, replace=True, random_state=seed).reset_index(drop=True)
    labels = sampled["true_emotion"].astype(str).str.lower().tolist()
    return {
        "video": generate_video_features(labels, rng),
        "audio": generate_audio_features(labels, rng),
        "context": generate_context_features(labels, rng),
        "labels": labels,
    }


def split_feature_bundle(bundle: dict[str, np.ndarray | list[str]], test_size: float, seed: int) -> dict[str, object]:
    indices = np.arange(len(bundle["labels"]))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=np.asarray(bundle["labels"]),
    )
    labels = np.asarray(bundle["labels"])
    return {
        "train": {
            "video": bundle["video"][train_idx],
            "audio": bundle["audio"][train_idx],
            "context": bundle["context"][train_idx],
            "labels": labels[train_idx].tolist(),
        },
        "test": {
            "video": bundle["video"][test_idx],
            "audio": bundle["audio"][test_idx],
            "context": bundle["context"][test_idx],
            "labels": labels[test_idx].tolist(),
        },
    }


def assemble_feature_matrix(split_bundle: dict[str, object], modalities: tuple[str, ...]) -> np.ndarray:
    parts = [np.asarray(split_bundle[modality]) for modality in modalities]
    return np.concatenate(parts, axis=1)


def apply_missing_modality(
    split_bundle: dict[str, object],
    modalities: tuple[str, ...],
    drop_probability: float,
    seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    degraded = {key: (value.copy() if hasattr(value, "copy") else value) for key, value in split_bundle.items()}
    for modality in modalities:
        matrix = np.asarray(degraded[modality]).copy()
        drop_mask = rng.random(matrix.shape[0]) < drop_probability
        matrix[drop_mask] = 0.0
        degraded[modality] = matrix
    return degraded


def measure_inference_latency(model, features: np.ndarray, repeats: int = 25) -> float:
    samples = features[: min(len(features), 32)]
    start = time.perf_counter()
    for _ in range(repeats):
        model.predict(samples)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return float(elapsed_ms / repeats)
