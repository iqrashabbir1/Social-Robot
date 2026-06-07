from __future__ import annotations

import math

import numpy as np


def add_audio_noise(waveform: np.ndarray, snr_db: float) -> np.ndarray:
    """Add Gaussian noise to an array while approximately matching the requested SNR."""

    signal = np.asarray(waveform, dtype=np.float32)
    if signal.size == 0:
        return signal.copy()

    signal_power = float(np.mean(np.square(signal)))
    if signal_power <= 1e-12:
        return signal.copy()

    snr_linear = 10.0 ** (float(snr_db) / 10.0)
    noise_power = signal_power / max(snr_linear, 1e-6)
    noise = np.random.default_rng().normal(0.0, math.sqrt(noise_power), size=signal.shape).astype(np.float32)
    return (signal + noise).astype(np.float32)


def reduce_brightness(image: np.ndarray, reduction_factor: float) -> np.ndarray:
    """Reduce brightness by scaling intensity while preserving the original dtype range."""

    factor = float(np.clip(reduction_factor, 0.0, 0.99))
    attenuation = 1.0 - factor
    array = np.asarray(image)

    if np.issubdtype(array.dtype, np.integer):
        darkened = np.clip(array.astype(np.float32) * attenuation, 0.0, 255.0)
        return darkened.astype(array.dtype)

    darkened = np.clip(array.astype(np.float32) * attenuation, -1.0, 1.0)
    return darkened.astype(np.float32)


def simulate_sensor_dropout(features: np.ndarray, dropout_prob: float) -> np.ndarray:
    """Zero out sample rows with the requested probability."""

    matrix = np.asarray(features, dtype=np.float32).copy()
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)

    drop_probability = float(np.clip(dropout_prob, 0.0, 1.0))
    if matrix.size == 0 or drop_probability <= 0.0:
        return matrix.astype(np.float32)

    rng = np.random.default_rng()
    drop_mask = rng.random(matrix.shape[0]) < drop_probability
    matrix[drop_mask] = 0.0
    return matrix.astype(np.float32)
