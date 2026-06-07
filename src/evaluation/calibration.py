from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common.io_utils import ensure_parent


def compute_ece(
    probabilities: np.ndarray,
    true_labels: list[str] | np.ndarray,
    class_labels: list[str],
    n_bins: int = 15,
) -> float:
    probs = np.asarray(probabilities, dtype=np.float64)
    truths = np.asarray(true_labels)
    predicted_indices = probs.argmax(axis=1)
    confidences = probs.max(axis=1)
    predicted_labels = np.asarray([class_labels[int(index)] for index in predicted_indices])
    correctness = (predicted_labels == truths).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for bin_index in range(n_bins):
        lower, upper = bin_edges[bin_index], bin_edges[bin_index + 1]
        if bin_index == n_bins - 1:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences >= lower) & (confidences < upper)
        if not np.any(mask):
            continue
        bin_accuracy = float(correctness[mask].mean())
        bin_confidence = float(confidences[mask].mean())
        ece += (float(mask.mean()) * abs(bin_accuracy - bin_confidence))
    return float(ece)


def compute_mce(
    probabilities: np.ndarray,
    true_labels: list[str] | np.ndarray,
    class_labels: list[str],
    n_bins: int = 15,
) -> float:
    probs = np.asarray(probabilities, dtype=np.float64)
    truths = np.asarray(true_labels)
    predicted_indices = probs.argmax(axis=1)
    confidences = probs.max(axis=1)
    predicted_labels = np.asarray([class_labels[int(index)] for index in predicted_indices])
    correctness = (predicted_labels == truths).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    max_gap = 0.0
    for bin_index in range(n_bins):
        lower, upper = bin_edges[bin_index], bin_edges[bin_index + 1]
        if bin_index == n_bins - 1:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences >= lower) & (confidences < upper)
        if not np.any(mask):
            continue
        bin_accuracy = float(correctness[mask].mean())
        bin_confidence = float(confidences[mask].mean())
        max_gap = max(max_gap, abs(bin_accuracy - bin_confidence))
    return float(max_gap)


def plot_reliability_diagram(
    probabilities: np.ndarray,
    true_labels: list[str] | np.ndarray,
    class_labels: list[str],
    output_path: Path,
    n_bins: int = 15,
    n_bootstrap: int = 1000,
    random_seed: int = 42,
    title: str = "Reliability Diagram",
) -> pd.DataFrame:
    probs = np.asarray(probabilities, dtype=np.float64)
    truths = np.asarray(true_labels)
    predicted_indices = probs.argmax(axis=1)
    confidences = probs.max(axis=1)
    predicted_labels = np.asarray([class_labels[int(index)] for index in predicted_indices])
    correctness = (predicted_labels == truths).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    rows: list[dict[str, float]] = []

    rng = np.random.default_rng(random_seed)
    for bin_index in range(n_bins):
        lower, upper = bin_edges[bin_index], bin_edges[bin_index + 1]
        if bin_index == n_bins - 1:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences >= lower) & (confidences < upper)
        if not np.any(mask):
            rows.append(
                {
                    "bin_index": bin_index,
                    "bin_center": float(centers[bin_index]),
                    "bin_lower": float(lower),
                    "bin_upper": float(upper),
                    "bin_count": 0,
                    "accuracy": np.nan,
                    "confidence": np.nan,
                    "accuracy_ci_low": np.nan,
                    "accuracy_ci_high": np.nan,
                }
            )
            continue
        bin_correctness = correctness[mask]
        bin_confidences = confidences[mask]
        bootstrap_accuracies = np.empty(n_bootstrap, dtype=np.float64)
        for sample_index in range(n_bootstrap):
            draw = rng.choice(bin_correctness, size=bin_correctness.size, replace=True)
            bootstrap_accuracies[sample_index] = float(draw.mean())
        rows.append(
            {
                "bin_index": bin_index,
                "bin_center": float(centers[bin_index]),
                "bin_lower": float(lower),
                "bin_upper": float(upper),
                "bin_count": int(mask.sum()),
                "accuracy": float(bin_correctness.mean()),
                "confidence": float(bin_confidences.mean()),
                "accuracy_ci_low": float(np.quantile(bootstrap_accuracies, 0.025)),
                "accuracy_ci_high": float(np.quantile(bootstrap_accuracies, 0.975)),
            }
        )

    calibration_df = pd.DataFrame(rows)
    ensure_parent(output_path)
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    valid_df = calibration_df.loc[calibration_df["bin_count"] > 0].copy()
    ax.plot([0, 1], [0, 1], linestyle="--", color="#666666", linewidth=1.2, label="Perfect calibration")
    ax.plot(valid_df["confidence"], valid_df["accuracy"], marker="o", color="#1f77b4", linewidth=2.0, label="Observed accuracy")
    ax.fill_between(
        valid_df["confidence"],
        valid_df["accuracy_ci_low"],
        valid_df["accuracy_ci_high"],
        color="#1f77b4",
        alpha=0.20,
        label="95% bootstrap band",
    )
    ax.set_title(title)
    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return calibration_df
