from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.model_selection import RepeatedStratifiedKFold

from src.evaluation.metrics_classification import compute_metrics


@dataclass(frozen=True)
class CrossValidationSummary:
    mean_accuracy: float
    std_accuracy: float
    accuracy_ci_low: float
    accuracy_ci_high: float
    mean_f1: float
    std_f1: float
    f1_ci_low: float
    f1_ci_high: float
    fold_results: pd.DataFrame
    predictions: pd.DataFrame


def compute_bootstrap_ci(
    values: list[float] | np.ndarray,
    *,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    random_seed: int = 42,
) -> dict[str, float]:
    samples = np.asarray(values, dtype=np.float64)
    if samples.size == 0:
        raise ValueError("Cannot compute bootstrap intervals on an empty array.")

    rng = np.random.default_rng(random_seed)
    resampled_means = np.empty(n_resamples, dtype=np.float64)
    for index in range(n_resamples):
        bootstrap_sample = rng.choice(samples, size=samples.size, replace=True)
        resampled_means[index] = float(np.mean(bootstrap_sample))

    alpha = 1.0 - confidence
    lower = float(np.quantile(resampled_means, alpha / 2.0))
    upper = float(np.quantile(resampled_means, 1.0 - alpha / 2.0))
    return {
        "mean": float(np.mean(samples)),
        "lower": lower,
        "upper": upper,
    }


def run_repeated_cross_validation(
    *,
    labels: list[str] | np.ndarray,
    fit_predict_callback: Callable[[np.ndarray, np.ndarray, int, int], dict[str, Any]],
    sample_ids: list[str] | np.ndarray | None = None,
    n_splits: int = 5,
    n_repeats: int = 10,
    random_seed: int = 42,
) -> CrossValidationSummary:
    y = np.asarray(labels)
    if y.size == 0:
        raise ValueError("Repeated cross-validation requires at least one labeled sample.")
    if sample_ids is None:
        sample_ids_array = np.asarray([f"sample_{index:06d}" for index in range(y.size)])
    else:
        sample_ids_array = np.asarray(sample_ids)

    splitter = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_seed,
    )

    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    dummy_x = np.zeros((y.size, 1), dtype=np.float32)

    for fold_index, (train_idx, test_idx) in enumerate(splitter.split(dummy_x, y), start=1):
        repeat_index = (fold_index - 1) // n_splits + 1
        split_index = (fold_index - 1) % n_splits + 1
        callback_result = fit_predict_callback(train_idx, test_idx, repeat_index, split_index)

        probabilities = np.asarray(callback_result["probabilities"], dtype=np.float64)
        if probabilities.ndim != 2:
            raise ValueError("fit_predict_callback must return a 2-D probability matrix.")
        class_labels = list(callback_result["class_labels"])
        predicted_indices = probabilities.argmax(axis=1)
        predicted_labels = [class_labels[int(index)] for index in predicted_indices]
        true_labels = y[test_idx].tolist()
        metrics = compute_metrics(true_labels, predicted_labels)

        fold_rows.append(
            {
                "repeat": repeat_index,
                "fold": split_index,
                "fold_index": fold_index,
                "num_train": int(len(train_idx)),
                "num_test": int(len(test_idx)),
                "accuracy": round(float(metrics.accuracy), 6),
                "macro_f1": round(float(metrics.macro_f1), 6),
                "weighted_f1": round(float(metrics.weighted_f1), 6),
                "unweighted_recall": round(float(metrics.unweighted_recall), 6),
                **{key: value for key, value in callback_result.items() if key not in {"probabilities", "class_labels"}},
            }
        )

        for local_index, sample_index in enumerate(test_idx):
            probability_row = probabilities[local_index]
            prediction_rows.append(
                {
                    "sample_id": str(sample_ids_array[sample_index]),
                    "true_label": str(y[sample_index]),
                    "predicted_label": predicted_labels[local_index],
                    "confidence": round(float(np.max(probability_row)), 6),
                    "repeat": repeat_index,
                    "fold": split_index,
                    "fold_index": fold_index,
                    **{f"prob_{label}": round(float(probability_row[idx]), 6) for idx, label in enumerate(class_labels)},
                }
            )

    fold_df = pd.DataFrame(fold_rows)
    prediction_df = pd.DataFrame(prediction_rows)
    accuracy_ci = compute_bootstrap_ci(fold_df["accuracy"].to_numpy(), n_resamples=10_000, confidence=0.95, random_seed=random_seed)
    f1_ci = compute_bootstrap_ci(fold_df["macro_f1"].to_numpy(), n_resamples=10_000, confidence=0.95, random_seed=random_seed)
    return CrossValidationSummary(
        mean_accuracy=float(fold_df["accuracy"].mean()),
        std_accuracy=float(fold_df["accuracy"].std(ddof=1)),
        accuracy_ci_low=float(accuracy_ci["lower"]),
        accuracy_ci_high=float(accuracy_ci["upper"]),
        mean_f1=float(fold_df["macro_f1"].mean()),
        std_f1=float(fold_df["macro_f1"].std(ddof=1)),
        f1_ci_low=float(f1_ci["lower"]),
        f1_ci_high=float(f1_ci["upper"]),
        fold_results=fold_df,
        predictions=prediction_df,
    )


def compare_models_statistically(
    baseline_scores: list[float] | np.ndarray,
    candidate_scores: list[float] | np.ndarray,
) -> dict[str, float | str]:
    baseline = np.asarray(baseline_scores, dtype=np.float64)
    candidate = np.asarray(candidate_scores, dtype=np.float64)
    if baseline.shape != candidate.shape:
        raise ValueError("Paired statistical comparison requires arrays of identical shape.")
    differences = candidate - baseline
    if np.allclose(differences, 0.0):
        p_value = 1.0
    else:
        try:
            _statistic, p_value = wilcoxon(candidate, baseline, alternative="two-sided", zero_method="wilcox")
        except ValueError:
            p_value = 1.0

    std_diff = float(np.std(differences, ddof=1)) if differences.size > 1 else 0.0
    cohen_d = 0.0 if std_diff == 0.0 else float(np.mean(differences) / std_diff)
    magnitude = abs(cohen_d)
    if magnitude < 0.2:
        interpretation = "negligible"
    elif magnitude < 0.5:
        interpretation = "small"
    elif magnitude < 0.8:
        interpretation = "medium"
    else:
        interpretation = "large"

    return {
        "p_value": float(p_value),
        "cohens_d": float(cohen_d),
        "effect_interpretation": interpretation,
    }
