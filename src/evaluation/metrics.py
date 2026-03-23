from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import sqrt
from statistics import mean
from typing import Iterable, Sequence


@dataclass(frozen=True)
class ClassificationSummary:
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def confusion_counts(
    y_true: Sequence[str],
    y_pred: Sequence[str],
) -> dict[str, Counter]:
    labels = sorted(set(y_true) | set(y_pred))
    counts = {label: Counter() for label in labels}
    for true_label, pred_label in zip(y_true, y_pred):
        if true_label == pred_label:
            counts[true_label]["tp"] += 1
        else:
            counts[true_label]["fn"] += 1
            counts[pred_label]["fp"] += 1
    return counts


def classification_summary(
    y_true: Sequence[str],
    y_pred: Sequence[str],
) -> ClassificationSummary:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if not y_true:
        raise ValueError("y_true and y_pred must be non-empty")

    counts = confusion_counts(y_true, y_pred)
    accuracy = _safe_div(sum(t == p for t, p in zip(y_true, y_pred)), len(y_true))

    precisions = []
    recalls = []
    f1s = []
    for label_counts in counts.values():
        tp = label_counts["tp"]
        fp = label_counts["fp"]
        fn = label_counts["fn"]
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return ClassificationSummary(
        accuracy=accuracy,
        macro_precision=mean(precisions),
        macro_recall=mean(recalls),
        macro_f1=mean(f1s),
    )


def expected_calibration_error(
    confidences: Sequence[float],
    correctness: Sequence[int],
    bins: int = 10,
) -> float:
    if len(confidences) != len(correctness):
        raise ValueError("confidences and correctness must have the same length")
    if not confidences:
        return 0.0

    bin_totals = [0] * bins
    bin_conf = [0.0] * bins
    bin_acc = [0.0] * bins

    for confidence, is_correct in zip(confidences, correctness):
        index = min(int(confidence * bins), bins - 1)
        bin_totals[index] += 1
        bin_conf[index] += confidence
        bin_acc[index] += is_correct

    total = len(confidences)
    ece = 0.0
    for idx in range(bins):
        if bin_totals[idx] == 0:
            continue
        avg_conf = bin_conf[idx] / bin_totals[idx]
        avg_acc = bin_acc[idx] / bin_totals[idx]
        ece += (bin_totals[idx] / total) * abs(avg_acc - avg_conf)
    return ece


def latency_summary(latencies_ms: Iterable[float]) -> dict[str, float]:
    values = list(latencies_ms)
    if not values:
        return {"mean_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
    values_sorted = sorted(values)
    p95_index = max(int(round(0.95 * (len(values_sorted) - 1))), 0)
    return {
        "mean_ms": mean(values_sorted),
        "p95_ms": values_sorted[p95_index],
        "max_ms": values_sorted[-1],
    }


def brier_score(
    probabilities: Sequence[float],
    outcomes: Sequence[int],
) -> float:
    if len(probabilities) != len(outcomes):
        raise ValueError("probabilities and outcomes must have the same length")
    if not probabilities:
        return 0.0
    return mean((p - y) ** 2 for p, y in zip(probabilities, outcomes))


def rmse(values: Sequence[float], targets: Sequence[float]) -> float:
    if len(values) != len(targets):
        raise ValueError("values and targets must have the same length")
    if not values:
        return 0.0
    return sqrt(mean((value - target) ** 2 for value, target in zip(values, targets)))
