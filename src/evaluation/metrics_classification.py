from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    macro_f1: float
    weighted_f1: float
    unweighted_recall: float


def compute_metrics(y_true: list[str], y_pred: list[str]) -> ClassificationMetrics:
    return ClassificationMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, average="macro")),
        weighted_f1=float(f1_score(y_true, y_pred, average="weighted")),
        unweighted_recall=float(recall_score(y_true, y_pred, average="macro")),
    )


def confusion_dataframe(y_true: list[str], y_pred: list[str], labels: list[str]) -> pd.DataFrame:
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(matrix, index=labels, columns=labels).reset_index().rename(columns={"index": "true_label"})
