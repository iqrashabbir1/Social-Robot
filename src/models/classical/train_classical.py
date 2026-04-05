from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.evaluation.metrics_classification import compute_metrics, confusion_dataframe
from src.models.inference_benchmark import assemble_feature_matrix, measure_inference_latency


@dataclass
class ClassicalResult:
    model: Pipeline
    metrics: dict[str, float]
    confusion: object
    predictions: list[str]


def _build_estimator(model_name: str, seed: int):
    normalized = model_name.strip().lower()
    if normalized == "svm":
        return SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=seed), True
    if normalized == "random_forest":
        return RandomForestClassifier(n_estimators=180, random_state=seed, class_weight="balanced"), False
    if normalized == "logistic_regression":
        return LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            class_weight="balanced",
            random_state=seed,
        ), True
    if normalized == "extra_trees":
        return ExtraTreesClassifier(n_estimators=220, random_state=seed, class_weight="balanced"), False
    if normalized == "gradient_boosting":
        return GradientBoostingClassifier(random_state=seed), False
    raise ValueError(f"Unsupported classical model '{model_name}'.")


def train_and_evaluate_classical(
    train_bundle: dict[str, object],
    test_bundle: dict[str, object],
    modalities: tuple[str, ...],
    labels: list[str],
    model_name: str = "svm",
    seed: int = 42,
) -> ClassicalResult:
    x_train = assemble_feature_matrix(train_bundle, modalities)
    x_test = assemble_feature_matrix(test_bundle, modalities)
    y_train = train_bundle["labels"]
    y_test = test_bundle["labels"]

    estimator, use_scaler = _build_estimator(model_name, seed)
    if use_scaler:
        model = Pipeline([("scaler", StandardScaler()), ("model", estimator)])
    else:
        model = Pipeline([("model", estimator)])

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test).tolist()
    scores = compute_metrics(y_test, y_pred)
    metrics = {
        "accuracy": round(scores.accuracy, 4),
        "macro_f1": round(scores.macro_f1, 4),
        "weighted_f1": round(scores.weighted_f1, 4),
        "uar": round(scores.unweighted_recall, 4),
        "inference_latency_ms": round(measure_inference_latency(model, x_test), 4),
    }
    confusion = confusion_dataframe(y_test, y_pred, labels)
    return ClassicalResult(model=model, metrics=metrics, confusion=confusion, predictions=y_pred)
