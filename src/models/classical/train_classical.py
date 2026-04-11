from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.common.config_loader import build_experiment_context
from src.common.io_utils import write_dataframe, write_json, write_yaml
from src.common.logging_utils import get_logger
from src.common.reproducibility import set_global_seed
from src.data.real_anchor_loader import baseline_predictions_from_frames, resolve_latest_session
from src.evaluation.metrics_classification import compute_metrics, confusion_dataframe
from src.models.inference_benchmark import LABELS, assemble_feature_matrix, build_dataset_split, load_visual_baseline_log, measure_inference_latency


@dataclass
class ClassicalResult:
    model: Pipeline
    metrics: dict[str, float]
    confusion: object
    predictions: list[str]


def _build_estimator(model_name: str, seed: int, hyperparameters: dict[str, Any] | None = None):
    normalized = model_name.strip().lower()
    hyperparameters = hyperparameters or {}
    if normalized == "svm":
        return SVC(
            kernel=str(hyperparameters.get("kernel", "rbf")),
            probability=True,
            class_weight="balanced",
            C=float(hyperparameters.get("C", 1.0)),
            gamma=str(hyperparameters.get("gamma", "scale")),
            random_state=seed,
        ), True
    if normalized == "random_forest":
        return RandomForestClassifier(
            n_estimators=int(hyperparameters.get("n_estimators", 180)),
            max_depth=hyperparameters.get("max_depth"),
            random_state=seed,
            class_weight="balanced",
        ), False
    if normalized == "logistic_regression":
        return LogisticRegression(
            max_iter=int(hyperparameters.get("max_iter", 1000)),
            solver=str(hyperparameters.get("solver", "lbfgs")),
            class_weight="balanced",
            random_state=seed,
        ), True
    if normalized == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=int(hyperparameters.get("n_estimators", 220)),
            max_depth=hyperparameters.get("max_depth"),
            random_state=seed,
            class_weight="balanced",
        ), False
    if normalized == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=int(hyperparameters.get("n_estimators", 100)),
            learning_rate=float(hyperparameters.get("learning_rate", 0.1)),
            random_state=seed,
        ), False
    if normalized == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError("xgboost is not installed. Install it or choose another classical model.") from exc
        return XGBClassifier(
            n_estimators=int(hyperparameters.get("n_estimators", 200)),
            max_depth=int(hyperparameters.get("max_depth", 4)),
            learning_rate=float(hyperparameters.get("learning_rate", 0.05)),
            subsample=float(hyperparameters.get("subsample", 0.9)),
            colsample_bytree=float(hyperparameters.get("colsample_bytree", 0.9)),
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=seed,
        ), False
    raise ValueError(f"Unsupported classical model '{model_name}'.")


def train_and_evaluate_classical(
    train_bundle: dict[str, object],
    test_bundle: dict[str, object],
    modalities: tuple[str, ...],
    labels: list[str],
    model_name: str = "svm",
    seed: int = 42,
    hyperparameters: dict[str, Any] | None = None,
) -> ClassicalResult:
    x_train = assemble_feature_matrix(train_bundle, modalities)
    x_test = assemble_feature_matrix(test_bundle, modalities)
    y_train = train_bundle["labels"]
    y_test = test_bundle["labels"]

    estimator, use_scaler = _build_estimator(model_name, seed, hyperparameters)
    if use_scaler:
        model = Pipeline([("scaler", StandardScaler()), ("model", estimator)])
    else:
        model = Pipeline([("model", estimator)])

    normalized_model_name = model_name.strip().lower()
    if normalized_model_name == "xgboost":
        label_encoder = LabelEncoder()
        encoded_y_train = label_encoder.fit_transform(y_train)
        encoded_y_test = label_encoder.transform(y_test)
        model.fit(x_train, encoded_y_train)
        encoded_predictions = model.predict(x_test)
        y_pred = label_encoder.inverse_transform(encoded_predictions).tolist()
        scores = compute_metrics(y_test, y_pred)
    else:
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


def run_classical_experiment(project_root: Path, config_path: Path) -> dict[str, str]:
    context = build_experiment_context(project_root, config_path)
    config = context.config
    if context.case_study != "CS3":
        raise ValueError(f"Classical trainer only supports CS3 configs, received {context.case_study}.")

    seed = int(config["seed"])
    set_global_seed(seed)
    logger = get_logger(f"paper1.cs3.{context.experiment_name}", context.log_path)
    logger.info("Running classical CS3 experiment '%s'.", context.experiment_name)

    model_cfg = config.get("model", {})
    model_name = str(model_cfg.get("name", "svm"))
    modalities = tuple(config.get("modalities", {}).get("selected", ["video", "audio"]))
    hyperparameters = model_cfg.get("hyperparameters", {})
    dataset_name = str(config.get("dataset", {}).get("name", "synthetic_aligned_multimodal_windows"))
    data_source_type = str(config.get("evaluation", {}).get("data_source_type", "synthetic"))
    runtime_type = str(config.get("evaluation", {}).get("runtime_type", "software_only"))
    model_status = str(config.get("evaluation", {}).get("model_status", "fully_runnable"))
    evidence_level = str(config.get("evaluation", {}).get("evidence_level", "benchmark_preliminary"))

    if str(model_name).strip().lower() == "baseline_visual":
        if dataset_name == "pilot_real_anchor_demonstration":
            requested_session = str(config.get("inputs", {}).get("session_dir", "")).strip()
            session_dir = Path(requested_session).resolve() if requested_session else resolve_latest_session(context.project_root)
            prediction_df = baseline_predictions_from_frames(session_dir, max_frames=int(config.get("dataset", {}).get("max_frames", 20)))
            metrics = {
                "accuracy": None,
                "macro_f1": None,
                "weighted_f1": None,
                "uar": None,
                "inference_latency_ms": None,
            }
            confusion = pd.DataFrame(columns=["true_label", *LABELS])
            predictions = prediction_df["predicted_emotion"].astype(str).tolist()
            write_dataframe(context.csv_dir / "pilot_real_anchor_predictions.csv", prediction_df)
            model_status = "partially_runnable"
            data_source_type = "pilot_real_anchor"
            evidence_level = "pilot_demonstration"
        else:
            baseline_df = load_visual_baseline_log(context.project_root)
            y_true = baseline_df["true_emotion"].astype(str).str.lower().tolist()
            y_pred = baseline_df["emotion"].astype(str).str.lower().tolist()
            scores = compute_metrics(y_true, y_pred)
            metrics = {
                "accuracy": round(scores.accuracy, 4),
                "macro_f1": round(scores.macro_f1, 4),
                "weighted_f1": round(scores.weighted_f1, 4),
                "uar": round(scores.unweighted_recall, 4),
                "inference_latency_ms": None,
            }
            confusion = confusion_dataframe(y_true, y_pred, LABELS)
            predictions = y_pred
        artifact_path = None
    else:
        split = build_dataset_split(context.project_root, config.get("dataset", {}), seed)
        result = train_and_evaluate_classical(
            split["train"],
            split["test"],
            modalities,
            LABELS,
            model_name=model_name,
            seed=seed,
            hyperparameters=hyperparameters,
        )
        metrics = result.metrics
        confusion = result.confusion
        predictions = result.predictions
        artifact_path = context.log_dir / f"{context.experiment_name}.joblib"
        joblib.dump({"model": result.model, "modalities": modalities, "labels": LABELS}, artifact_path)

    metrics_row = {
        "experiment_name": context.experiment_name,
        "case_study": context.case_study,
        "model_family": str(model_cfg.get("family", "classical")),
        "algorithm_name": model_name,
        "modality_setting": "_".join(modalities),
        "seed": seed,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "uar": metrics["uar"],
        "inference_latency_ms": metrics["inference_latency_ms"],
        "data_source_type": data_source_type,
        "runtime_type": runtime_type,
        "model_status": model_status,
        "evidence_level": evidence_level,
        "data_regime": dataset_name,
    }
    write_yaml(context.config_snapshot_path, config)
    write_dataframe(context.metrics_csv_path, pd.DataFrame([metrics_row]))
    write_dataframe(context.csv_dir / "model_performance_summary.csv", pd.DataFrame([metrics_row]))
    write_dataframe(context.csv_dir / "confusion_matrix.csv", confusion)
    write_dataframe(context.csv_dir / "predictions.csv", pd.DataFrame({"prediction": predictions}))

    summary = {
        "experiment_name": context.experiment_name,
        "case_study": context.case_study,
        "config_path": str(context.config_path),
        "summary_json": str(context.summary_json_path),
        "metrics_csv": str(context.metrics_csv_path),
        "model_performance_summary": str(context.csv_dir / "model_performance_summary.csv"),
        "confusion_matrix_csv": str(context.csv_dir / "confusion_matrix.csv"),
        "prediction_csv": str(context.csv_dir / "predictions.csv"),
        "artifact_path": str(artifact_path) if artifact_path is not None else "",
        "log_path": str(context.log_path),
    }
    write_json(context.summary_json_path, summary)
    logger.info("Finished classical CS3 experiment '%s'.", context.experiment_name)
    return {key: str(value) for key, value in summary.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one classical CS3 experiment from a single config.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_classical_experiment(Path(args.project_root).resolve(), Path(args.config).resolve())


if __name__ == "__main__":
    main()
