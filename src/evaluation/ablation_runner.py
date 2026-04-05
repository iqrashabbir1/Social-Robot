from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from src.common.io_utils import read_yaml, write_dataframe, write_json
from src.common.logging_utils import get_logger
from src.common.paths import Paper1Paths
from src.common.reproducibility import set_global_seed
from src.common.tracking_utils import RunTracker
from src.evaluation.metrics_classification import compute_metrics, confusion_dataframe
from src.features.video_features import load_visual_baseline_log
from src.models.classical.train_classical import train_and_evaluate_classical
from src.models.deep.train_deep_fusion import train_and_evaluate_deep_fusion
from src.models.deep.train_deep_fusion_gpu import train_and_evaluate_deep_fusion_gpu
from src.models.inference_benchmark import (
    LABELS,
    apply_missing_modality,
    build_synthetic_multimodal_dataset,
    split_feature_bundle,
)
from src.models.torch_runtime import TorchRuntime, resolve_torch_runtime
from src.models.transformer.train_transformer_fusion import train_and_evaluate_transformer_fusion
from src.models.transformer.train_transformer_fusion_gpu import train_and_evaluate_transformer_fusion_gpu


ABLATIONS = {
    "video_only": ("video",),
    "audio_only": ("audio",),
    "context_only": ("context",),
    "video_audio": ("video", "audio"),
    "video_audio_context": ("video", "audio", "context"),
}

CLASSICAL_ID_MAP = {
    "svm": "B1_SVM",
    "random_forest": "B1_RF",
    "logistic_regression": "B1_LOGREG",
    "extra_trees": "B1_EXTRA",
    "gradient_boosting": "B1_GB",
}


def _tuple_modalities(raw_modalities: object, fallback: tuple[str, ...]) -> tuple[str, ...]:
    if not raw_modalities:
        return fallback
    if isinstance(raw_modalities, str):
        return tuple(part.strip() for part in raw_modalities.split(",") if part.strip()) or fallback
    if isinstance(raw_modalities, (list, tuple)):
        values = tuple(str(item).strip() for item in raw_modalities if str(item).strip())
        return values or fallback
    return fallback


def _csv_output_dir(project_root: Path, output_subdir: str) -> Path:
    base_dir = Paper1Paths.from_project_root(project_root).outputs_csv_cs3
    return base_dir / output_subdir if output_subdir else base_dir


def _resolve_optional_dir(project_root: Path, raw_path: str) -> Path | None:
    cleaned = raw_path.strip()
    if not cleaned:
        return None
    candidate = Path(cleaned)
    return candidate if candidate.is_absolute() else project_root / candidate


def _evaluate_baseline_zero(project_root: Path) -> tuple[dict[str, object], pd.DataFrame]:
    df = load_visual_baseline_log(project_root)
    y_true = df["true_emotion"].astype(str).str.lower().tolist()
    y_pred = df["emotion"].astype(str).str.lower().tolist()
    scores = compute_metrics(y_true, y_pred)
    metrics = {
        "model_id": "B0",
        "model_family": "existing_repo_baseline",
        "algorithm_name": "deepface_visual_baseline",
        "modality_setting": "video_only",
        "accuracy": round(scores.accuracy, 4),
        "macro_f1": round(scores.macro_f1, 4),
        "weighted_f1": round(scores.weighted_f1, 4),
        "uar": round(scores.unweighted_recall, 4),
        "inference_latency_ms": None,
        "robustness_missing_modality_macro_f1": None,
        "epochs": None,
        "evidence_level": "implemented_real_baseline",
        "data_regime": "real_visual_log",
        "notes": "Preserved DeepFace-style visual baseline evaluated from tests/emotion_log_labeled.csv.",
    }
    confusion = confusion_dataframe(y_true, y_pred, LABELS)
    return metrics, confusion


def _classical_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    algorithms = config.get("classical_algorithms")
    if isinstance(algorithms, list) and algorithms:
        return [{"name": str(name).strip().lower()} for name in algorithms if str(name).strip()]
    fallback = str(config.get("classical", "svm")).strip().lower()
    return [{"name": fallback}]


def _deep_specs(config: dict[str, Any], training_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    variants = config.get("deep_variants")
    forced_epochs = int(training_cfg.get("deep_epochs", 14))
    force_override = bool(training_cfg.get("force_deep_epochs", False))
    if isinstance(variants, list) and variants:
        normalized = []
        for index, variant in enumerate(variants, start=1):
            if not isinstance(variant, dict):
                continue
            normalized.append(
                {
                    "name": str(variant.get("name", f"deep_variant_{index}")).strip(),
                    "hidden_layers": tuple(int(value) for value in variant.get("hidden_layers", [96, 48])),
                    "learning_rate_init": float(variant.get("learning_rate_init", 0.001)),
                    "epochs": forced_epochs if force_override else int(variant.get("epochs", forced_epochs)),
                }
            )
        if normalized:
            return normalized
    return [
        {
            "name": str(config.get("deep", "late_fusion_mlp")).strip(),
            "hidden_layers": (96, 48),
            "learning_rate_init": 0.001,
            "epochs": forced_epochs,
        }
    ]


def _transformer_specs(config: dict[str, Any], training_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    variants = config.get("transformer_variants")
    forced_epochs = int(training_cfg.get("transformer_epochs", 14))
    force_override = bool(training_cfg.get("force_transformer_epochs", False))
    if isinstance(variants, list) and variants:
        normalized = []
        for index, variant in enumerate(variants, start=1):
            if not isinstance(variant, dict):
                continue
            normalized.append(
                {
                    "name": str(variant.get("name", f"transformer_variant_{index}")).strip(),
                    "hidden_dim": int(variant.get("hidden_dim", 16)),
                    "alpha": float(variant.get("alpha", 0.0001)),
                    "epochs": forced_epochs if force_override else int(variant.get("epochs", forced_epochs)),
                }
            )
        if normalized:
            return normalized
    return [
        {
            "name": str(config.get("transformer", "lightweight_cross_attention")).strip(),
            "hidden_dim": 16,
            "alpha": 0.0001,
            "epochs": forced_epochs,
        }
    ]


def _majority_vote(prediction_lists: list[list[str]], labels: list[str]) -> list[str]:
    voted: list[str] = []
    for vote_group in zip(*prediction_lists):
        counts = Counter(vote_group)
        winner = sorted(counts.items(), key=lambda item: (-item[1], labels.index(item[0]) if item[0] in labels else len(labels)))[0][0]
        voted.append(winner)
    return voted


def _sort_models(df: pd.DataFrame, selection_cfg: dict[str, Any]) -> pd.DataFrame:
    if df.empty:
        return df
    primary_metric = str(selection_cfg.get("primary_metric", "macro_f1"))
    ascending_metrics = set(selection_cfg.get("ascending_metrics", ["inference_latency_ms"]))
    tie_breakers = list(selection_cfg.get("tie_breakers", ["weighted_f1", "accuracy", "inference_latency_ms"]))
    metric_order = [primary_metric] + [metric for metric in tie_breakers if metric != primary_metric]

    ranked = df.copy()
    if "rank" in ranked.columns:
        ranked = ranked.drop(columns=["rank"])
    for metric in metric_order:
        if metric not in ranked.columns:
            ranked[metric] = None
    ranked["inference_latency_ms"] = ranked["inference_latency_ms"].fillna(float("inf"))
    ascending = [metric in ascending_metrics for metric in metric_order]
    ranked = ranked.sort_values(metric_order, ascending=ascending, na_position="last").reset_index(drop=True)
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    return ranked


def _row_with_common_metadata(
    model_id: str,
    family: str,
    algorithm_name: str,
    modalities: tuple[str, ...],
    metrics: dict[str, float],
    evidence_level: str,
    notes: str,
    epochs: int | None,
    missing_macro_f1: float | None,
) -> dict[str, object]:
    return {
        "model_id": model_id,
        "model_family": family,
        "algorithm_name": algorithm_name,
        "modality_setting": "_".join(modalities),
        **metrics,
        "robustness_missing_modality_macro_f1": missing_macro_f1,
        "epochs": epochs,
        "evidence_level": evidence_level,
        "data_regime": "synthetic_aligned_multimodal_windows",
        "notes": notes,
    }


def _run_ablation_set(
    project_root: Path,
    train_bundle: dict[str, object],
    test_bundle: dict[str, object],
    seed: int,
    models_cfg: dict[str, Any],
    execution_cfg: dict[str, Any],
    training_cfg: dict[str, Any],
    selection_cfg: dict[str, Any],
    runtime: TorchRuntime,
    tracker: RunTracker | None = None,
) -> tuple[pd.DataFrame, list[dict[str, object]], dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    performance_rows: list[dict[str, object]] = []
    ablation_rows: list[dict[str, object]] = []
    confusion_outputs: dict[str, pd.DataFrame] = {}
    training_curves: list[pd.DataFrame] = []
    family_best_rows: list[dict[str, object]] = []

    prediction_cache: dict[str, list[str]] = {}
    missing_prediction_cache: dict[str, list[str]] = {}
    family_confusions: dict[str, list[tuple[dict[str, object], pd.DataFrame]]] = {"deep": [], "transformer": [], "hybrid": []}

    main_modalities = _tuple_modalities(execution_cfg.get("main_modalities"), ABLATIONS["video_audio_context"])
    include_classical = bool(execution_cfg.get("include_classical_b1", True))
    include_deep = bool(execution_cfg.get("include_deep_b2", True))
    include_transformer = bool(execution_cfg.get("include_transformer_b3", True))
    include_missing_modality = bool(execution_cfg.get("include_missing_modality", True))
    include_ablations = bool(execution_cfg.get("include_ablations", True))
    include_hybrid = bool(models_cfg.get("hybrid", {}).get("enabled", False))
    use_gpu_backend = runtime.active_backend == "gpu"

    missing_test_bundle = (
        apply_missing_modality(test_bundle, main_modalities, 0.2, seed + 101)
        if include_missing_modality
        else test_bundle
    )
    checkpoint_every = int(training_cfg.get("checkpoint_every", 0))
    checkpoint_dir = _resolve_optional_dir(project_root, str(training_cfg.get("checkpoint_dir", "")))
    ablation_epochs = int(training_cfg.get("ablation_epochs", 14))

    if include_classical:
        for spec in _classical_specs(models_cfg):
            model_name = spec["name"]
            model_id = CLASSICAL_ID_MAP.get(model_name, f"B1_{model_name.upper()}")
            result = train_and_evaluate_classical(train_bundle, test_bundle, main_modalities, LABELS, model_name=model_name, seed=seed)
            missing_result = (
                train_and_evaluate_classical(train_bundle, missing_test_bundle, main_modalities, LABELS, model_name=model_name, seed=seed)
                if include_missing_modality
                else result
            )
            prediction_cache[model_id] = result.predictions
            missing_prediction_cache[model_id] = missing_result.predictions
            performance_rows.append(
                _row_with_common_metadata(
                    model_id=model_id,
                    family="classical",
                    algorithm_name=model_name,
                    modalities=main_modalities,
                    metrics=result.metrics,
                    evidence_level="synthetic_placeholder_benchmark",
                    notes=f"Classical handcrafted-feature benchmark using {model_name}.",
                    epochs=None,
                    missing_macro_f1=missing_result.metrics["macro_f1"],
                )
            )

    if include_deep:
        for index, spec in enumerate(_deep_specs(models_cfg, training_cfg), start=1):
            model_id = f"B2_{index}"
            if tracker is not None:
                tracker.model_started(model_id, "deep", str(spec["name"]), int(spec["epochs"]))
            result = train_and_evaluate_deep_fusion(
                train_bundle,
                test_bundle,
                main_modalities,
                LABELS,
                seed=seed,
                epochs=int(spec["epochs"]),
                model_id=model_id,
                checkpoint_dir=checkpoint_dir,
                checkpoint_every=checkpoint_every,
                hidden_layers=tuple(spec["hidden_layers"]),
                learning_rate_init=float(spec["learning_rate_init"]),
                progress_callback=(
                    lambda epoch_row, tracker=tracker, model_id=model_id, spec=spec: tracker.epoch_progress(
                        model_id=model_id,
                        model_family="deep",
                        algorithm_name=str(spec["name"]),
                        epoch=int(epoch_row["epoch"]),
                        total_epochs=int(epoch_row["total_epochs"]),
                        metrics=epoch_row,
                    )
                ) if tracker is not None else None,
            ) if not use_gpu_backend else train_and_evaluate_deep_fusion_gpu(
                train_bundle,
                test_bundle,
                main_modalities,
                LABELS,
                device=runtime.device,
                seed=seed,
                epochs=int(spec["epochs"]),
                model_id=model_id,
                checkpoint_dir=checkpoint_dir,
                checkpoint_every=checkpoint_every,
                hidden_layers=tuple(spec["hidden_layers"]),
                learning_rate_init=float(spec["learning_rate_init"]),
                progress_callback=(
                    lambda epoch_row, tracker=tracker, model_id=model_id, spec=spec: tracker.epoch_progress(
                        model_id=model_id,
                        model_family="deep",
                        algorithm_name=str(spec["name"]),
                        epoch=int(epoch_row["epoch"]),
                        total_epochs=int(epoch_row["total_epochs"]),
                        metrics=epoch_row,
                    )
                ) if tracker is not None else None,
            )
            missing_result = (
                train_and_evaluate_deep_fusion(
                    train_bundle,
                    missing_test_bundle,
                    main_modalities,
                    LABELS,
                    seed=seed,
                    epochs=int(spec["epochs"]),
                    model_id=model_id,
                    hidden_layers=tuple(spec["hidden_layers"]),
                    learning_rate_init=float(spec["learning_rate_init"]),
                ) if not use_gpu_backend else train_and_evaluate_deep_fusion_gpu(
                    train_bundle,
                    missing_test_bundle,
                    main_modalities,
                    LABELS,
                    device=runtime.device,
                    seed=seed,
                    epochs=int(spec["epochs"]),
                    model_id=model_id,
                    hidden_layers=tuple(spec["hidden_layers"]),
                    learning_rate_init=float(spec["learning_rate_init"]),
                )
                if include_missing_modality
                else result
            )
            prediction_cache[model_id] = result.predictions
            missing_prediction_cache[model_id] = missing_result.predictions
            row = _row_with_common_metadata(
                model_id=model_id,
                family="deep",
                algorithm_name=str(spec["name"]),
                modalities=main_modalities,
                metrics=result.metrics,
                evidence_level="synthetic_placeholder_benchmark",
                notes=f"Late-fusion MLP with hidden layers {tuple(spec['hidden_layers'])} on {runtime.active_backend.upper()} backend.",
                epochs=int(spec["epochs"]),
                missing_macro_f1=missing_result.metrics["macro_f1"],
            )
            performance_rows.append(row)
            training_curves.append(result.training_curve.assign(algorithm_name=str(spec["name"]), model_family="deep"))
            family_confusions["deep"].append((row, result.confusion))
            if tracker is not None:
                tracker.model_completed(model_id, "deep", str(spec["name"]), result.metrics)

    if include_transformer:
        for index, spec in enumerate(_transformer_specs(models_cfg, training_cfg), start=1):
            model_id = f"B3_{index}"
            if tracker is not None:
                tracker.model_started(model_id, "transformer", str(spec["name"]), int(spec["epochs"]))
            result = train_and_evaluate_transformer_fusion(
                train_bundle,
                test_bundle,
                LABELS,
                modalities=main_modalities,
                seed=seed,
                epochs=int(spec["epochs"]),
                model_id=model_id,
                checkpoint_dir=checkpoint_dir,
                checkpoint_every=checkpoint_every,
                hidden_dim=int(spec["hidden_dim"]),
                alpha=float(spec["alpha"]),
                progress_callback=(
                    lambda epoch_row, tracker=tracker, model_id=model_id, spec=spec: tracker.epoch_progress(
                        model_id=model_id,
                        model_family="transformer",
                        algorithm_name=str(spec["name"]),
                        epoch=int(epoch_row["epoch"]),
                        total_epochs=int(epoch_row["total_epochs"]),
                        metrics=epoch_row,
                    )
                ) if tracker is not None else None,
            ) if not use_gpu_backend else train_and_evaluate_transformer_fusion_gpu(
                train_bundle,
                test_bundle,
                LABELS,
                device=runtime.device,
                modalities=main_modalities,
                seed=seed,
                epochs=int(spec["epochs"]),
                model_id=model_id,
                checkpoint_dir=checkpoint_dir,
                checkpoint_every=checkpoint_every,
                hidden_dim=int(spec["hidden_dim"]),
                alpha=float(spec["alpha"]),
                progress_callback=(
                    lambda epoch_row, tracker=tracker, model_id=model_id, spec=spec: tracker.epoch_progress(
                        model_id=model_id,
                        model_family="transformer",
                        algorithm_name=str(spec["name"]),
                        epoch=int(epoch_row["epoch"]),
                        total_epochs=int(epoch_row["total_epochs"]),
                        metrics=epoch_row,
                    )
                ) if tracker is not None else None,
            )
            missing_result = (
                train_and_evaluate_transformer_fusion(
                    train_bundle,
                    missing_test_bundle,
                    LABELS,
                    modalities=main_modalities,
                    seed=seed,
                    epochs=int(spec["epochs"]),
                    model_id=model_id,
                    hidden_dim=int(spec["hidden_dim"]),
                    alpha=float(spec["alpha"]),
                ) if not use_gpu_backend else train_and_evaluate_transformer_fusion_gpu(
                    train_bundle,
                    missing_test_bundle,
                    LABELS,
                    device=runtime.device,
                    modalities=main_modalities,
                    seed=seed,
                    epochs=int(spec["epochs"]),
                    model_id=model_id,
                    hidden_dim=int(spec["hidden_dim"]),
                    alpha=float(spec["alpha"]),
                )
                if include_missing_modality
                else result
            )
            prediction_cache[model_id] = result.predictions
            missing_prediction_cache[model_id] = missing_result.predictions
            row = _row_with_common_metadata(
                model_id=model_id,
                family="transformer",
                algorithm_name=str(spec["name"]),
                modalities=main_modalities,
                metrics=result.metrics,
                evidence_level="synthetic_placeholder_benchmark",
                notes=f"Lightweight fusion transformer with hidden_dim={int(spec['hidden_dim'])} on {runtime.active_backend.upper()} backend.",
                epochs=int(spec["epochs"]),
                missing_macro_f1=missing_result.metrics["macro_f1"],
            )
            performance_rows.append(row)
            training_curves.append(result.training_curve.assign(algorithm_name=str(spec["name"]), model_family="transformer"))
            family_confusions["transformer"].append((row, result.confusion))
            if tracker is not None:
                tracker.model_completed(model_id, "transformer", str(spec["name"]), result.metrics)

    if include_hybrid:
        hybrid_cfg = models_cfg.get("hybrid", {})
        members = [str(member).strip() for member in hybrid_cfg.get("members", []) if str(member).strip()]
        resolved_members = [member for member in members if member in prediction_cache]
        if not resolved_members:
            resolved_members = [row["model_id"] for row in performance_rows if row["model_family"] in {"classical", "deep", "transformer"}][:3]
        if resolved_members:
            ensemble_predictions = _majority_vote([prediction_cache[member] for member in resolved_members], LABELS)
            y_true = list(test_bundle["labels"])
            scores = compute_metrics(y_true, ensemble_predictions)
            missing_macro_f1 = None
            if include_missing_modality and all(member in missing_prediction_cache for member in resolved_members):
                missing_predictions = _majority_vote([missing_prediction_cache[member] for member in resolved_members], LABELS)
                missing_macro_f1 = compute_metrics(y_true, missing_predictions).macro_f1
            latency = 0.0
            for member in resolved_members:
                row = next((item for item in performance_rows if item["model_id"] == member), None)
                if row and row["inference_latency_ms"] is not None:
                    latency += float(row["inference_latency_ms"])
            hybrid_metrics = {
                "accuracy": round(scores.accuracy, 4),
                "macro_f1": round(scores.macro_f1, 4),
                "weighted_f1": round(scores.weighted_f1, 4),
                "uar": round(scores.unweighted_recall, 4),
                "inference_latency_ms": round(latency, 4),
            }
            hybrid_row = _row_with_common_metadata(
                model_id="H1_HYBRID",
                family="hybrid",
                algorithm_name=str(hybrid_cfg.get("name", "majority_vote_ensemble")),
                modalities=main_modalities,
                metrics=hybrid_metrics,
                evidence_level="synthetic_placeholder_benchmark",
                notes=f"Hybrid majority-vote ensemble over members: {', '.join(resolved_members)}.",
                epochs=max((int(row["epochs"]) for row in performance_rows if row["model_id"] in resolved_members and row["epochs"] is not None), default=None),
                missing_macro_f1=round(missing_macro_f1, 4) if missing_macro_f1 is not None else None,
            )
            performance_rows.append(hybrid_row)
            family_confusions["hybrid"].append((hybrid_row, confusion_dataframe(y_true, ensemble_predictions, LABELS)))

    performance_df = pd.DataFrame(performance_rows)
    ranked_df = _sort_models(performance_df, selection_cfg)
    best_model_df = ranked_df.head(1).copy()

    for family_name, outputs in family_confusions.items():
        if not outputs:
            continue
        family_rows = pd.DataFrame([row for row, _ in outputs])
        family_ranked = _sort_models(family_rows, selection_cfg)
        best_family_model_id = family_ranked.iloc[0]["model_id"]
        best_confusion = next(confusion for row, confusion in outputs if row["model_id"] == best_family_model_id)
        confusion_outputs[f"confusion_matrix_{family_name}"] = best_confusion
        family_best_rows.append(family_ranked.iloc[0].to_dict())

    if include_ablations:
        configured_ablation_names = execution_cfg.get("ablation_names", list(ABLATIONS.keys()))
        if not configured_ablation_names:
            configured_ablation_names = list(ABLATIONS.keys())
        ablation_model_ids = set(
            str(model_id).strip()
            for model_id in execution_cfg.get("ablation_model_ids", ranked_df["model_id"].head(4).tolist())
            if str(model_id).strip()
        )
        for ablation_name in configured_ablation_names:
            if ablation_name not in ABLATIONS:
                continue
            modalities = ABLATIONS[ablation_name]
            condition_pairs = [("nominal", test_bundle)]
            if include_missing_modality:
                condition_pairs.append(
                    (
                        "missing_modality_20pct",
                        apply_missing_modality(test_bundle, modalities, 0.2, seed + len(modalities)),
                    )
                )

            for condition_name, condition_bundle in condition_pairs:
                for row in ranked_df.to_dict(orient="records"):
                    if row["model_id"] not in ablation_model_ids:
                        continue
                    if row["model_family"] == "classical":
                        result = train_and_evaluate_classical(
                            train_bundle,
                            condition_bundle,
                            modalities,
                            LABELS,
                            model_name=str(row["algorithm_name"]),
                            seed=seed,
                        )
                        metrics = result.metrics
                    elif row["model_family"] == "deep":
                        spec = next(spec for spec in _deep_specs(models_cfg, training_cfg) if spec["name"] == row["algorithm_name"])
                        metrics = train_and_evaluate_deep_fusion(
                            train_bundle,
                            condition_bundle,
                            modalities,
                            LABELS,
                            seed=seed,
                            epochs=min(int(spec["epochs"]), ablation_epochs),
                            model_id=row["model_id"],
                            hidden_layers=tuple(spec["hidden_layers"]),
                            learning_rate_init=float(spec["learning_rate_init"]),
                        ).metrics if not use_gpu_backend else train_and_evaluate_deep_fusion_gpu(
                            train_bundle,
                            condition_bundle,
                            modalities,
                            LABELS,
                            device=runtime.device,
                            seed=seed,
                            epochs=min(int(spec["epochs"]), ablation_epochs),
                            model_id=row["model_id"],
                            hidden_layers=tuple(spec["hidden_layers"]),
                            learning_rate_init=float(spec["learning_rate_init"]),
                        ).metrics
                    elif row["model_family"] == "transformer":
                        spec = next(spec for spec in _transformer_specs(models_cfg, training_cfg) if spec["name"] == row["algorithm_name"])
                        metrics = train_and_evaluate_transformer_fusion(
                            train_bundle,
                            condition_bundle,
                            LABELS,
                            modalities=modalities,
                            seed=seed,
                            epochs=min(int(spec["epochs"]), ablation_epochs),
                            model_id=row["model_id"],
                            hidden_dim=int(spec["hidden_dim"]),
                            alpha=float(spec["alpha"]),
                        ).metrics if not use_gpu_backend else train_and_evaluate_transformer_fusion_gpu(
                            train_bundle,
                            condition_bundle,
                            LABELS,
                            device=runtime.device,
                            modalities=modalities,
                            seed=seed,
                            epochs=min(int(spec["epochs"]), ablation_epochs),
                            model_id=row["model_id"],
                            hidden_dim=int(spec["hidden_dim"]),
                            alpha=float(spec["alpha"]),
                        ).metrics
                    else:
                        continue
                    ablation_rows.append(
                        {
                            "model_id": row["model_id"],
                            "algorithm_name": row["algorithm_name"],
                            "ablation_name": ablation_name,
                            "condition": condition_name,
                            "accuracy": metrics["accuracy"],
                            "macro_f1": metrics["macro_f1"],
                            "weighted_f1": metrics["weighted_f1"],
                            "uar": metrics["uar"],
                            "inference_latency_ms": metrics["inference_latency_ms"],
                            "evidence_level": row["evidence_level"],
                        }
                    )

    training_curve_df = pd.concat(training_curves, ignore_index=True) if training_curves else pd.DataFrame()
    family_best_df = pd.DataFrame(family_best_rows)
    return ranked_df, ablation_rows, confusion_outputs, training_curve_df, best_model_df, family_best_df


def run_cs3(project_root: Path, config_path: Path) -> dict[str, str]:
    config = read_yaml(config_path)
    seed = int(config.get("seed", 42))
    models_cfg = config.get("models", {})
    execution_cfg = config.get("execution", {})
    training_cfg = config.get("training", {})
    selection_cfg = config.get("selection", {})
    runtime = resolve_torch_runtime(
        str(training_cfg.get("runtime_backend", "cpu")),
        str(training_cfg.get("torch_device", "auto")),
    )
    output_subdir = str(config.get("outputs", {}).get("output_subdir", "")).strip()
    run_tag = output_subdir or "default"

    set_global_seed(seed)
    paper1_paths = Paper1Paths.from_project_root(project_root)
    paper1_paths.ensure()

    csv_output_dir = _csv_output_dir(project_root, output_subdir)
    csv_output_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger("paper1.cs3", paper1_paths.outputs_logs / f"paper1_cs3_{run_tag}.log")
    logger.info("Running CS3 emotion recognition benchmark with tag '%s'.", run_tag)
    logger.info("Runtime backend resolved to %s on device %s. %s", runtime.active_backend, runtime.device, runtime.reason)
    tracker = RunTracker(
        run_tag=run_tag,
        progress_log_path=paper1_paths.outputs_logs / f"paper1_cs3_{run_tag}_progress.jsonl",
        latest_status_path=paper1_paths.outputs_logs / f"paper1_cs3_{run_tag}_latest_status.json",
        epoch_progress_path=csv_output_dir / "epoch_progress.csv",
        logger=logger,
        log_every=int(training_cfg.get("log_every", 1)),
    )
    tracker.event(
        "run_started",
        config_path=str(config_path),
        csv_output_dir=str(csv_output_dir),
        runtime_backend=runtime.active_backend,
        requested_backend=runtime.requested_backend,
        device=runtime.device,
        runtime_reason=runtime.reason,
    )

    performance_frames: list[pd.DataFrame] = []
    include_baseline_b0 = bool(execution_cfg.get("include_baseline_b0", True))
    if include_baseline_b0:
        baseline_metrics, baseline_confusion = _evaluate_baseline_zero(project_root)
        performance_frames.append(pd.DataFrame([baseline_metrics]))
        write_dataframe(csv_output_dir / "confusion_matrix_baseline.csv", baseline_confusion)

    bundle = build_synthetic_multimodal_dataset(
        project_root=project_root,
        seed=seed,
        n_samples=int(config.get("dataset", {}).get("n_samples", 480)),
    )
    split = split_feature_bundle(bundle, test_size=float(config.get("dataset", {}).get("test_size", 0.25)), seed=seed)

    ranked_df, ablation_rows, confusion_outputs, training_curve_df, best_model_df, family_best_df = _run_ablation_set(
        project_root,
        split["train"],
        split["test"],
        seed,
        models_cfg,
        execution_cfg,
        training_cfg,
        selection_cfg,
        runtime,
        tracker,
    )

    if performance_frames:
        non_ranked = ranked_df.copy()
        non_ranked["rank"] = non_ranked["rank"] + len(performance_frames)
        performance_df = pd.concat([pd.DataFrame(performance_frames[0]), non_ranked], ignore_index=True)
    else:
        performance_df = ranked_df
    performance_df = _sort_models(performance_df, selection_cfg)

    write_dataframe(csv_output_dir / "model_performance_summary.csv", performance_df)
    write_dataframe(csv_output_dir / "model_ranking.csv", performance_df)
    write_dataframe(csv_output_dir / "best_model_summary.csv", best_model_df)
    write_dataframe(csv_output_dir / "family_best_summary.csv", family_best_df)

    for confusion_name, confusion_df in confusion_outputs.items():
        write_dataframe(csv_output_dir / f"{confusion_name}.csv", confusion_df)
    if ablation_rows:
        write_dataframe(csv_output_dir / "ablation_results.csv", pd.DataFrame(ablation_rows))
    if not training_curve_df.empty:
        write_dataframe(csv_output_dir / "training_curves.csv", training_curve_df)

    summary = {
        "case_study": "CS3",
        "config_path": str(config_path),
        "run_tag": run_tag,
        "csv_output_dir": str(csv_output_dir),
        "runtime_backend": runtime.active_backend,
        "device": runtime.device,
        "progress_log": str(paper1_paths.outputs_logs / f"paper1_cs3_{run_tag}_progress.jsonl"),
        "latest_status": str(paper1_paths.outputs_logs / f"paper1_cs3_{run_tag}_latest_status.json"),
        "epoch_progress": str(csv_output_dir / "epoch_progress.csv"),
        "model_performance_summary": str(csv_output_dir / "model_performance_summary.csv"),
        "model_ranking": str(csv_output_dir / "model_ranking.csv"),
        "best_model_summary": str(csv_output_dir / "best_model_summary.csv"),
        "ablation_results": str(csv_output_dir / "ablation_results.csv"),
        "training_curves": str(csv_output_dir / "training_curves.csv"),
    }
    write_json(paper1_paths.outputs_logs / f"paper1_cs3_{run_tag}_summary.json", summary)
    tracker.run_completed(best_model_df.iloc[0].to_dict() if not best_model_df.empty else None)
    logger.info("CS3 outputs written to %s", csv_output_dir)
    return {key: str(value) for key, value in summary.items()}


def _set_nested(config: dict[str, Any], keys: tuple[str, ...], value: Any) -> None:
    cursor = config
    for key in keys[:-1]:
        next_value = cursor.get(key)
        if not isinstance(next_value, dict):
            next_value = {}
            cursor[key] = next_value
        cursor = next_value
    cursor[keys[-1]] = value


def _apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    overrides = {
        ("training", "deep_epochs"): args.deep_epochs,
        ("training", "transformer_epochs"): args.transformer_epochs,
        ("training", "ablation_epochs"): args.ablation_epochs,
        ("training", "checkpoint_every"): args.checkpoint_every,
        ("training", "log_every"): args.log_every,
        ("training", "runtime_backend"): args.runtime_backend,
        ("training", "torch_device"): args.torch_device,
        ("outputs", "output_subdir"): args.output_subdir,
        ("dataset", "n_samples"): args.n_samples,
    }
    for key_path, value in overrides.items():
        if value is not None:
            _set_nested(config, key_path, value)
    if args.deep_epochs is not None:
        _set_nested(config, ("training", "force_deep_epochs"), True)
    if args.transformer_epochs is not None:
        _set_nested(config, ("training", "force_transformer_epochs"), True)
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CS3 multimodal emotion recognition benchmark.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--config", default="configs/cs3/default.yaml")
    parser.add_argument("--deep-epochs", type=int, default=None, help="Override deep model epochs from the command line.")
    parser.add_argument("--transformer-epochs", type=int, default=None, help="Override transformer model epochs from the command line.")
    parser.add_argument("--ablation-epochs", type=int, default=None, help="Override ablation epochs from the command line.")
    parser.add_argument("--checkpoint-every", type=int, default=None, help="Override checkpoint frequency from the command line.")
    parser.add_argument("--log-every", type=int, default=None, help="Log training progress every N epochs.")
    parser.add_argument("--runtime-backend", default=None, help="Choose CPU, GPU, or auto backend for deep and transformer models.")
    parser.add_argument("--torch-device", default=None, help="Torch device such as cpu, cuda, or cuda:0.")
    parser.add_argument("--output-subdir", default=None, help="Write CSV outputs into outputs/csv/cs3/<output-subdir>.")
    parser.add_argument("--n-samples", type=int, default=None, help="Override synthetic CS3 sample count.")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = _apply_cli_overrides(read_yaml(config_path), args)
    temp_config_path = Path(args.project_root).resolve() / "outputs" / "logs" / "paper1_cs3_runtime_config.json"
    write_json(temp_config_path, config)
    run_cs3(Path(args.project_root).resolve(), temp_config_path)


if __name__ == "__main__":
    main()
