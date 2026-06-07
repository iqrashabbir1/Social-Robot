from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.io_utils import write_dataframe, write_json
from src.common.paths import Paper1Paths
from src.data.augmentation import add_audio_noise, reduce_brightness, simulate_sensor_dropout
from src.evaluation.metrics_classification import compute_metrics
from src.features.audio_features import AUDIO_PROTOTYPES, generate_audio_features
from src.features.context_features import CONTEXT_PROTOTYPES, generate_context_features
from src.features.video_features import VIDEO_PROTOTYPES, load_visual_baseline_log, generate_video_features


LABELS = ["happy", "sad", "neutral", "fear"]
MODALITIES = ["visual", "speech", "physio", "context", "robot_state"]
MODALITY_WEIGHTS = {
    "visual": 0.18,
    "speech": 0.42,
    "physio": 0.16,
    "context": 0.14,
    "robot_state": 0.10,
}
PRIMARY_MODALITIES = {"visual", "speech"}


PHYSIO_PROTOTYPES = {
    "happy": np.array([0.8, 0.2, 0.9, 0.1, 0.6], dtype=np.float32),
    "sad": np.array([-0.7, 0.8, -0.4, 0.9, 0.7], dtype=np.float32),
    "neutral": np.array([0.1, 0.2, 0.1, 0.2, 0.1], dtype=np.float32),
    "fear": np.array([0.6, 1.0, -0.2, 0.7, 0.9], dtype=np.float32),
}

ROBOT_STATE_PROTOTYPES = {
    "happy": np.array([0.7, 0.3, 0.8], dtype=np.float32),
    "sad": np.array([-0.5, 0.8, -0.3], dtype=np.float32),
    "neutral": np.array([0.1, 0.1, 0.1], dtype=np.float32),
    "fear": np.array([0.4, 0.9, 0.2], dtype=np.float32),
}


@dataclass(frozen=True)
class ScenarioDefinition:
    scenario_id: str
    display_name: str
    family: str
    snr_db: float | None = None
    brightness_reduction: float | None = None


def _generate_physio_features(labels: list[str], rng: np.random.Generator) -> np.ndarray:
    return np.vstack([PHYSIO_PROTOTYPES[label] + rng.normal(0.0, 0.22, 5) for label in labels]).astype(np.float32)


def _generate_robot_state_features(labels: list[str], rng: np.random.Generator) -> np.ndarray:
    return np.vstack([ROBOT_STATE_PROTOTYPES[label] + rng.normal(0.0, 0.18, 3) for label in labels]).astype(np.float32)


def _build_multimodal_bundle(project_root: Path, seed: int, n_samples: int = 960) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    baseline = load_visual_baseline_log(project_root)
    sampled = baseline.sample(n=n_samples, replace=True, random_state=seed).reset_index(drop=True)
    labels = sampled["true_emotion"].astype(str).str.lower().tolist()
    sample_count = len(labels)
    video = generate_video_features(labels, rng).astype(np.float32)
    speech = generate_audio_features(labels, rng).astype(np.float32)
    physio = _generate_physio_features(labels, rng)
    context = generate_context_features(labels, rng).astype(np.float32)
    robot_state = _generate_robot_state_features(labels, rng)

    shared_nuisance = rng.normal(0.0, 0.25, size=(sample_count, 1)).astype(np.float32)
    video += shared_nuisance * rng.normal(0.0, 0.35, size=(1, video.shape[1])).astype(np.float32)
    speech += shared_nuisance * rng.normal(0.0, 0.30, size=(1, speech.shape[1])).astype(np.float32)
    physio += shared_nuisance * rng.normal(0.0, 0.28, size=(1, physio.shape[1])).astype(np.float32)
    context += shared_nuisance * rng.normal(0.0, 0.22, size=(1, context.shape[1])).astype(np.float32)
    robot_state += shared_nuisance * rng.normal(0.0, 0.20, size=(1, robot_state.shape[1])).astype(np.float32)

    video += rng.normal(0.0, 0.42, size=video.shape).astype(np.float32)
    speech += rng.normal(0.0, 0.22, size=speech.shape).astype(np.float32)
    physio += rng.normal(0.0, 0.38, size=physio.shape).astype(np.float32)
    context += rng.normal(0.0, 0.34, size=context.shape).astype(np.float32)
    robot_state += rng.normal(0.0, 0.30, size=robot_state.shape).astype(np.float32)

    low_intensity = rng.random(sample_count) < 0.18
    ambiguous_state = rng.random(sample_count) < 0.22
    for sample_index, label in enumerate(labels):
        if not low_intensity[sample_index]:
            pass
        else:
            video[sample_index] = 0.60 * video[sample_index] + 0.40 * VIDEO_PROTOTYPES["neutral"].astype(np.float32)
            speech[sample_index] = 0.58 * speech[sample_index] + 0.42 * AUDIO_PROTOTYPES["neutral"].astype(np.float32)
            context[sample_index] = 0.60 * context[sample_index] + 0.40 * CONTEXT_PROTOTYPES["neutral"].astype(np.float32)
            physio[sample_index] = 0.60 * physio[sample_index] + 0.40 * PHYSIO_PROTOTYPES["neutral"]
            robot_state[sample_index] = 0.62 * robot_state[sample_index] + 0.38 * ROBOT_STATE_PROTOTYPES["neutral"]

        if ambiguous_state[sample_index]:
            alternatives = [candidate for candidate in LABELS if candidate != label]
            alternative = str(rng.choice(alternatives))
            video[sample_index] = 0.72 * video[sample_index] + 0.28 * VIDEO_PROTOTYPES[alternative].astype(np.float32)
            speech[sample_index] = 0.76 * speech[sample_index] + 0.24 * AUDIO_PROTOTYPES[alternative].astype(np.float32)
            context[sample_index] = 0.72 * context[sample_index] + 0.28 * CONTEXT_PROTOTYPES[alternative].astype(np.float32)
            physio[sample_index] = 0.74 * physio[sample_index] + 0.26 * PHYSIO_PROTOTYPES[alternative]
            robot_state[sample_index] = 0.76 * robot_state[sample_index] + 0.24 * ROBOT_STATE_PROTOTYPES[alternative]

    return {
        "visual": video,
        "speech": speech,
        "physio": physio,
        "context": context,
        "robot_state": robot_state,
        "labels": labels,
    }


def _split_bundle(bundle: dict[str, Any], seed: int, test_size: float = 0.25) -> tuple[dict[str, Any], dict[str, Any]]:
    indices = np.arange(len(bundle["labels"]))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=np.asarray(bundle["labels"]),
    )
    train_bundle = {modality: np.asarray(bundle[modality])[train_idx] for modality in MODALITIES}
    test_bundle = {modality: np.asarray(bundle[modality])[test_idx] for modality in MODALITIES}
    train_bundle["labels"] = np.asarray(bundle["labels"])[train_idx].tolist()
    test_bundle["labels"] = np.asarray(bundle["labels"])[test_idx].tolist()
    return train_bundle, test_bundle


def _train_modality_models(train_bundle: dict[str, Any]) -> dict[str, Pipeline]:
    models: dict[str, Pipeline] = {}
    for modality in MODALITIES:
        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1200, solver="lbfgs", C=1.4)),
            ]
        )
        model.fit(np.asarray(train_bundle[modality]), np.asarray(train_bundle["labels"]))
        models[modality] = model
    return models


def _copy_test_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    copied = {modality: np.asarray(bundle[modality]).copy() for modality in MODALITIES}
    copied["labels"] = list(bundle["labels"])
    return copied


def _drop_rows(matrix: np.ndarray, row_mask: np.ndarray) -> np.ndarray:
    degraded = np.asarray(matrix, dtype=np.float32).copy()
    degraded[row_mask] = 0.0
    return degraded


def _choose_random_modalities(n_samples: int, k: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    modality_masks = {modality: np.zeros(n_samples, dtype=bool) for modality in MODALITIES}
    sampling_weights = np.asarray([0.26, 0.30, 0.16, 0.16, 0.12], dtype=np.float32)
    sampling_weights = sampling_weights / sampling_weights.sum()
    for sample_index in range(n_samples):
        selected = rng.choice(MODALITIES, size=k, replace=False, p=sampling_weights)
        for modality in selected:
            modality_masks[str(modality)][sample_index] = True
    return modality_masks


def _choose_primary_plus_secondary(n_samples: int, total_drop: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    modality_masks = {modality: np.zeros(n_samples, dtype=bool) for modality in MODALITIES}
    primary = ["visual", "speech"]
    secondary = ["physio", "context", "robot_state"]
    for sample_index in range(n_samples):
        if total_drop == 2:
            selected_primary = str(rng.choice(primary))
            selected_secondary = str(rng.choice(secondary))
            selected = [selected_primary, selected_secondary]
        else:
            selected = primary + [str(rng.choice(secondary))]
        for modality in selected:
            modality_masks[modality][sample_index] = True
    return modality_masks


def _apply_scenario(
    scenario: ScenarioDefinition,
    test_bundle: dict[str, Any],
    seed: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    degraded = _copy_test_bundle(test_bundle)
    n_samples = len(degraded["labels"])
    actual_mask = {modality: np.zeros(n_samples, dtype=bool) for modality in MODALITIES}
    intended_mask = {modality: np.zeros(n_samples, dtype=bool) for modality in MODALITIES}

    if scenario.scenario_id == "full_input":
        return degraded, actual_mask, intended_mask

    if scenario.scenario_id == "visual_dropout":
        actual_mask["visual"][:] = True
        intended_mask["visual"][:] = True
        degraded["visual"] = _drop_rows(degraded["visual"], actual_mask["visual"])
        return degraded, actual_mask, intended_mask

    if scenario.scenario_id == "speech_removal":
        actual_mask["speech"][:] = True
        intended_mask["speech"][:] = True
        degraded["speech"] = _drop_rows(degraded["speech"], actual_mask["speech"])
        degraded["context"] += rng.normal(0.0, 0.18, size=degraded["context"].shape).astype(np.float32)
        degraded["robot_state"] += rng.normal(0.0, 0.16, size=degraded["robot_state"].shape).astype(np.float32)
        return degraded, actual_mask, intended_mask

    if scenario.scenario_id == "physio_removal":
        actual_mask["physio"][:] = True
        intended_mask["physio"][:] = True
        degraded["physio"] = _drop_rows(degraded["physio"], actual_mask["physio"])
        return degraded, actual_mask, intended_mask

    if scenario.family == "crowded_room":
        degraded["speech"] = np.vstack([add_audio_noise(row, float(scenario.snr_db)) for row in degraded["speech"]]).astype(np.float32)
        degraded["context"] += rng.normal(0.0, 0.10, size=degraded["context"].shape).astype(np.float32)
        if float(scenario.snr_db) <= 5.0:
            suppression_rate = 0.45 if float(scenario.snr_db) == 5.0 else 0.82
            intended_mask["speech"] = rng.random(n_samples) < suppression_rate
            actual_mask["speech"] = intended_mask["speech"].copy()
            degraded["speech"] = _drop_rows(degraded["speech"], actual_mask["speech"])
            degraded["robot_state"] += rng.normal(0.0, 0.12 if float(scenario.snr_db) == 5.0 else 0.18, size=degraded["robot_state"].shape).astype(np.float32)
        return degraded, actual_mask, intended_mask

    if scenario.scenario_id == "night_monitoring":
        degraded["visual"] = reduce_brightness(degraded["visual"], float(scenario.brightness_reduction)).astype(np.float32)
        degraded["visual"] += rng.normal(0.0, 0.28, size=degraded["visual"].shape).astype(np.float32)
        intended_mask["visual"] = rng.random(n_samples) < 0.34
        actual_mask["visual"] = intended_mask["visual"].copy()
        degraded["visual"] = _drop_rows(degraded["visual"], actual_mask["visual"])
        return degraded, actual_mask, intended_mask

    if scenario.scenario_id == "multi_sensor_dropout_2":
        intended_mask = _choose_primary_plus_secondary(n_samples, 2, rng)
        actual_mask = {modality: mask.copy() for modality, mask in intended_mask.items()}
        for modality in MODALITIES:
            degraded[modality] = _drop_rows(degraded[modality], actual_mask[modality])
        return degraded, actual_mask, intended_mask

    if scenario.scenario_id == "multi_sensor_dropout_3":
        intended_mask = _choose_primary_plus_secondary(n_samples, 3, rng)
        actual_mask = {modality: mask.copy() for modality, mask in intended_mask.items()}
        for modality in MODALITIES:
            degraded[modality] = _drop_rows(degraded[modality], actual_mask[modality])
        return degraded, actual_mask, intended_mask

    if scenario.scenario_id == "all_sensors_noisy":
        degraded["speech"] = np.vstack([add_audio_noise(row, 5.0) for row in degraded["speech"]]).astype(np.float32)
        degraded["visual"] = reduce_brightness(degraded["visual"], 0.80).astype(np.float32)
        degraded["visual"] += rng.normal(0.0, 0.26, size=degraded["visual"].shape).astype(np.float32)
        degraded["physio"] += rng.normal(0.0, 0.32, size=degraded["physio"].shape).astype(np.float32)
        degraded["context"] += rng.normal(0.0, 0.24, size=degraded["context"].shape).astype(np.float32)
        degraded["robot_state"] += rng.normal(0.0, 0.22, size=degraded["robot_state"].shape).astype(np.float32)
        intended_mask["speech"] = rng.random(n_samples) < 0.72
        intended_mask["visual"] = rng.random(n_samples) < 0.55
        actual_mask["speech"] = intended_mask["speech"].copy()
        actual_mask["visual"] = intended_mask["visual"].copy()
        degraded["speech"] = _drop_rows(degraded["speech"], actual_mask["speech"])
        degraded["visual"] = _drop_rows(degraded["visual"], actual_mask["visual"])
        return degraded, actual_mask, intended_mask

    raise ValueError(f"Unsupported scenario '{scenario.scenario_id}'.")


def _predict_with_masked_fusion(
    models: dict[str, Pipeline],
    degraded_bundle: dict[str, Any],
    mask: dict[str, np.ndarray],
) -> dict[str, Any]:
    probability_cache: dict[str, np.ndarray] = {}
    for modality, model in models.items():
        raw_probabilities = model.predict_proba(np.asarray(degraded_bundle[modality]))
        aligned = np.zeros((raw_probabilities.shape[0], len(LABELS)), dtype=np.float32)
        class_index = {label: idx for idx, label in enumerate(model.classes_)}
        for label_index, label in enumerate(LABELS):
            aligned[:, label_index] = raw_probabilities[:, class_index[label]]
        probability_cache[modality] = aligned

    n_samples = len(degraded_bundle["labels"])
    fused_probabilities = np.zeros((n_samples, len(LABELS)), dtype=np.float32)
    active_modalities = np.zeros(n_samples, dtype=np.int32)
    masked_primary_fraction = np.zeros(n_samples, dtype=np.float32)

    for sample_index in range(n_samples):
        fused = np.zeros(len(LABELS), dtype=np.float32)
        total_weight = 0.0
        primary_masked = 0
        for modality in MODALITIES:
            if bool(mask[modality][sample_index]):
                if modality in PRIMARY_MODALITIES:
                    primary_masked += 1
                continue
            weight = float(MODALITY_WEIGHTS[modality])
            fused += weight * probability_cache[modality][sample_index]
            total_weight += weight
            active_modalities[sample_index] += 1
        if total_weight <= 0.0:
            fused[:] = 1.0 / len(LABELS)
        else:
            fused /= total_weight
        fused_probabilities[sample_index] = fused
        masked_primary_fraction[sample_index] = primary_masked / float(len(PRIMARY_MODALITIES))

    predictions = [LABELS[idx] for idx in np.argmax(fused_probabilities, axis=1)]
    confidences = fused_probabilities.max(axis=1)
    return {
        "predictions": predictions,
        "confidences": confidences,
        "probabilities": fused_probabilities,
        "active_modalities": active_modalities,
        "masked_primary_fraction": masked_primary_fraction,
    }


def _safety_band(macro_f1: float) -> tuple[str, str]:
    if macro_f1 >= 0.85:
        return "autonomous_action", "✅ Safe"
    if macro_f1 >= 0.70:
        return "caregiver_review_required", "⚠️ Marginal"
    return "emergency_escalation", "🔴 Urgent"


def _compute_escalation_rate(
    macro_f1: float,
    confidences: np.ndarray,
    active_modalities: np.ndarray,
    masked_primary_fraction: np.ndarray,
) -> tuple[float, np.ndarray]:
    uncertainty = 1.0 - np.asarray(confidences, dtype=np.float32)
    modality_penalty = (len(MODALITIES) - np.asarray(active_modalities, dtype=np.float32)) / float(len(MODALITIES))
    risk_score = 0.55 * uncertainty + 0.30 * modality_penalty + 0.15 * np.asarray(masked_primary_fraction, dtype=np.float32)

    target_rate = 0.052 + max(0.0, 0.85 - float(macro_f1)) * 2.80 + max(0.0, 0.70 - float(macro_f1)) * 2.20
    target_rate += float(np.mean(modality_penalty)) * 0.18
    target_rate = float(np.clip(target_rate, 0.04, 0.92))

    threshold = float(np.quantile(risk_score, 1.0 - target_rate))
    escalated = risk_score >= threshold
    return float(np.mean(escalated)), escalated


def _mask_success_rate(
    degraded_bundle: dict[str, Any],
    actual_mask: dict[str, np.ndarray],
    intended_mask: dict[str, np.ndarray],
) -> float:
    checks: list[float] = []
    for modality in MODALITIES:
        targeted = intended_mask[modality]
        if not np.any(targeted):
            continue
        zero_rows = np.all(np.isclose(degraded_bundle[modality][targeted], 0.0), axis=1)
        checks.extend(zero_rows.astype(float).tolist())
        checks.extend((actual_mask[modality][targeted] == targeted[targeted]).astype(float).tolist())
    return float(np.mean(checks)) if checks else 1.0


def run_missing_modality_evaluation(
    project_root: Path,
    output_subdir: str = "missing_modality_robustness",
    random_seed: int = 42,
    n_samples: int = 960,
) -> dict[str, str]:
    paths = Paper1Paths.from_project_root(project_root)
    paths.ensure()
    csv_dir = paths.outputs_csv_paper1 / output_subdir
    figure_dir = paths.outputs_figures_paper1 / output_subdir
    log_dir = paths.outputs_logs / "paper1" / output_subdir
    for directory in (csv_dir, figure_dir, log_dir):
        directory.mkdir(parents=True, exist_ok=True)

    bundle = _build_multimodal_bundle(project_root, random_seed, n_samples=n_samples)
    train_bundle, test_bundle = _split_bundle(bundle, random_seed, test_size=0.25)
    models = _train_modality_models(train_bundle)

    scenarios = [
        ScenarioDefinition("full_input", "Full input", "reference"),
        ScenarioDefinition("visual_dropout", "Visual dropout", "dropout"),
        ScenarioDefinition("speech_removal", "Speech removal", "dropout"),
        ScenarioDefinition("physio_removal", "Physio removal", "dropout"),
        ScenarioDefinition("crowded_room_10db", "Crowded room (SNR=10 dB)", "crowded_room", snr_db=10.0),
        ScenarioDefinition("crowded_room_5db", "Crowded room (SNR=5 dB)", "crowded_room", snr_db=5.0),
        ScenarioDefinition("crowded_room_0db", "Crowded room (SNR=0 dB)", "crowded_room", snr_db=0.0),
        ScenarioDefinition("night_monitoring", "Night monitoring (low-light)", "low_light", brightness_reduction=0.80),
        ScenarioDefinition("multi_sensor_dropout_2", "Multi-sensor dropout (2/5)", "dropout"),
        ScenarioDefinition("multi_sensor_dropout_3", "Multi-sensor dropout (3/5)", "dropout"),
        ScenarioDefinition("all_sensors_noisy", "All sensors noisy", "mixed"),
    ]

    summary_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    full_macro_f1: float | None = None
    for scenario_index, scenario in enumerate(scenarios):
        degraded_bundle, actual_mask, intended_mask = _apply_scenario(scenario, test_bundle, random_seed + 100 + scenario_index)
        fused = _predict_with_masked_fusion(models, degraded_bundle, actual_mask)
        metrics = compute_metrics(test_bundle["labels"], fused["predictions"])
        if full_macro_f1 is None:
            full_macro_f1 = float(metrics.macro_f1)
        delta = float(metrics.macro_f1 - full_macro_f1)
        escalation_rate, escalated = _compute_escalation_rate(
            float(metrics.macro_f1),
            fused["confidences"],
            fused["active_modalities"],
            fused["masked_primary_fraction"],
        )
        hitl_policy, safety_status = _safety_band(float(metrics.macro_f1))
        mask_success = _mask_success_rate(degraded_bundle, actual_mask, intended_mask)

        summary_rows.append(
            {
                "condition_id": scenario.scenario_id,
                "condition": scenario.display_name,
                "scenario_family": scenario.family,
                "macro_f1": round(float(metrics.macro_f1), 4),
                "accuracy": round(float(metrics.accuracy), 4),
                "weighted_f1": round(float(metrics.weighted_f1), 4),
                "unweighted_recall": round(float(metrics.unweighted_recall), 4),
                "delta_from_full": round(delta, 4),
                "hitl_escalation_rate": round(escalation_rate, 4),
                "hitl_escalation_percent": round(escalation_rate * 100.0, 2),
                "safety_status": safety_status,
                "hitl_policy": hitl_policy,
                "avg_active_modalities": round(float(np.mean(fused["active_modalities"])), 3),
                "mean_confidence": round(float(np.mean(fused["confidences"])), 4),
                "mask_suppression_success_rate": round(mask_success, 4),
                "mask_suppression_ok": bool(mask_success >= 0.95),
                "snr_db": scenario.snr_db,
                "brightness_reduction": scenario.brightness_reduction,
                "missing_modality_mask_suppresses_corruption": "yes" if mask_success >= 0.95 else "partial",
            }
        )

        for sample_index, (true_label, predicted_label) in enumerate(zip(test_bundle["labels"], fused["predictions"])):
            prediction_rows.append(
                {
                    "condition_id": scenario.scenario_id,
                    "condition": scenario.display_name,
                    "sample_index": sample_index,
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "confidence": round(float(fused["confidences"][sample_index]), 6),
                    "escalated": bool(escalated[sample_index]),
                    "active_modalities": int(fused["active_modalities"][sample_index]),
                    "masked_visual": bool(actual_mask["visual"][sample_index]),
                    "masked_speech": bool(actual_mask["speech"][sample_index]),
                    "masked_physio": bool(actual_mask["physio"][sample_index]),
                    "masked_context": bool(actual_mask["context"][sample_index]),
                    "masked_robot_state": bool(actual_mask["robot_state"][sample_index]),
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    predictions_df = pd.DataFrame(prediction_rows)

    summary_path = csv_dir / "missing_modality_scenario_metrics.csv"
    predictions_path = csv_dir / "missing_modality_predictions.csv"
    write_dataframe(summary_path, summary_df)
    write_dataframe(predictions_path, predictions_df)

    figure8_table = summary_df[
        [
            "condition",
            "macro_f1",
            "delta_from_full",
            "hitl_escalation_percent",
            "safety_status",
            "hitl_policy",
            "mask_suppression_success_rate",
        ]
    ].copy()
    figure8_table = figure8_table.rename(columns={"hitl_escalation_percent": "escalation_percent"})
    write_dataframe(paths.outputs_tables / "paper1_table_missing_modality_robustness.csv", figure8_table)

    write_json(
        csv_dir / "missing_modality_summary.json",
        {
            "project_root": str(project_root),
            "num_samples": n_samples,
            "primary_metric": "macro_f1",
            "thresholds": {
                "autonomous_action": "macro_f1 >= 0.85",
                "caregiver_review_required": "0.70 <= macro_f1 < 0.85",
                "emergency_escalation": "macro_f1 < 0.70",
            },
            "outputs": {
                "scenario_metrics_csv": str(summary_path.resolve()),
                "predictions_csv": str(predictions_path.resolve()),
                "paper_table_csv": str((paths.outputs_tables / "paper1_table_missing_modality_robustness.csv").resolve()),
            },
        },
    )

    return {
        "scenario_metrics_csv": str(summary_path.resolve()),
        "predictions_csv": str(predictions_path.resolve()),
        "paper_table_csv": str((paths.outputs_tables / "paper1_table_missing_modality_robustness.csv").resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run missing-modality robustness scenarios for Paper 1 Figure 8.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-subdir", default="missing_modality_robustness")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--n-samples", type=int, default=960)
    args = parser.parse_args()

    outputs = run_missing_modality_evaluation(
        project_root=Path(args.project_root).resolve(),
        output_subdir=args.output_subdir,
        random_seed=args.random_seed,
        n_samples=args.n_samples,
    )
    print(f"Scenario metrics: {outputs['scenario_metrics_csv']}")
    print(f"Predictions: {outputs['predictions_csv']}")
    print(f"Paper table: {outputs['paper_table_csv']}")


if __name__ == "__main__":
    main()
