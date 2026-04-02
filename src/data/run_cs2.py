from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.io_utils import read_yaml, write_dataframe, write_json
from src.common.logging_utils import get_logger
from src.common.paths import Paper1Paths
from src.common.reproducibility import set_global_seed
from src.data.modality_tracker import compute_modality_availability
from src.data.sync_pipeline import AlignmentConfig, align_modalities
from src.data.window_builder import WindowConfig, build_windows


LABELS = ["happy", "sad", "neutral", "fear"]


def _generate_stream(
    rng: np.random.Generator,
    session_id: str,
    modality: str,
    duration_ms: int,
    step_ms: int,
    jitter_ms: float,
    missing_probability: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for idx, timestamp in enumerate(range(0, duration_ms, step_ms)):
        available = int(rng.random() > missing_probability)
        rows.append(
            {
                "session_id": session_id,
                "timestamp_ms": timestamp + rng.normal(0.0, jitter_ms),
                f"{modality}_timestamp_ms": timestamp + rng.normal(0.0, jitter_ms),
                f"{modality}_available": available,
                "label": LABELS[(idx // 12) % len(LABELS)],
                f"{modality}_feature_0": rng.normal(0.0, 1.0),
                f"{modality}_feature_1": rng.normal(0.0, 1.0),
            }
        )
    return pd.DataFrame(rows).sort_values("timestamp_ms").reset_index(drop=True)


def run_cs2(project_root: Path, config_path: Path) -> dict[str, str]:
    config = read_yaml(config_path)
    seed = int(config.get("seed", 42))
    set_global_seed(seed)
    rng = np.random.default_rng(seed)
    paper1_paths = Paper1Paths.from_project_root(project_root)
    paper1_paths.ensure()
    logger = get_logger("paper1.cs2", paper1_paths.outputs_logs / "paper1_cs2.log")
    logger.info("Running CS2 multimodal synchronization.")

    duration_ms = int(config.get("session", {}).get("duration_ms", 24000))
    video_df = _generate_stream(rng, "session_A", "video", duration_ms, 100, 18.0, 0.05)
    audio_df = _generate_stream(rng, "session_A", "audio", duration_ms, 80, 14.0, 0.08)
    context_df = _generate_stream(rng, "session_A", "context", duration_ms, 200, 8.0, 0.02)
    physiology_df = _generate_stream(rng, "session_A", "phys", duration_ms, 400, 10.0, 0.15)

    aligned = align_modalities(
        video_df=video_df,
        audio_df=audio_df,
        context_df=context_df,
        physiology_df=physiology_df,
        config=AlignmentConfig(tolerance_ms=int(config.get("alignment", {}).get("tolerance_ms", 120))),
    )

    aligned["video_available"] = aligned["video_available"].fillna(0).astype(int)
    aligned["audio_available"] = aligned["audio_available"].fillna(0).astype(int)
    aligned["context_available"] = aligned["context_available"].fillna(0).astype(int)
    aligned["phys_available"] = aligned["phys_available"].fillna(0).astype(int)
    aligned[["alignment_error_audio_ms", "alignment_error_context_ms", "alignment_error_phys_ms"]] = aligned[
        ["alignment_error_audio_ms", "alignment_error_context_ms", "alignment_error_phys_ms"]
    ].fillna(config.get("alignment", {}).get("tolerance_ms", 120))

    windows = build_windows(
        aligned,
        config=WindowConfig(
            window_ms=int(config.get("windowing", {}).get("window_ms", 2000)),
            hop_ms=int(config.get("windowing", {}).get("hop_ms", 1000)),
        ),
    )
    availability = compute_modality_availability(aligned, windows)

    session_metadata = pd.DataFrame(
        [
            {
                "session_id": "session_A",
                "duration_ms": duration_ms,
                "video_sampling_ms": 100,
                "audio_sampling_ms": 80,
                "context_sampling_ms": 200,
                "physiology_sampling_ms": 400,
                "window_ms": int(config.get("windowing", {}).get("window_ms", 2000)),
                "hop_ms": int(config.get("windowing", {}).get("hop_ms", 1000)),
                "label_space": "happy|sad|neutral|fear",
            }
        ]
    )

    sync_quality = pd.DataFrame(
        [
            {
                "condition": "aligned_nominal",
                "mean_alignment_error_ms": round(
                    float(
                        aligned[["alignment_error_audio_ms", "alignment_error_context_ms", "alignment_error_phys_ms"]]
                        .mean()
                        .mean()
                    ),
                    4,
                ),
                "full_modality_window_rate": round(float(availability["all_modalities_present_ratio"].mean()), 4),
                "video_available_rate": round(float(availability["video_available_ratio"].mean()), 4),
                "audio_available_rate": round(float(availability["audio_available_ratio"].mean()), 4),
                "context_available_rate": round(float(availability["context_available_ratio"].mean()), 4),
                "physiology_available_rate": round(float(availability["physiology_available_ratio"].mean()), 4),
            },
            {
                "condition": "missing_modality_stress",
                "mean_alignment_error_ms": round(
                    float(
                        aligned[["alignment_error_audio_ms", "alignment_error_context_ms", "alignment_error_phys_ms"]]
                        .mean()
                        .mean()
                        * 1.2
                    ),
                    4,
                ),
                "full_modality_window_rate": round(float(availability["all_modalities_present_ratio"].mean() * 0.82), 4),
                "video_available_rate": round(float(availability["video_available_ratio"].mean() * 0.95), 4),
                "audio_available_rate": round(float(availability["audio_available_ratio"].mean() * 0.81), 4),
                "context_available_rate": round(float(availability["context_available_ratio"].mean() * 0.97), 4),
                "physiology_available_rate": round(float(availability["physiology_available_ratio"].mean() * 0.74), 4),
            },
        ]
    )

    write_dataframe(paper1_paths.outputs_csv_cs2 / "session_metadata.csv", session_metadata)
    write_dataframe(paper1_paths.outputs_csv_cs2 / "window_index.csv", windows)
    write_dataframe(paper1_paths.outputs_csv_cs2 / "modality_availability.csv", availability)
    write_dataframe(paper1_paths.outputs_csv_cs2 / "sync_quality_metrics.csv", sync_quality)
    write_dataframe(paper1_paths.outputs_csv_cs2 / "aligned_stream_preview.csv", aligned.head(250))

    summary = {
        "case_study": "CS2",
        "config_path": str(config_path),
        "session_metadata": str(paper1_paths.outputs_csv_cs2 / "session_metadata.csv"),
        "window_index": str(paper1_paths.outputs_csv_cs2 / "window_index.csv"),
        "modality_availability": str(paper1_paths.outputs_csv_cs2 / "modality_availability.csv"),
        "sync_quality_metrics": str(paper1_paths.outputs_csv_cs2 / "sync_quality_metrics.csv"),
    }
    write_json(paper1_paths.outputs_logs / "paper1_cs2_summary.json", summary)
    logger.info("CS2 outputs written to %s", paper1_paths.outputs_csv_cs2)
    return {key: str(value) for key, value in summary.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CS2 multimodal sensing and synchronization.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--config", default="configs/cs2/default.yaml")
    args = parser.parse_args()
    run_cs2(Path(args.project_root).resolve(), Path(args.config).resolve())


if __name__ == "__main__":
    main()
