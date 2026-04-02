from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.evaluation.export_results import export_paper1_tables


def test_export_results_creates_summary_tables(tmp_path: Path) -> None:
    (tmp_path / "outputs" / "csv" / "cs1").mkdir(parents=True)
    (tmp_path / "outputs" / "csv" / "cs2").mkdir(parents=True)
    (tmp_path / "outputs" / "csv" / "cs3").mkdir(parents=True)
    (tmp_path / "outputs" / "tables").mkdir(parents=True)

    pd.DataFrame([{"topic": "/camera/image_raw", "message_type": "sensor_msgs/Image", "producer": "sim", "consumer": "sync", "description": "test"}]).to_csv(
        tmp_path / "outputs" / "csv" / "cs1" / "interface_spec.csv",
        index=False,
    )
    pd.DataFrame([{"mode": "M1", "mean_latency_ms": 10, "p95_latency_ms": 15, "max_latency_ms": 20, "message_drop_rate": 0, "task_success_rate": 1, "recovery_rate": 0, "cpu_percent_mean": 20, "memory_mb_mean": 100}]).to_csv(
        tmp_path / "outputs" / "csv" / "cs1" / "latency_summary.csv",
        index=False,
    )
    pd.DataFrame([{"session_id": "s1", "duration_ms": 1000}]).to_csv(tmp_path / "outputs" / "csv" / "cs2" / "session_metadata.csv", index=False)
    pd.DataFrame([{"condition": "aligned_nominal", "mean_alignment_error_ms": 10, "full_modality_window_rate": 0.8, "video_available_rate": 0.9, "audio_available_rate": 0.8, "context_available_rate": 0.7, "physiology_available_rate": 0.6}]).to_csv(
        tmp_path / "outputs" / "csv" / "cs2" / "sync_quality_metrics.csv",
        index=False,
    )
    pd.DataFrame([{"model_id": "B0", "accuracy": 0.8, "macro_f1": 0.75, "weighted_f1": 0.78, "uar": 0.74, "inference_latency_ms": 1.2, "robustness_missing_modality_macro_f1": None, "evidence_level": "implemented_real_baseline"}]).to_csv(
        tmp_path / "outputs" / "csv" / "cs3" / "model_performance_summary.csv",
        index=False,
    )
    pd.DataFrame([{"model_id": "B1", "ablation_name": "video_only", "condition": "nominal", "accuracy": 0.8, "macro_f1": 0.77, "weighted_f1": 0.79, "uar": 0.76, "inference_latency_ms": 1.1, "evidence_level": "synthetic_placeholder_benchmark"}]).to_csv(
        tmp_path / "outputs" / "csv" / "cs3" / "ablation_results.csv",
        index=False,
    )

    result = export_paper1_tables(tmp_path)

    assert Path(result["paper1_table_system_summary"]).exists()
    assert Path(result["paper1_table_metrics_summary"]).exists()
    assert Path(result["paper1_table_ablation_summary"]).exists()
