from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.io_utils import read_yaml, write_dataframe, write_json
from src.common.logging_utils import get_logger
from src.common.paths import Paper1Paths
from src.common.reproducibility import set_global_seed
from src.digital_twin.fault_injection import apply_fault_profile, default_fault_profiles
from src.digital_twin.metrics import summarize_cs1_latency, summarize_fault_results
from src.digital_twin.twin_state import DigitalTwinState
from src.ros2.interface_spec import interface_spec_dataframe
from src.ros2.topic_logger import TopicEvent, TopicLogger


TOPIC_ORDER = [
    "/camera/image_raw",
    "/audio/stream",
    "/robot_pose",
    "/head_cmd",
    "/speech_cmd",
    "/event_log",
    "/system_health",
]


def _base_delay_for_mode(mode: str) -> float:
    return {
        "M1": 18.0,
        "M2": 27.0,
        "M3": 31.0,
        "M4": 39.0,
    }[mode]


def _simulate_mode(mode: str, steps: int, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    logger = TopicLogger()
    twin = DigitalTwinState()
    sync_rows: list[dict[str, object]] = []
    base_delay = _base_delay_for_mode(mode)
    failure_streak = 0

    for step in range(steps):
        nominal_time_ms = step * 100.0
        camera_ts = nominal_time_ms + rng.normal(0.0, 7.0)
        audio_ts = nominal_time_ms + rng.normal(0.0, 8.0)
        pose_ts = nominal_time_ms + rng.normal(0.0, 5.0)
        sync_error = max(camera_ts, audio_ts, pose_ts) - min(camera_ts, audio_ts, pose_ts)
        cpu_percent = 18.0 + {"M1": 5.0, "M2": 8.5, "M3": 6.5, "M4": 12.0}[mode] + rng.normal(0.0, 1.1)
        memory_mb = 185.0 + {"M1": 12.0, "M2": 19.0, "M3": 17.0, "M4": 24.0}[mode] + rng.normal(0.0, 2.8)
        success_flag = int(sync_error < {"M1": 24.0, "M2": 28.0, "M3": 30.0, "M4": 36.0}[mode])
        recovered_flag = int(failure_streak > 0 and success_flag == 1)
        failure_streak = 0 if success_flag == 1 else failure_streak + 1

        topic_payloads = {
            "/camera/image_raw": (camera_ts, "rgb_frame"),
            "/audio/stream": (audio_ts, "wav_chunk"),
            "/robot_pose": (pose_ts, "pose_state"),
            "/head_cmd": (nominal_time_ms + 10.0, "gaze_adjust"),
            "/speech_cmd": (nominal_time_ms + 12.0, "comfort_prompt"),
            "/event_log": (nominal_time_ms + 14.0, "sync_tick"),
            "/system_health": (nominal_time_ms + 16.0, "system_ok"),
        }

        for topic in TOPIC_ORDER:
            source_ts, payload_note = topic_payloads[topic]
            latency_ms = abs(rng.normal(base_delay, 4.0)) + (0.12 * sync_error)
            received_ts = source_ts + latency_ms
            mirrored_ts = received_ts + rng.normal(2.2, 0.8)
            payload_size = int(abs(rng.normal(256.0, 60.0)))
            twin.update(topic, mirrored_ts, {"command": payload_note, "health_state": "nominal"})
            logger.log(
                TopicEvent(
                    mode=mode,
                    step=step,
                    topic=topic,
                    source_timestamp_ms=round(float(source_ts), 4),
                    received_timestamp_ms=round(float(received_ts), 4),
                    mirrored_timestamp_ms=round(float(mirrored_ts), 4),
                    latency_ms=round(float(latency_ms), 4),
                    payload_size=payload_size,
                    dropped=0,
                    success_flag=success_flag,
                    recovered_flag=recovered_flag,
                    cpu_percent=round(float(cpu_percent), 4),
                    memory_mb=round(float(memory_mb), 4),
                    payload_note=payload_note,
                )
            )

        sync_rows.append(
            {
                "mode": mode,
                "step": step,
                "simulation_time_ms": round(float(nominal_time_ms), 4),
                "sync_error_ms": round(float(twin.synchronization_error_ms()), 4),
                "camera_audio_offset_ms": round(float(abs(camera_ts - audio_ts)), 4),
                "pose_command_offset_ms": round(float(abs(pose_ts - (nominal_time_ms + 10.0))), 4),
            }
        )

    return logger.to_dataframe(), pd.DataFrame(sync_rows)


def _simulate_fault_mode(steps: int, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    nominal_events, nominal_sync = _simulate_mode("M4", steps=steps, rng=rng)
    all_faulty_frames: list[pd.DataFrame] = []
    for profile in default_fault_profiles():
        faulty = apply_fault_profile(nominal_events, profile, rng)
        faulty["success_flag"] = (
            (faulty["latency_ms"] < (55.0 if profile.severity in {"mild", "moderate"} else 72.0)) & (faulty["dropped"] == 0)
        ).astype(int)
        faulty["recovered_flag"] = (
            (faulty["success_flag"] == 1) & (faulty.groupby("topic")["success_flag"].shift(fill_value=1) == 0)
        ).astype(int)
        all_faulty_frames.append(faulty)

    fault_events = pd.concat(all_faulty_frames, ignore_index=True)
    fault_sync_rows: list[pd.DataFrame] = []
    for profile in default_fault_profiles():
        profile_events = fault_events.loc[
            (fault_events["fault_type"] == profile.fault_type) & (fault_events["severity"] == profile.severity)
        ]
        merged = nominal_sync.copy()
        merged["fault_type"] = profile.fault_type
        merged["severity"] = profile.severity
        merged["sync_error_ms"] = merged["sync_error_ms"] + profile.delay_ms + rng.normal(
            0.0,
            max(profile.sync_noise_ms, 1.0),
            len(merged),
        )
        merged["message_drop_rate_proxy"] = float(profile_events["dropped"].mean())
        fault_sync_rows.append(merged)
    return fault_events, pd.concat(fault_sync_rows, ignore_index=True)


def run_cs1(project_root: Path, config_path: Path) -> dict[str, str]:
    config = read_yaml(config_path)
    seed = int(config.get("seed", 42))
    steps = int(config.get("simulation", {}).get("steps", 80))
    set_global_seed(seed)
    rng = np.random.default_rng(seed)

    paper1_paths = Paper1Paths.from_project_root(project_root)
    paper1_paths.ensure()
    logger = get_logger("paper1.cs1", paper1_paths.outputs_logs / "paper1_cs1.log")
    logger.info("Running CS1 digital-twin validation.")

    mode_frames: list[pd.DataFrame] = []
    sync_frames: list[pd.DataFrame] = []
    for mode in ("M1", "M2", "M3"):
        event_df, sync_df = _simulate_mode(mode, steps, rng)
        mode_frames.append(event_df)
        sync_frames.append(sync_df)

    fault_events, fault_sync = _simulate_fault_mode(steps, rng)
    mode_frames.append(fault_events)
    sync_frames.append(fault_sync)

    latency_df = pd.concat(mode_frames, ignore_index=True)
    sync_df = pd.concat(sync_frames, ignore_index=True)
    fault_df = summarize_fault_results(latency_df)
    summary_df = summarize_cs1_latency(latency_df)

    write_dataframe(paper1_paths.outputs_csv_cs1 / "latency_metrics.csv", latency_df)
    write_dataframe(paper1_paths.outputs_csv_cs1 / "sync_error_timeseries.csv", sync_df)
    write_dataframe(paper1_paths.outputs_csv_cs1 / "fault_injection_results.csv", fault_df)
    write_dataframe(paper1_paths.outputs_csv_cs1 / "latency_summary.csv", summary_df)
    write_dataframe(paper1_paths.outputs_csv_cs1 / "interface_spec.csv", interface_spec_dataframe())

    summary_payload = {
        "case_study": "CS1",
        "config_path": str(config_path),
        "latency_metrics": str(paper1_paths.outputs_csv_cs1 / "latency_metrics.csv"),
        "sync_error_timeseries": str(paper1_paths.outputs_csv_cs1 / "sync_error_timeseries.csv"),
        "fault_injection_results": str(paper1_paths.outputs_csv_cs1 / "fault_injection_results.csv"),
    }
    write_json(paper1_paths.outputs_logs / "paper1_cs1_summary.json", summary_payload)
    logger.info("CS1 outputs written to %s", paper1_paths.outputs_csv_cs1)
    return {key: str(value) for key, value in summary_payload.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CS1 ROS2 plus digital twin validation.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--config", default="configs/cs1/default.yaml")
    args = parser.parse_args()
    run_cs1(Path(args.project_root).resolve(), Path(args.config).resolve())


if __name__ == "__main__":
    main()
