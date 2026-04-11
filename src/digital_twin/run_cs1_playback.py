from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config_loader import build_experiment_context
from src.common.io_utils import write_dataframe, write_json, write_yaml
from src.common.logging_utils import get_logger
from src.common.reproducibility import set_global_seed
from src.digital_twin.metrics import summarize_cs1_latency
from src.digital_twin.run_cs1 import _simulate_mode
from src.ros2.bag_or_emulated_replay import load_or_emulate_topic_stream
from src.ros2.interface_spec import interface_spec_dataframe
from src.ros2.playback_adapter import replay_topic_stream
from src.visualization.plot_cs1 import plot_latency_distribution, plot_simulator_vs_playback, plot_sync_error


def run_cs1_playback(project_root: Path, config_path: Path) -> dict[str, str]:
    context = build_experiment_context(project_root, config_path)
    config = context.config
    set_global_seed(int(config["seed"]))
    logger = get_logger(f"paper1.cs1.{context.experiment_name}", context.log_path)
    logger.info("Running playback-grounded CS1 experiment '%s'.", context.experiment_name)

    steps = int(config.get("simulation", {}).get("steps", 80))
    replay_source = str(config.get("inputs", {}).get("replay_source", "")).strip()
    emulated_events, runtime_meta = load_or_emulate_topic_stream(context.project_root, replay_source, seed=int(config["seed"]), steps=steps)
    latency_df, sync_df, event_timing_df = replay_topic_stream(emulated_events, seed=int(config["seed"]))
    latency_summary = summarize_cs1_latency(latency_df)

    simulator_events, _ = _simulate_mode("M3", steps=steps, rng=np.random.default_rng(int(config["seed"])))
    simulator_summary = summarize_cs1_latency(simulator_events)
    comparison_df = pd.DataFrame(
        [
            {"source": "simulator_only", "mean_latency_ms": float(simulator_summary["mean_latency_ms"].mean())},
            {"source": "playback_grounded", "mean_latency_ms": float(latency_summary["mean_latency_ms"].mean())},
        ]
    )

    for frame in (latency_df, sync_df, event_timing_df, latency_summary, comparison_df):
        frame["data_source_type"] = str(config.get("evaluation", {}).get("data_source_type", "mixed"))
        frame["runtime_type"] = runtime_meta["runtime_type"]
        frame["model_status"] = str(config.get("evaluation", {}).get("model_status", "fully_runnable"))
        frame["evidence_level"] = str(config.get("evaluation", {}).get("evidence_level", "framework_validation"))

    write_yaml(context.config_snapshot_path, config)
    write_dataframe(context.csv_dir / "latency_metrics.csv", latency_df)
    write_dataframe(context.csv_dir / "sync_error_timeseries.csv", sync_df)
    write_dataframe(context.csv_dir / "event_timing.csv", event_timing_df)
    write_dataframe(context.metrics_csv_path, latency_summary)
    write_dataframe(context.csv_dir / "latency_summary.csv", latency_summary)
    write_dataframe(context.csv_dir / "simulator_vs_playback_comparison.csv", comparison_df)
    write_dataframe(context.csv_dir / "interface_spec.csv", interface_spec_dataframe())

    plot_latency_distribution(context.csv_dir / "latency_metrics.csv", context.figure_dir / "latency_distribution")
    plot_sync_error(context.csv_dir / "sync_error_timeseries.csv", context.figure_dir / "sync_error_over_time")
    plot_simulator_vs_playback(context.csv_dir / "simulator_vs_playback_comparison.csv", context.figure_dir / "simulator_vs_playback_comparison")

    summary = {
        "experiment_name": context.experiment_name,
        "case_study": context.case_study,
        "config_path": str(context.config_path),
        "runtime_type": runtime_meta["runtime_type"],
        "replay_mode": runtime_meta["replay_mode"],
        "latency_metrics": str(context.csv_dir / "latency_metrics.csv"),
        "sync_error_timeseries": str(context.csv_dir / "sync_error_timeseries.csv"),
        "event_timing": str(context.csv_dir / "event_timing.csv"),
        "comparison_csv": str(context.csv_dir / "simulator_vs_playback_comparison.csv"),
        "log_path": str(context.log_path),
    }
    write_json(context.summary_json_path, summary)
    logger.info("CS1 playback-grounded outputs written to %s", context.csv_dir)
    return {key: str(value) for key, value in summary.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CS1 playback-grounded validation with ROS2-compatible replay.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--config", default="configs/cs1/playback_grounded.yaml")
    args = parser.parse_args()
    run_cs1_playback(Path(args.project_root).resolve(), Path(args.config).resolve())


if __name__ == "__main__":
    main()
