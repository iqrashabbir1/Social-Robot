from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.common.io_utils import write_dataframe, write_json
from src.common.paths import Paper1Paths


def _extract_topic_rows(metadata: dict[str, Any], rosbag_dir: Path) -> list[dict[str, Any]]:
    bag_info = metadata.get("rosbag2_bagfile_information", {})
    topics = bag_info.get("topics_with_message_count", []) or []
    rows: list[dict[str, Any]] = []
    for item in topics:
        topic_meta = item.get("topic_metadata", {}) or {}
        rows.append(
            {
                "rosbag_dir": str(rosbag_dir),
                "topic_name": topic_meta.get("name", ""),
                "message_type": topic_meta.get("type", ""),
                "serialization_format": topic_meta.get("serialization_format", ""),
                "offered_qos_profiles": topic_meta.get("offered_qos_profiles", ""),
                "message_count": item.get("message_count", 0),
                "source_status": "available",
            }
        )
    return rows


def extract_rosbag_summary(rosbag_dir: Path | None, project_root: Path) -> dict[str, str]:
    paths = Paper1Paths.from_project_root(project_root)
    paths.ensure()

    summary_json_path = paths.outputs_csv_paper1 / "hybrid_rosbag_summary.json"
    summary_csv_path = paths.outputs_csv_paper1 / "hybrid_rosbag_summary.csv"
    topics_csv_path = paths.outputs_csv_paper1 / "hybrid_rosbag_topics.csv"

    if rosbag_dir is None:
        write_dataframe(summary_csv_path, pd.DataFrame([{"source_status": "missing", "note": "No rosbag directory provided."}]))
        write_dataframe(topics_csv_path, pd.DataFrame([{"source_status": "missing", "note": "No rosbag directory provided."}]))
        write_json(
            summary_json_path,
            {
                "source_status": "missing",
                "rosbag_dir": None,
                "note": "No rosbag directory provided. Hybrid figure regeneration can still use CSV logs.",
            },
        )
        return {
            "summary_csv": str(summary_csv_path),
            "topics_csv": str(topics_csv_path),
            "summary_json": str(summary_json_path),
        }

    rosbag_dir = rosbag_dir.resolve()
    metadata_path = rosbag_dir / "metadata.yaml"
    if not metadata_path.exists():
        write_dataframe(summary_csv_path, pd.DataFrame([{"source_status": "missing", "rosbag_dir": str(rosbag_dir), "note": "metadata.yaml not found."}]))
        write_dataframe(topics_csv_path, pd.DataFrame([{"source_status": "missing", "rosbag_dir": str(rosbag_dir), "note": "metadata.yaml not found."}]))
        write_json(
            summary_json_path,
            {
                "source_status": "missing",
                "rosbag_dir": str(rosbag_dir),
                "note": "metadata.yaml not found.",
            },
        )
        return {
            "summary_csv": str(summary_csv_path),
            "topics_csv": str(topics_csv_path),
            "summary_json": str(summary_json_path),
        }

    metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}
    bag_info = metadata.get("rosbag2_bagfile_information", {})
    duration_ns = int(bag_info.get("duration", {}).get("nanoseconds", 0) or 0)
    starting_ns = int(bag_info.get("starting_time", {}).get("nanoseconds_since_epoch", 0) or 0)
    message_count = int(bag_info.get("message_count", 0) or 0)
    storage_identifier = str(bag_info.get("storage_identifier", ""))
    file_paths = bag_info.get("relative_file_paths", []) or []
    duration_sec = duration_ns / 1e9 if duration_ns else None

    topic_rows = _extract_topic_rows(metadata, rosbag_dir)
    summary_row = {
        "rosbag_dir": str(rosbag_dir),
        "source_status": "available",
        "storage_identifier": storage_identifier,
        "duration_sec": duration_sec,
        "message_count": message_count,
        "topic_count": len(topic_rows),
        "starting_time_ns": starting_ns,
        "file_count": len(file_paths),
        "runtime_type": "ros2_live_windows_stream_wsl_core",
        "data_source_type": "mixed",
        "evidence_level": "pilot_demonstration",
    }
    write_dataframe(summary_csv_path, pd.DataFrame([summary_row]))
    write_dataframe(topics_csv_path, pd.DataFrame(topic_rows))
    write_json(
        summary_json_path,
        {
            **summary_row,
            "relative_file_paths": file_paths,
            "topics": topic_rows,
        },
    )
    return {
        "summary_csv": str(summary_csv_path),
        "topics_csv": str(topics_csv_path),
        "summary_json": str(summary_json_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract a compact Paper 1 rosbag summary.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--rosbag-dir", default="")
    args = parser.parse_args()
    rosbag_dir = Path(args.rosbag_dir).resolve() if args.rosbag_dir else None
    extract_rosbag_summary(rosbag_dir=rosbag_dir, project_root=Path(args.project_root).resolve())


if __name__ == "__main__":
    main()
