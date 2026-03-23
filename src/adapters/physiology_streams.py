from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class PhysiologyPacket:
    timestamp: str
    heart_rate_bpm: float
    systolic_bp_mmHg: float
    diastolic_bp_mmHg: float
    spo2_percent: float
    gait_variability: float
    activity_index: float
    source: str


def load_physiology_csv(csv_path: Path, source_name: str = "csv_import") -> list[PhysiologyPacket]:
    df = pd.read_csv(csv_path)
    packets = [
        PhysiologyPacket(
            timestamp=str(row["timestamp"]),
            heart_rate_bpm=float(row["heart_rate_bpm"]),
            systolic_bp_mmHg=float(row["systolic_bp_mmHg"]),
            diastolic_bp_mmHg=float(row["diastolic_bp_mmHg"]),
            spo2_percent=float(row["spo2_percent"]),
            gait_variability=float(row["gait_variability"]),
            activity_index=float(row["activity_index"]),
            source=source_name,
        )
        for _, row in df.iterrows()
    ]
    return packets


def export_ros2_jsonl(packets: Iterable[PhysiologyPacket], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for packet in packets:
            handle.write(json.dumps(packet.__dict__) + "\n")
    return output_path


def build_real_adapter_manifest(project_root: Path) -> Path:
    rows = pd.DataFrame(
        [
            {
                "adapter": "csv_tail_adapter",
                "input_type": "wearable_csv",
                "status": "implemented",
                "notes": "Loads timestamped physiology traces from exported wearable CSV files.",
            },
            {
                "adapter": "jsonl_ros2_bridge",
                "input_type": "stream_packets",
                "status": "implemented",
                "notes": "Exports stream packets into JSONL for ROS2 replay or bridge publishers.",
            },
            {
                "adapter": "serial_ble_adapter",
                "input_type": "wearable_live_stream",
                "status": "stub_for_site_integration",
                "notes": "Integration point reserved for actual device SDK or BLE gateway code.",
            },
        ]
    )
    output_path = project_root / "outputs" / "tables" / "real_adapter_manifest.csv"
    rows.to_csv(output_path, index=False)
    return output_path
