from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import yaml


def ros2_cli_available() -> bool:
    return shutil.which("ros2") is not None


def normalize_path(path: str | Path) -> Path:
    raw = str(path)
    expanded = os.path.expandvars(os.path.expanduser(raw))
    if "/~/" in expanded:
        expanded = expanded[expanded.index("/~/") + 1 :]
        expanded = os.path.expanduser(expanded)
    return Path(expanded).resolve()


def load_yaml(path: str | Path) -> dict[str, Any]:
    normalized = normalize_path(path)
    with normalized.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in YAML file: {normalized}")
    return data


def dump_json_line(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def ensure_rclpy() -> tuple[Any, Any]:
    try:
        import rclpy
        from rclpy.node import Node
    except ImportError as exc:  # pragma: no cover - depends on ROS2 runtime
        raise RuntimeError(
            "rclpy is not available. Run this package inside WSL2 Ubuntu 24.04 "
            "with ROS 2 Jazzy sourced, or use the existing playback-grounded/offline paths."
        ) from exc
    return rclpy, Node
