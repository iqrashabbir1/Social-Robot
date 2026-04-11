from __future__ import annotations

from pathlib import Path
from typing import Any

from social_robot.runtime import load_yaml, normalize_path


def resolve_ros_config(config_path: str | None, fallback_path: Path) -> dict[str, Any]:
    selected = normalize_path(config_path) if config_path else normalize_path(fallback_path)
    return load_yaml(selected)


def nested_ros_params(config: dict[str, Any], node_name: str) -> dict[str, Any]:
    section = config.get(node_name, {})
    ros_params = section.get("ros__parameters", {})
    if not isinstance(ros_params, dict):
        raise ValueError(f"Invalid ros__parameters for node '{node_name}'.")
    return ros_params
