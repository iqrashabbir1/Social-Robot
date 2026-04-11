from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in YAML file: {path}")
    return data


def write_yaml(path: Path, payload: Any) -> None:
    ensure_parent(path)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_dataframe(path: Path, dataframe: pd.DataFrame) -> None:
    ensure_parent(path)
    dataframe.to_csv(path, index=False)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    write_dataframe(path, pd.DataFrame(rows))
