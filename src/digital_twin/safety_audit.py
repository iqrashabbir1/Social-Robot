from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.common.io_utils import ensure_parent


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


@dataclass(frozen=True)
class AuditRecord:
    incident_id: str
    incident_type: str
    timestamp_ms: float
    sync_error_ms: float
    previous_hash: str
    signature: str
    state_snapshot: dict[str, Any]


class SafetyAuditor:
    def __init__(self, audit_log_path: Path) -> None:
        self.audit_log_path = audit_log_path.resolve()
        ensure_parent(self.audit_log_path)
        if not self.audit_log_path.exists():
            self.audit_log_path.write_text("", encoding="utf-8")

    def _read_records(self) -> list[dict[str, Any]]:
        lines = [line for line in self.audit_log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return [json.loads(line) for line in lines]

    def _last_hash(self) -> str:
        records = self._read_records()
        if not records:
            return "GENESIS"
        return str(records[-1]["signature"])

    def record_incident(
        self,
        incident_type: str,
        timestamp: float,
        state_snapshot: dict[str, Any],
        sync_error: float,
    ) -> str:
        previous_hash = self._last_hash()
        incident_payload = {
            "incident_id": f"{incident_type}_{int(timestamp)}_{len(self._read_records()) + 1:04d}",
            "incident_type": incident_type,
            "timestamp_ms": float(timestamp),
            "sync_error_ms": float(sync_error),
            "previous_hash": previous_hash,
            "state_snapshot": state_snapshot,
        }
        signature = hashlib.sha256(_canonical_json(incident_payload).encode("utf-8")).hexdigest()
        incident_payload["signature"] = signature
        with self.audit_log_path.open("a", encoding="utf-8") as handle:
            handle.write(_canonical_json(incident_payload) + "\n")
        return str(incident_payload["incident_id"])

    def replay_incident(self, incident_id: str) -> pd.DataFrame:
        for record in self._read_records():
            if str(record["incident_id"]) == str(incident_id):
                window = record.get("state_snapshot", {}).get("pre_incident_window", [])
                if not window:
                    return pd.DataFrame([record.get("state_snapshot", {})])
                return pd.DataFrame(window)
        raise KeyError(f"Incident '{incident_id}' was not found in the audit log.")

    def generate_audit_timeline(self) -> pd.DataFrame:
        records = self._read_records()
        if not records:
            return pd.DataFrame(
                columns=[
                    "incident_id",
                    "incident_type",
                    "timestamp_ms",
                    "sync_error_ms",
                    "previous_hash",
                    "signature",
                ]
            )
        return pd.DataFrame(
            [
                {
                    "incident_id": record["incident_id"],
                    "incident_type": record["incident_type"],
                    "timestamp_ms": record["timestamp_ms"],
                    "sync_error_ms": record["sync_error_ms"],
                    "previous_hash": record["previous_hash"],
                    "signature": record["signature"],
                }
                for record in records
            ]
        )

    def verify_chain(self) -> bool:
        previous_hash = "GENESIS"
        for record in self._read_records():
            record_copy = dict(record)
            signature = record_copy.pop("signature")
            if record_copy.get("previous_hash") != previous_hash:
                return False
            expected = hashlib.sha256(_canonical_json(record_copy).encode("utf-8")).hexdigest()
            if expected != signature:
                return False
            previous_hash = signature
        return True
