from __future__ import annotations

import json
import logging
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import psutil
except ImportError:  # pragma: no cover - optional runtime dependency
    psutil = None


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt_metric(value: float | None, suffix: str = "") -> str:
    return f"{value:.1f}{suffix}" if value is not None else "n/a"


def _capture_gpu_usage() -> dict[str, float | None]:
    if shutil.which("nvidia-smi") is None:
        return {
            "gpu_util_percent": None,
            "gpu_memory_used_mb": None,
            "gpu_memory_total_mb": None,
        }
    try:
        command = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
        result = subprocess.run(command, capture_output=True, text=True, timeout=2, check=True)
        line = next((item.strip() for item in result.stdout.splitlines() if item.strip()), "")
        if not line:
            raise ValueError("No GPU rows returned.")
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            raise ValueError("Unexpected GPU output format.")
        return {
            "gpu_util_percent": _safe_float(parts[0]),
            "gpu_memory_used_mb": _safe_float(parts[1]),
            "gpu_memory_total_mb": _safe_float(parts[2]),
        }
    except (subprocess.SubprocessError, ValueError):
        return {
            "gpu_util_percent": None,
            "gpu_memory_used_mb": None,
            "gpu_memory_total_mb": None,
        }


@dataclass
class RunTracker:
    run_tag: str
    progress_log_path: Path
    latest_status_path: Path
    epoch_progress_path: Path
    logger: logging.Logger
    log_every: int = 1

    def __post_init__(self) -> None:
        self._epoch_rows: list[dict[str, Any]] = []
        self._run_start = time.perf_counter()
        self._model_start_times: dict[str, float] = {}
        self.progress_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.latest_status_path.parent.mkdir(parents=True, exist_ok=True)
        self.epoch_progress_path.parent.mkdir(parents=True, exist_ok=True)
        if self.progress_log_path.exists():
            self.progress_log_path.unlink()
        if self.latest_status_path.exists():
            self.latest_status_path.unlink()
        if self.epoch_progress_path.exists():
            self.epoch_progress_path.unlink()
        self._process = psutil.Process() if psutil is not None else None
        if psutil is not None:
            psutil.cpu_percent(interval=None)
        if self._process is not None:
            self._process.cpu_percent(interval=None)

    def _capture_resources(self) -> dict[str, float | None]:
        cpu_system = psutil.cpu_percent(interval=None) if psutil is not None else None
        cpu_process = self._process.cpu_percent(interval=None) if self._process is not None else None
        memory_mb = (
            round(self._process.memory_info().rss / (1024 * 1024), 2)
            if self._process is not None
            else None
        )
        gpu_metrics = _capture_gpu_usage()
        return {
            "cpu_system_percent": _safe_float(cpu_system),
            "cpu_process_percent": _safe_float(cpu_process),
            "memory_mb": memory_mb,
            **gpu_metrics,
        }

    def event(self, event_type: str, **payload: Any) -> None:
        event_payload = {"run_tag": self.run_tag, "event_type": event_type, **payload}
        with self.progress_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event_payload) + "\n")
        self.latest_status_path.write_text(json.dumps(event_payload, indent=2), encoding="utf-8")

    def model_started(self, model_id: str, model_family: str, algorithm_name: str, total_epochs: int | None) -> None:
        self._model_start_times[model_id] = time.perf_counter()
        resource_snapshot = self._capture_resources()
        total_elapsed_seconds = round(time.perf_counter() - self._run_start, 3)
        self.event(
            "model_started",
            model_id=model_id,
            model_family=model_family,
            algorithm_name=algorithm_name,
            total_epochs=total_epochs,
            total_elapsed_seconds=total_elapsed_seconds,
            **resource_snapshot,
        )
        self.logger.info(
            "Training started | model=%s | family=%s | algorithm=%s | total_epochs=%s | total_elapsed=%.1fs | cpu=%s | mem=%s",
            model_id,
            model_family,
            algorithm_name,
            total_epochs,
            total_elapsed_seconds,
            _fmt_metric(resource_snapshot.get("cpu_system_percent"), "%"),
            _fmt_metric(resource_snapshot.get("memory_mb"), "MB"),
        )

    def epoch_progress(self, *, model_id: str, model_family: str, algorithm_name: str, epoch: int, total_epochs: int, metrics: dict[str, Any]) -> None:
        now = time.perf_counter()
        model_elapsed_seconds = round(now - self._model_start_times.get(model_id, self._run_start), 3)
        total_elapsed_seconds = round(now - self._run_start, 3)
        avg_epoch_seconds = model_elapsed_seconds / max(epoch, 1)
        estimated_remaining_seconds = round(avg_epoch_seconds * max(total_epochs - epoch, 0), 3)
        resource_snapshot = self._capture_resources()
        row = {
            "run_tag": self.run_tag,
            "model_id": model_id,
            "model_family": model_family,
            "algorithm_name": algorithm_name,
            "epoch": epoch,
            "total_epochs": total_epochs,
            "model_elapsed_seconds": model_elapsed_seconds,
            "estimated_remaining_seconds": estimated_remaining_seconds,
            "total_elapsed_seconds": total_elapsed_seconds,
            **metrics,
            **resource_snapshot,
        }
        self._epoch_rows.append(row)
        self.event("epoch_progress", **row)
        if epoch == 1 or epoch == total_epochs or epoch % max(self.log_every, 1) == 0:
            self.logger.info(
                "Epoch %s/%s | model=%s | val_macro_f1=%.4f | val_accuracy=%.4f | loss=%.6f | elapsed=%.1fs | eta=%.1fs | cpu=%s | mem=%s | gpu=%s",
                epoch,
                total_epochs,
                model_id,
                float(metrics.get("val_macro_f1", 0.0)),
                float(metrics.get("val_accuracy", 0.0)),
                float(metrics.get("loss", 0.0)),
                model_elapsed_seconds,
                estimated_remaining_seconds,
                _fmt_metric(resource_snapshot.get("cpu_system_percent"), "%"),
                _fmt_metric(resource_snapshot.get("memory_mb"), "MB"),
                _fmt_metric(resource_snapshot.get("gpu_util_percent"), "%"),
            )
        pd.DataFrame(self._epoch_rows).to_csv(self.epoch_progress_path, index=False)

    def model_completed(self, model_id: str, model_family: str, algorithm_name: str, metrics: dict[str, Any]) -> None:
        model_elapsed_seconds = round(time.perf_counter() - self._model_start_times.get(model_id, self._run_start), 3)
        total_elapsed_seconds = round(time.perf_counter() - self._run_start, 3)
        resource_snapshot = self._capture_resources()
        self.event(
            "model_completed",
            model_id=model_id,
            model_family=model_family,
            algorithm_name=algorithm_name,
            model_elapsed_seconds=model_elapsed_seconds,
            total_elapsed_seconds=total_elapsed_seconds,
            metrics=metrics,
            **resource_snapshot,
        )
        self.logger.info(
            "Training completed | model=%s | family=%s | algorithm=%s | macro_f1=%.4f | accuracy=%.4f | model_elapsed=%.1fs | total_elapsed=%.1fs",
            model_id,
            model_family,
            algorithm_name,
            float(metrics.get("macro_f1", 0.0)),
            float(metrics.get("accuracy", 0.0)),
            model_elapsed_seconds,
            total_elapsed_seconds,
        )

    def run_completed(self, best_model: dict[str, Any] | None = None) -> None:
        total_elapsed_seconds = round(time.perf_counter() - self._run_start, 3)
        resource_snapshot = self._capture_resources()
        self.event("run_completed", best_model=best_model or {}, total_elapsed_seconds=total_elapsed_seconds, **resource_snapshot)
        if best_model:
            self.logger.info(
                "Run completed | best_model=%s | algorithm=%s | macro_f1=%.4f | accuracy=%.4f | total_elapsed=%.1fs",
                best_model.get("model_id"),
                best_model.get("algorithm_name"),
                float(best_model.get("macro_f1", 0.0)),
                float(best_model.get("accuracy", 0.0)),
                total_elapsed_seconds,
            )
