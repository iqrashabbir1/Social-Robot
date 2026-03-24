from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import sounddevice as sd
import soundfile as sf


@dataclass
class HardwareValidationResult:
    python_runtime: str
    webcam_accessible: bool
    microphone_accessible: bool
    webcam_frame_saved: bool
    microphone_sample_saved: bool
    deepface_available: bool
    speech_model_available: bool
    baseline_vision_executed: bool
    baseline_speech_executed: bool
    face_emotion: str | None
    speech_emotion: str | None
    notes: list[str]
    input_devices: list[dict[str, Any]]
    output_devices: list[dict[str, Any]]


def _deepface_available() -> bool:
    return importlib.util.find_spec("deepface") is not None


def _list_devices() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    devices = sd.query_devices()
    inputs: list[dict[str, Any]] = []
    outputs: list[dict[str, Any]] = []
    for index, device in enumerate(devices):
        entry = {
            "index": index,
            "name": device["name"],
            "max_input_channels": int(device["max_input_channels"]),
            "max_output_channels": int(device["max_output_channels"]),
            "default_samplerate": float(device["default_samplerate"]),
        }
        if device["max_input_channels"] > 0:
            inputs.append(entry)
        if device["max_output_channels"] > 0:
            outputs.append(entry)
    return inputs, outputs


def validate_live_baseline(
    project_root: Path,
    audio_seconds: int = 2,
    save_raw_artifacts: bool = False,
) -> HardwareValidationResult:
    notes: list[str] = []
    input_devices, output_devices = _list_devices()
    webcam_path = project_root / "outputs" / "logs" / "hardware_webcam_frame.jpg"
    audio_path = project_root / "outputs" / "logs" / "hardware_mic_sample.wav"
    webcam_path.parent.mkdir(parents=True, exist_ok=True)

    webcam_accessible = False
    webcam_frame_saved = False
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    frame = None
    if cap.isOpened():
        ret, frame = cap.read()
        webcam_accessible = bool(ret and frame is not None)
        if webcam_accessible:
            if save_raw_artifacts:
                cv2.imwrite(str(webcam_path), frame)
                webcam_frame_saved = True
        else:
            notes.append("Webcam opened but no frame could be captured.")
        cap.release()
    else:
        notes.append("No accessible webcam device was detected for OpenCV capture.")

    microphone_accessible = len(input_devices) > 0
    microphone_sample_saved = False
    audio_signal = None
    if microphone_accessible:
        try:
            samplerate = int(input_devices[0]["default_samplerate"])
            audio_signal = sd.rec(int(audio_seconds * samplerate), samplerate=samplerate, channels=1, dtype="float32")
            sd.wait()
            sf.write(audio_path, audio_signal, samplerate)
            microphone_sample_saved = True
        except Exception as exc:
            notes.append(f"Microphone capture failed: {exc}")
    else:
        notes.append("No accessible microphone device was detected.")

    deepface_available = _deepface_available()
    speech_model_available = (project_root / "data" / "speech" / "speech_svm_crema_balanced.joblib").exists()
    baseline_vision_executed = False
    baseline_speech_executed = False
    face_emotion: str | None = None
    speech_emotion: str | None = None

    if webcam_accessible and deepface_available:
        try:
            from perception.face_emotion import detect_face_emotion_from_frame

            face_emotion, _ = detect_face_emotion_from_frame(frame, enforce_detection=False)
            baseline_vision_executed = True
        except Exception as exc:
            notes.append(f"DeepFace baseline vision execution failed: {exc}")
    elif webcam_accessible and not deepface_available:
        notes.append("Webcam capture succeeded, but DeepFace is not installed in the project environment.")

    if microphone_sample_saved and speech_model_available:
        try:
            from perception.speech_emotion import predict_speech_emotion_robust_3class

            speech_emotion = predict_speech_emotion_robust_3class(str(audio_path))
            baseline_speech_executed = True
        except Exception as exc:
            notes.append(f"Speech baseline execution failed: {exc}")

    if not save_raw_artifacts:
        if webcam_path.exists():
            webcam_path.unlink()
        if audio_path.exists():
            audio_path.unlink()
        webcam_frame_saved = False
        microphone_sample_saved = False

    return HardwareValidationResult(
        python_runtime=str(Path(__file__).resolve()),
        webcam_accessible=webcam_accessible,
        microphone_accessible=microphone_accessible,
        webcam_frame_saved=webcam_frame_saved,
        microphone_sample_saved=microphone_sample_saved,
        deepface_available=deepface_available,
        speech_model_available=speech_model_available,
        baseline_vision_executed=baseline_vision_executed,
        baseline_speech_executed=baseline_speech_executed,
        face_emotion=face_emotion,
        speech_emotion=speech_emotion,
        notes=notes,
        input_devices=input_devices,
        output_devices=output_devices,
    )


def write_validation_artifacts(project_root: Path, save_raw_artifacts: bool = False) -> Path:
    result = validate_live_baseline(project_root, save_raw_artifacts=save_raw_artifacts)
    output_path = project_root / "outputs" / "logs" / "hardware_validation_summary.json"
    output_path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    return output_path
