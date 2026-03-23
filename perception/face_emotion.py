from __future__ import annotations
from typing import Optional, Tuple
from deepface import DeepFace
import numpy as np


def detect_face_emotion_from_frame(
    frame_bgr: np.ndarray,
    enforce_detection: bool = False,
) -> Tuple[Optional[str], Optional[float]]:
    """
    Run DeepFace emotion analysis on a single BGR frame (from OpenCV).

    Returns:
      (dominant_emotion, confidence) or (None, None) if detection fails.

    Notes:
      - DeepFace.analyze can take numpy arrays (BGR) when cv2 is installed.
      - enforce_detection=False prevents hard crashes when no face is found.
    """
    try:
        # DeepFace will internally handle color format (BGR/RGB) when using OpenCV backend.
        result = DeepFace.analyze(
            frame_bgr,
            actions=["emotion"],
            enforce_detection=enforce_detection,
        )

        # DeepFace >= 0.0.78 returns a list/dict; handle both
        if isinstance(result, list):
            result = result[0]

        emo_dict = result.get("emotion", {})
        dominant = result.get("dominant_emotion", None)

        confidence = None
        if dominant is not None and emo_dict and dominant in emo_dict:
            confidence = float(emo_dict[dominant])

        return dominant, confidence

    except Exception as e:
        print("Error in detect_face_emotion_from_frame:", e)
        return None, None
