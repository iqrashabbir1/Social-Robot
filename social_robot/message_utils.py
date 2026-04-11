from __future__ import annotations

import json
import time
from typing import Any

import cv2
import numpy as np


def now_ms() -> float:
    return time.time() * 1000.0


def image_to_rosmsg(frame_bgr: np.ndarray, stamp: Any, frame_id: str = "camera_frame") -> Any:
    from sensor_msgs.msg import Image

    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = int(frame_bgr.shape[0])
    msg.width = int(frame_bgr.shape[1])
    msg.encoding = "bgr8"
    msg.is_bigendian = False
    msg.step = int(frame_bgr.shape[1] * frame_bgr.shape[2])
    msg.data = frame_bgr.tobytes()
    return msg


def rosmsg_to_image(msg: Any) -> np.ndarray:
    frame = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    return frame.reshape((msg.height, msg.width, 3)).copy()


def float_audio_to_msg(audio: np.ndarray) -> Any:
    from std_msgs.msg import Float32MultiArray

    msg = Float32MultiArray()
    msg.data = audio.astype(np.float32).reshape(-1).tolist()
    return msg


def msg_to_float_audio(msg: Any) -> np.ndarray:
    return np.asarray(msg.data, dtype=np.float32)


def json_string_msg(payload: dict[str, Any]) -> Any:
    from std_msgs.msg import String

    msg = String()
    msg.data = json.dumps(payload)
    return msg


def decode_json_string(data: str) -> dict[str, Any]:
    try:
        value = json.loads(data)
        return value if isinstance(value, dict) else {"raw": value}
    except json.JSONDecodeError:
        return {"raw": data}


def resize_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
