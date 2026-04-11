from __future__ import annotations

import sys

import cv2


def main() -> None:
    camera_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"FAILED: camera index {camera_index} could not be opened.")
        raise SystemExit(1)

    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        print(f"FAILED: camera index {camera_index} opened but no frame was captured.")
        raise SystemExit(2)

    print(f"OK: captured frame {frame.shape[1]}x{frame.shape[0]} from camera index {camera_index}.")


if __name__ == "__main__":
    main()
