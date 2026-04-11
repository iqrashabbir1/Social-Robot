from __future__ import annotations

import argparse
import socket
import struct
import time

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plain Python Windows camera streamer for WSL ROS2 bridge.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--frame-rate", type=float, default=10.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--jpeg-quality", type=int, default=85)
    return parser.parse_args()


def open_camera(camera_index: int, width: int, height: int):
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def main() -> None:
    args = parse_args()
    cap = open_camera(args.camera_index, args.width, args.height)
    if not cap.isOpened():
        print(f"WARNING: webcam index {args.camera_index} could not be opened. Waiting and retrying.")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((args.host, args.port))
        server.listen(1)
        print(f"Windows camera streamer listening on {args.host}:{args.port}")

        while True:
            if not cap.isOpened():
                cap.release()
                time.sleep(2.0)
                cap = open_camera(args.camera_index, args.width, args.height)
                continue

            print("Waiting for WSL bridge client...")
            conn, addr = server.accept()
            print(f"Bridge connected from {addr[0]}:{addr[1]}")
            with conn:
                conn.settimeout(5.0)
                while True:
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        print("WARNING: temporary frame read failure; retrying.")
                        time.sleep(0.2)
                        continue

                    frame = cv2.resize(frame, (args.width, args.height), interpolation=cv2.INTER_AREA)
                    encode_ok, encoded = cv2.imencode(
                        ".jpg",
                        frame,
                        [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)],
                    )
                    if not encode_ok:
                        print("WARNING: JPEG encoding failed; skipping frame.")
                        time.sleep(0.05)
                        continue

                    payload = encoded.tobytes()
                    packet = struct.pack("!I", len(payload)) + payload
                    try:
                        conn.sendall(packet)
                    except OSError as exc:
                        print(f"WARNING: bridge connection lost: {exc}")
                        break
                    time.sleep(max(0.0, 1.0 / max(args.frame_rate, 0.1)))


if __name__ == "__main__":
    main()
