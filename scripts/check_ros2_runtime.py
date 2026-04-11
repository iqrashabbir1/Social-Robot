from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


def main() -> None:
    ros2_path = shutil.which("ros2")
    payload = {
        "ros2_available_on_path": ros2_path is not None,
        "ros2_path": ros2_path or "",
        "runtime_type": "ros2_live_laptop_sensors" if ros2_path else "ros2_playback_grounded",
        "fallback_reason": "",
    }
    if ros2_path:
        try:
            completed = subprocess.run([ros2_path, "--help"], capture_output=True, text=True, timeout=10, check=False)
            payload["help_exit_code"] = completed.returncode
        except Exception as exc:  # pragma: no cover - defensive runtime check
            payload["fallback_reason"] = str(exc)
    else:
        payload["fallback_reason"] = "ROS2 CLI not found on PATH; playback will use ROS2-compatible emulation."

    output_path = Path("outputs") / "logs" / "ros2_runtime_status.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
