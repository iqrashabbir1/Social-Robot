from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hardware.live_validation import validate_live_baseline


def main() -> None:
    project_root = PROJECT_ROOT
    result = validate_live_baseline(project_root, audio_seconds=1, save_raw_artifacts=False)
    payload = asdict(result)
    output_path = project_root / "outputs" / "logs" / "local_device_status.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
