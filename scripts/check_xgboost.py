from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    payload = {"available": False, "version": "", "status": "optional_not_installed"}
    try:
        import xgboost  # type: ignore

        payload["available"] = True
        payload["version"] = getattr(xgboost, "__version__", "unknown")
        payload["status"] = "fully_runnable"
    except Exception as exc:  # pragma: no cover - defensive runtime check
        payload["error"] = str(exc)

    output_path = Path("outputs") / "logs" / "xgboost_status.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
