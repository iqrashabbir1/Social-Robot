#!/usr/bin/env bash
set -euo pipefail

echo "== /system_health =="
timeout 5 ros2 topic echo /system_health || true

echo
echo "== /emotion_state =="
timeout 5 ros2 topic echo /emotion_state || true
