#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${1:-bags/paper1_live_session}"

ros2 bag record \
  /camera/image_raw \
  /audio/stream \
  /robot_pose \
  /event_log \
  /system_health \
  /emotion_state \
  -o "${OUTPUT_DIR}"
