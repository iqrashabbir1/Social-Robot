#!/usr/bin/env bash
set -euo pipefail

BAG_PATH="${1:-bags/paper1_live_session}"

ros2 bag play "${BAG_PATH}"
