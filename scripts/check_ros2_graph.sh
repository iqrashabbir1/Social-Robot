#!/usr/bin/env bash
set -euo pipefail

echo "== ros2 doctor =="
ros2 doctor || true

echo
echo "== ros2 node list =="
ros2 node list || true

echo
echo "== ros2 topic list =="
ros2 topic list || true
