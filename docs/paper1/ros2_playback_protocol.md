# ROS2 Playback Protocol

## Goal
Ground CS1 beyond software-only simulation by replaying recorded or emulated ROS2-compatible topic streams.

## Runtime modes
- preferred: real ROS2 or rosbag2 replay when available
- current machine: ROS2-compatible playback fallback

## Topics preserved
- `/camera/image_raw`
- `/audio/stream`
- `/robot_pose`
- `/head_cmd`
- `/speech_cmd`
- `/event_log`
- `/system_health`

## Current implementation
- `src/ros2/bag_or_emulated_replay.py`
- `src/ros2/playback_adapter.py`
- `src/digital_twin/run_cs1_playback.py`

## Claim boundary
Playback grounding improves runtime realism, but it is still not equivalent to a live deployed ROS2 robot stack on this machine.
