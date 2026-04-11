# What Is Real vs Synthetic

## Synthetic
- CS1 simulator-only
- CS2 placeholder synchronization sessions
- most trainable CS3 benchmark comparisons

## Pilot real-anchor
- locally collected session `paper1_anchor_demo`
- CS2 real-anchor synchronization export
- CS3 pilot baseline inference on the collected frames

## Playback-grounded
- CS1 playback-grounded replay using ROS2-compatible topic playback

## Mixed
- paper-level summaries that combine preserved baseline evidence with synthetic benchmark rows
## Runtime Categories
- `software_only`: offline scripts and non-ROS experiments
- `ros2_playback_grounded`: replay of recorded/emulated topic streams through the Paper 1 topic graph
- `ros2_live_laptop_sensors`: live ROS 2 runtime using webcam, microphone, and placeholder laptop/demo state
- `ros2_live_windows_stream_wsl_core`: Windows plain-Python camera streamer feeding a WSL ROS 2 bridge/core graph
- `ros2_live_simulator`: future simulator-backed live graph
- `ros2_live_robot`: future physical robot runtime, not claimed in Paper 1

## Current Status
- real webcam and microphone collection: yes
- real ROS 2 graph on this Windows machine: not yet verified
- rosbag-like playback grounding: yes
- physical robot: no
