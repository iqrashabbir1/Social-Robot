# ROS 2 Live Extension For Paper 1

Paper 1 now supports three runtime categories:
- `software_only`
- `ros2_playback_grounded`
- `ros2_live_laptop_sensors`

## What Is New
- a repo-root ROS 2 Python package for Jazzy/WSL2
- live sensing nodes for webcam, microphone, and placeholder context state
- live digital twin updates on the same Paper 1 topic graph
- live baseline emotion inference on ROS 2 topics
- structured CSV event logging inside the live graph

## What Is Still Not Claimed
- no physical robot deployment
- no clinical validation
- no claim that laptop sensors equal a caregiving robot in the field

## Why This Strengthens Paper 1
- the live graph shows that the Paper 1 architecture can operate as a real ROS 2 runtime
- the playback-grounded path remains valid for reproducible replay evaluation
- the offline benchmark stack remains separate and reproducible
