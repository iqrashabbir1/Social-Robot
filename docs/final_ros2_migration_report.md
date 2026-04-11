# Final ROS 2 Migration Report

## Reused Existing Code
- Paper 1 playback-grounded utilities in `src/ros2/*`
- twin-state logic in `src/digital_twin/twin_state.py`
- real-anchor collection/loading in `src/data/*`
- baseline perception/fusion code in `perception/*` and `fusion/*`

## Converted Into ROS 2 Nodes
- `social_robot/camera_node.py`
- `social_robot/audio_node.py`
- `social_robot/robot_state_node.py`
- `social_robot/digital_twin_node.py`
- `social_robot/emotion_inference_node.py`
- `social_robot/event_logger_node.py`
- `social_robot/playback_adapter_node.py`

## Still Playback-Grounded Only
- the legacy offline CS1 playback scripts remain the runnable fallback on non-ROS machines
- rosbag replay on this Windows machine is not verified because the ROS 2 CLI is not available locally

## What Can Run Live Now
When the repo is placed in `~/social_robot_ws/src/social_robot` and built in WSL2 Ubuntu 24.04 with ROS 2 Jazzy:
- live camera/audio/context sensing
- live digital twin updates
- live baseline emotion inference
- structured event logging
- playback-grounded replay into the same topic graph
- `enable_audio:=false` cleanly skips `audio_node`
- missing webcam or microphone devices should not crash the graph

## What Requires WSL2 + Ubuntu 24.04 + ROS 2 Jazzy
- `colcon build`
- `ros2 launch social_robot ...`
- `ros2 bag record`
- `ros2 bag play`
- `ros2 node list` and `ros2 topic list`

## Remaining Gaps Before Simulator Integration
- replace placeholder state publisher with simulator publisher
- optionally replace laptop webcam/audio with simulator sensor publishers
- verify bag record/replay end-to-end under Jazzy

## Remaining Gaps Before Any Physical Robot Claim
- physical robot hardware integration
- real actuation on `/head_cmd` and `/speech_cmd`
- safety validation
- deployment testing
- ethics/clinical study design outside Paper 1
