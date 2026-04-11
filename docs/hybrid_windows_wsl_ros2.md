# Hybrid Windows-WSL ROS 2 For Paper 1

## Goal
Run webcam capture on Windows with plain Python + OpenCV, then stream frames to the Paper 1 ROS 2 graph running in WSL2 Ubuntu 24.04 + ROS 2 Jazzy.

## Runtime Split
- Windows:
  - `windows_nodes/camera_streamer.py`
  - captures webcam frames
  - sends JPEG-compressed frames over TCP
- WSL:
  - `windows_camera_bridge_node`
  - republishes frames to `/camera/image_raw`
  - `digital_twin_node`
  - `robot_state_node`
  - `emotion_inference_node`
  - `event_logger_node`
  - rosbag record/playback

## Why This Removes The Windows ROS 2 Requirement
- Windows does not need `rclpy`
- Windows does not need `sensor_msgs`
- Windows does not need `cv_bridge`
- only WSL needs ROS 2 Jazzy and `cv_bridge`

## Connection Model
- Windows camera streamer binds a TCP server on `host:port`
- WSL bridge node connects to that socket
- decoded frames are published into the WSL ROS 2 graph as `/camera/image_raw`

## Verification
From WSL after the Windows camera streamer starts and the bridge node is running:
```bash
ros2 topic list
ros2 node list
ros2 topic echo /camera/image_raw
```

Expected:
- `/camera/image_raw` is visible in WSL
- `windows_camera_bridge_node` appears in `ros2 node list`

## Fallback
- If the Windows stream is unavailable, the bridge node stays alive and keeps retrying.
- If the hybrid path is unavailable on the current machine, keep using `playback_grounded.launch.py`.
