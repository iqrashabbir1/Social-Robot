# How To Verify The ROS 2 Runtime

## Doctor
```bash
source /opt/ros/jazzy/setup.bash
ros2 doctor
```

## Graph Checks
```bash
ros2 node list
ros2 topic list
ros2 topic echo /system_health
ros2 topic echo /emotion_state
```

## Expected Live Nodes
- `/camera_node`
- `/audio_node`
- `/robot_state_node`
- `/digital_twin_node`
- `/emotion_inference_node`
- `/event_logger_node`
- `/playback_adapter_node` in playback mode only

## What Counts As Live
- nodes present in `ros2 node list`
- topics present in `ros2 topic list`
- non-empty output on `/system_health`
- non-empty output on `/event_log`
- optional non-empty `/emotion_state` when emotion inference is enabled

## Graceful Degradation
- `enable_audio:=false` means `audio_node` is not launched at all.
- If webcam or microphone access is missing in WSL, the graph should stay alive and surface the issue on `/system_health`.
- If live sensing is not available, use `playback_grounded.launch.py`.
