# Live Vs Offline Architecture

## Live ROS 2 Runtime
Runs continuously as nodes:
- `camera_node`
- `audio_node`
- `robot_state_node`
- `digital_twin_node`
- `emotion_inference_node`
- `event_logger_node`
- `playback_adapter_node`

## Offline Components
Remain outside the live ROS graph:
- classical training
- deep training
- transformer training
- benchmark aggregation
- figure generation
- paper table export

## Why This Separation Matters
- live nodes must remain lightweight and debuggable
- training jobs should not block a real-time graph
- Paper 1 evidence can distinguish live runtime validation from offline benchmarking

## Model Export Path
- train offline
- export lightweight model artifact or reuse the existing baseline artifact
- load that artifact inside `emotion_inference_node` only for inference

## Evidence Mapping
- `software_only`: offline CS1/CS2/CS3 scripts
- `ros2_playback_grounded`: replay path using recorded/emulated topic streams
- `ros2_live_laptop_sensors`: live ROS 2 graph using webcam, microphone, and placeholder laptop/demo state
- `ros2_live_simulator`: future simulator-backed live graph
- `ros2_live_robot`: future physical robot runtime, not claimed in Paper 1
