# ROS 2 Topic Interface Specification

| Topic | Message type | Publish frequency | Producer | Consumer | Notes |
|---|---|---:|---|---|---|
| `/camera/image_raw` | `sensor_msgs/Image` | 8-10 Hz | `camera_node` or `playback_adapter_node` | `digital_twin_node`, `emotion_inference_node`, `event_logger_node` | Laptop webcam, simulator, or playback source |
| `/audio/stream` | `std_msgs/Float32MultiArray` | 10 Hz equivalent chunks | `audio_node` or `playback_adapter_node` | `digital_twin_node`, `emotion_inference_node`, `event_logger_node` | Standard-message audio fallback, avoids custom audio dependency |
| `/robot_pose` | `geometry_msgs/PoseStamped` | 5 Hz | `robot_state_node` or future simulator/robot publisher | `digital_twin_node`, `emotion_inference_node`, `event_logger_node` | Placeholder laptop/demo state today; simulator/robot later |
| `/head_cmd` | `std_msgs/String` | 1-2 Hz | `digital_twin_node` or future controller | robot head controller, logger | Placeholder command topic for downstream integration |
| `/speech_cmd` | `std_msgs/String` | 1-2 Hz | `digital_twin_node` or future dialogue policy | TTS/controller, logger | Placeholder command topic for downstream integration |
| `/event_log` | `std_msgs/String` | event-driven | `digital_twin_node`, `playback_adapter_node` | `event_logger_node`, offline analysis | JSON-encoded structured events |
| `/system_health` | `diagnostic_msgs/DiagnosticArray` | 5 Hz | `robot_state_node` | `digital_twin_node`, `event_logger_node` | CPU/memory plus runtime label |
| `/emotion_state` | `std_msgs/String` | 1 Hz | `emotion_inference_node` | `digital_twin_node`, `event_logger_node` | JSON-encoded baseline emotion inference |

## Placeholder vs Real Signals
- `camera_node`: real laptop webcam when available
- `audio_node`: real laptop microphone when available
- `robot_state_node`: non-robot placeholder state today
- `playback_adapter_node`: rosbag-like or emulated topic source for playback-grounded validation
- future simulator publishers can replace `camera_node`, `audio_node`, or `robot_state_node` without changing downstream nodes
