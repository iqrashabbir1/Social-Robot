# Runtime Baseline Freeze

## Paper 1 baseline runtime
- Primary baseline runtime: `ros2_live_windows_stream_wsl_core`
- Secondary controlled fallback: `ros2_playback_grounded`
- Legacy optional live mode retained: `ros2_live_laptop_sensors`

## Verified baseline facts
- WSL ROS 2 Jazzy live runtime has been verified by the project user in the WSL workspace.
- Windows native webcam streaming over TCP has been verified.
- `camera_node` in `windows_stream_bridge` mode has been verified.
- `/camera/image_raw` publication has been verified in the hybrid runtime.
- rosbag recording support remains part of the hybrid runtime workflow.
- playback-grounded mode remains available as the controlled fallback.

## What remains partial
- The local Windows repository does not currently include a tracked hybrid `ros2_event_log.csv`.
- The local Windows repository does not currently include a tracked hybrid `ros2_system_health.csv`.
- A tracked hybrid rosbag export is not currently present in the repository.
- Full live emotion-demo verification still requires a fresh hybrid session export.

## Paper 1 interpretation
- Treat `ros2_live_windows_stream_wsl_core` as the strongest live technical baseline for Paper 1.
- Treat `ros2_playback_grounded` as the repeatable non-live comparison baseline.
- Do not treat any runtime in this repository as a deployed caregiving robot runtime.
- Do not treat any runtime in this repository as clinically validated.
