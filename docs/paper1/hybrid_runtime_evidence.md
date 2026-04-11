# Hybrid Runtime Evidence

## Runtime type
- `ros2_live_windows_stream_wsl_core`

## Verified interpretation
- real webcam frames originate on Windows
- frames are streamed over TCP to WSL
- `camera_node` in `windows_stream_bridge` mode republishes them onto `/camera/image_raw`
- the downstream Paper 1 ROS 2 graph remains in WSL
- the runtime is suitable for a methods/platform paper as a live technical baseline

## Evidence boundary
- this is a hybrid live runtime, not a physical robot deployment
- it is stronger than playback-only because the image stream is live
- it remains a laptop-sensor and systems-methods demonstration
- hybrid frame-rate and system-health figures require exported logger CSVs or rosbag metadata to become fully populated in the repository
