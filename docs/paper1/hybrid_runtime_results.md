# Hybrid Runtime Results

## Scope
This section summarizes the paper-evidence layer built on top of the verified hybrid runtime:
- Windows webcam capture via `windows_nodes/camera_streamer.py`
- WSL ROS 2 core graph via `camera_node`, `digital_twin_node`, `emotion_inference_node`, and `event_logger_node`
- runtime label: `ros2_live_windows_stream_wsl_core`

## What is already evidenced
- the hybrid runtime architecture is documented and exported as CSV-backed figures
- the runtime comparison table distinguishes playback, legacy live laptop sensing, and hybrid Windows-stream operation
- the figure-generation layer can regenerate publication-style figures from rosbag metadata and logger CSVs

## What should be used in the paper right now
- use the hybrid architecture figure as the primary runtime-methods figure
- use the runtime evidence summary table as the main verification artifact
- treat the runtime comparison figure as optional supplementary material, or convert it into a compact table

## What should not be used in the paper right now
- do not include `hybrid_camera_sample_frame.png`
- do not include `hybrid_camera_sample_panel.png`
- do not include `hybrid_camera_fps_over_time.png`
- do not include `system_health_over_time.png`
- these current local outputs are placeholders because no tracked hybrid event-log, system-health CSV, or frame export has been copied into the repository

## What still depends on a fresh hybrid export
- measured frame-rate over time from a tracked hybrid `ros2_event_log.csv`
- measured CPU and memory traces from a tracked hybrid `ros2_system_health.csv`
- a true hybrid sample-frame panel exported from `/camera/image_raw` or a hybrid rosbag

## Interpretation for Paper 1
- the hybrid runtime is stronger than playback-only because it validates a live camera transport path and a live ROS 2 core graph
- the hybrid runtime remains a laptop-sensor and methods-platform demonstration
- deployment, simulator coupling, and physical robot embodiment remain future work
