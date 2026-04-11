# Next Stage Status Report

## Preserved baseline
- `ros2_playback_grounded` remains intact
- `ros2_live_laptop_sensors` remains intact
- `ros2_live_windows_stream_wsl_core` is preserved as the strongest live Paper 1 baseline

## New evidence layer
- added hybrid runtime metric collection
- added rosbag summary extraction
- added publication-style hybrid runtime plotting
- added sample-frame extraction utilities
- added updated Paper 1 manifests, caption drafts, and regeneration instructions

## Generated outputs
- architecture and runtime-verification figure scaffolds
- runtime mode comparison tables and figures
- hybrid metrics CSV and summary JSON scaffolds
- sample-frame export utility with explicit missing-source placeholders when no tracked hybrid frame export is present

## What is ready now
- the repository can regenerate Paper 1 figures and tables from copied hybrid rosbag metadata, event logs, and frame exports
- the live graph does not need to change to support the paper-evidence workflow
- future hybrid runs will automatically emit `ros2_system_health.csv` through the updated event logger

## Remaining blocker before a fully populated live-results section
- copy one real hybrid session export into the Windows repo or point the utilities to it
- then rerun the evidence scripts to replace the placeholder hybrid frame-rate and system-health outputs with measured values
