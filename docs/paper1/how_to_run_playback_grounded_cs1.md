# How To Run Playback-Grounded CS1

## Check ROS2 runtime status
```powershell
cd "C:\Users\masim\OneDrive\Desktop\Social Robot"
.\.venv\Scripts\Activate.ps1
python scripts\check_ros2_runtime.py
```

## Run playback-grounded CS1
```powershell
python -m src.digital_twin.run_cs1_playback --project-root . --config configs/cs1/playback_grounded.yaml
```

Key outputs:
- `outputs/csv/cs1/playback_grounded/latency_metrics.csv`
- `outputs/csv/cs1/playback_grounded/sync_error_timeseries.csv`
- `outputs/csv/cs1/playback_grounded/event_timing.csv`
- `outputs/figures/cs1/playback_grounded/latency_distribution.png`
- `outputs/figures/cs1/playback_grounded/sync_error_over_time.png`
- `outputs/figures/cs1/playback_grounded/simulator_vs_playback_comparison.png`
