# Runtime Comparison Discussion

## Why the hybrid runtime is stronger than playback-only
- It preserves live camera sensing instead of relying exclusively on recorded or emulated input.
- It keeps the WSL ROS 2 graph active, which better matches the intended live runtime topology for later simulator and robot integration.
- It avoids the fragile direct-webcam-in-WSL dependency that limited the earlier `ros2_live_laptop_sensors` path.

## Why the hybrid runtime is still preliminary
- It remains a laptop-sensor platform demonstration rather than a robot deployment.
- The current evidence does not establish caregiving effectiveness, safety validation, or clinical utility.
- Paper 1 should treat the hybrid runtime as a stronger methods-platform baseline, not as a final embodied system.

## Recommended paper narrative
- Use `ros2_playback_grounded` to show repeatable non-live validation.
- Use `ros2_live_windows_stream_wsl_core` as the primary live technical baseline.
- Present `ros2_live_laptop_sensors` only as a retained legacy mode to show why the hybrid path is preferable.
