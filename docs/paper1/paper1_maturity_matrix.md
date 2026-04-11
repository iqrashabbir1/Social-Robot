# Paper 1 Maturity Matrix

| Module | Current maturity | Evidence label | Claim level |
| --- | --- | --- | --- |
| CS1 simulator-only | runnable | synthetic | framework validation |
| CS1 playback-grounded | runnable | playback-grounded | framework validation |
| CS2 synthetic sync | runnable | synthetic | framework validation |
| CS2 pilot real-anchor sync | runnable | pilot real-anchor | pilot demonstration |
| CS3 preserved baseline | runnable | mixed | benchmark preliminary |
| CS3 XGBoost/classical benchmark | runnable | synthetic | benchmark preliminary |
| CS3 deep/transformer benchmark | runnable | synthetic | benchmark preliminary |
| Offline dataset evaluation | runnable | offline_dataset_evaluation | benchmark preliminary |
| ROS2 dataset replay | runnable in Jazzy/WSL2 | ros2_dataset_replay | framework validation |
| Live ROS2 runtime | not available on this machine | not claimed | deployment not claimed |
| Real robot deployment | not implemented | not claimed | deployment not claimed |
## ROS 2 Runtime Extension

| Component | Current maturity | Runtime type | Claim boundary |
|---|---|---|---|
| Offline benchmark scripts | runnable | `software_only` | benchmark preliminary |
| Playback-grounded CS1 | runnable | `ros2_playback_grounded` | framework validation |
| Offline dataset evaluation | runnable | `offline_dataset_evaluation` | controlled perception evidence |
| Dataset replay through ROS2 | runnable in Jazzy/WSL2 | `ros2_dataset_replay` | replay-grounded pipeline validation |
| Live laptop-sensor ROS 2 package | implemented for Jazzy/WSL2 | `ros2_live_laptop_sensors` | integration demonstration only |
| Hybrid Windows streamer + WSL ROS 2 core | implemented as a config-driven path | `ros2_live_windows_stream_wsl_core` | integration demonstration only |
| Live simulator runtime | planned | `ros2_live_simulator` | future work |
| Live physical robot runtime | not implemented | `ros2_live_robot` | not claimed |
