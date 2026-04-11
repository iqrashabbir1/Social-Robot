# How To Run Dataset Replay

## WSL ROS2 commands

```bash
cd ~/social_robot_ws
source /opt/ros/jazzy/setup.bash
python3 -m pip install --break-system-packages -r src/social_robot/requirements.txt
colcon build --symlink-install --packages-select social_robot
source install/setup.bash

ros2 launch social_robot dataset_replay.launch.py \
  config_path:=$HOME/social_robot_ws/src/social_robot/config/dataset_replay.yaml \
  runtime_type:=ros2_dataset_replay \
  enable_emotion:=true
```

## Recommended public-dataset replay configs
- `config/dataset_replay_rafdb.yaml` for RAF-DB image replay
- `config/dataset_replay_cremad.yaml` for CREMA-D frame replay

Example:

```bash
ros2 launch social_robot dataset_replay.launch.py \
  config_path:=$HOME/social_robot_ws/src/social_robot/config/dataset_replay_cremad.yaml \
  runtime_type:=ros2_dataset_replay \
  enable_emotion:=true
```

## What this does
- publishes dataset frames to `/camera/image_raw`
- keeps `digital_twin_node`, `event_logger_node`, and `emotion_inference_node` on the same ROS graph
- allows the existing Paper 1 runtime to consume controlled visual input

## If optional dependencies are missing
- `dataset_replay_node` now stays alive and logs a warning if `pandas` or other dataset dependencies are not installed yet.
- `emotion_inference_node` now stays alive and logs a warning if `deepface` or audio inference dependencies are unavailable.
- For the full Paper 1 replay path, install the Python requirements in the WSL ROS shell before launching.
