# How To Replay A Rosbag

## Native ROS 2 Replay
```bash
cd ~/social_robot_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 bag play bags/paper1_live_session
```

## Launch Downstream Graph In Another Shell
```bash
cd ~/social_robot_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch social_robot playback_grounded.launch.py \
  config_path:=~/social_robot_ws/src/social_robot/config/playback_grounded.yaml
```

## Fallback
- If a true rosbag is not available, `playback_adapter_node` can publish emulated topic streams using the same topic names.
