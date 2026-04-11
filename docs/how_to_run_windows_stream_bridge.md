# How To Run The Windows Stream Mode In WSL

## Build And Source The WSL Workspace
```bash
cd ~/social_robot_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --packages-select social_robot
source install/setup.bash
```

## Launch The Hybrid Graph
```bash
WINDOWS_HOST_IP=$(awk '/nameserver/ {print $2; exit}' /etc/resolv.conf)

ros2 launch social_robot live_sensing.launch.py \
  config_path:=$HOME/social_robot_ws/src/social_robot/config/live_sensing.yaml \
  camera_input_mode:=windows_stream_bridge \
  runtime_type:=ros2_live_windows_stream_wsl_core \
  windows_camera_host:=${WINDOWS_HOST_IP} \
  windows_camera_port:=5001 \
  enable_audio:=false
```

## Verify The Camera Topic
```bash
ros2 topic list
ros2 topic echo /camera/image_raw
ros2 topic echo /system_health
```

## Notes
- In `windows_stream_bridge` mode, `camera_node` itself connects to the Windows TCP camera streamer.
- If the Windows stream is unavailable, `camera_node` stays alive and retries.
- No direct WSL webcam access is required in this mode.
