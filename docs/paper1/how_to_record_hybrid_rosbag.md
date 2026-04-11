# How To Record Hybrid Rosbag

## Purpose
Record the live hybrid runtime so the Paper 1 evidence scripts can regenerate frame-rate, runtime, and sample-frame figures from a tracked session.

## WSL commands
Run the hybrid graph first:

```bash
cd ~/social_robot_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
WINDOWS_HOST_IP=$(ip route | awk '/default/ {print $3; exit}')

ros2 launch social_robot live_sensing.launch.py \
  config_path:=$HOME/social_robot_ws/src/social_robot/config/live_sensing.yaml \
  camera_input_mode:=windows_stream_bridge \
  runtime_type:=ros2_live_windows_stream_wsl_core \
  windows_camera_host:=${WINDOWS_HOST_IP} \
  windows_camera_port:=5001 \
  enable_audio:=false
```

In a second WSL shell, record the hybrid session:

```bash
cd ~/social_robot_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash

mkdir -p bags
ros2 bag record \
  /camera/image_raw \
  /robot_pose \
  /event_log \
  /system_health \
  /emotion_state \
  -o bags/paper1_hybrid_runtime
```

## Recommended export step
After recording, copy the bag or at least its `metadata.yaml` into the Windows project if you want the paper-figure scripts to run from the Windows repo without reaching into WSL directly.
