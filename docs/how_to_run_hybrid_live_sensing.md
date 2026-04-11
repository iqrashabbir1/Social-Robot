# How To Run Hybrid Live Sensing

## Windows Side
```powershell
cd "C:\Users\masim\OneDrive\Desktop\Social Robot"
python -m pip install --upgrade pip
python -m pip install opencv-python
python .\windows_nodes\test_camera_only.py
python .\windows_nodes\camera_streamer.py --host 0.0.0.0 --port 5001 --camera-index 0 --frame-rate 10 --width 640 --height 480
```

## WSL Side
```bash
cd ~/social_robot_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --packages-select social_robot
source install/setup.bash
WINDOWS_HOST_IP=$(awk '/nameserver/ {print $2; exit}' /etc/resolv.conf)

ros2 launch social_robot live_sensing.launch.py \
  config_path:=$HOME/social_robot_ws/src/social_robot/config/live_sensing.yaml \
  camera_input_mode:=windows_stream_bridge \
  runtime_type:=ros2_live_windows_stream_wsl_core \
  windows_camera_host:=${WINDOWS_HOST_IP} \
  windows_camera_port:=5001 \
  enable_audio:=false
```

Startup should include a line like:
```text
camera_node starting in windows_stream_bridge mode, host=X, port=Y
```

## Legacy Local Camera Mode
```bash
ros2 launch social_robot live_sensing.launch.py \
  config_path:=$HOME/social_robot_ws/src/social_robot/config/live_sensing.yaml \
  camera_input_mode:=local_camera \
  runtime_type:=ros2_live_laptop_sensors
```
