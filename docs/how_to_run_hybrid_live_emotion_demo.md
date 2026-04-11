# How To Run Hybrid Live Emotion Demo

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

ros2 launch social_robot live_emotion_demo.launch.py \
  config_path:=$HOME/social_robot_ws/src/social_robot/config/live_emotion_demo.yaml \
  camera_input_mode:=windows_stream_bridge \
  runtime_type:=ros2_live_windows_stream_wsl_core \
  windows_camera_host:=${WINDOWS_HOST_IP} \
  windows_camera_port:=5001 \
  enable_audio:=false \
  enable_emotion:=true
```

## Notes
- In hybrid mode, `camera_node` itself connects to the Windows TCP streamer and publishes `/camera/image_raw`.
- If audio dependencies are missing, keep `enable_audio:=false` and the demo runs in video-only mode.
