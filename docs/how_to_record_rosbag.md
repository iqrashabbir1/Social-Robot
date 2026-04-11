# How To Record A Rosbag

## Command
```bash
cd ~/social_robot_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 bag record \
  /camera/image_raw \
  /audio/stream \
  /robot_pose \
  /event_log \
  /system_health \
  /emotion_state \
  -o bags/paper1_live_session
```

## Notes
- Record after launching `live_emotion_demo.launch.py` or `paper1_minimal.launch.py`.
- The resulting bag can be replayed into the same downstream graph.
- Paper 1 should label rosbag-based evaluation as `ros2_playback_grounded` unless the graph was live at collection time and explicitly described as `mixed`.
