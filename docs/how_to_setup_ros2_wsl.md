# How To Set Up ROS 2 In WSL2

## Target Environment
- WSL2
- Ubuntu 24.04
- ROS 2 Jazzy
- Python `venv` or system Python, but no active Conda environment in the ROS shell

## Recommended Workspace Layout
```bash
mkdir -p ~/social_robot_ws/src
cd ~/social_robot_ws/src
git clone https://github.com/iqrashabbir1/Social-Robot.git social_robot
```

## Install ROS 2 Jazzy
Follow the official Jazzy installation for Ubuntu 24.04, then verify:
```bash
source /opt/ros/jazzy/setup.bash
ros2 --help
```

## Avoid Conda Conflicts
- Open a fresh shell that does not auto-activate Conda.
- If Conda activates automatically, run:
```bash
conda deactivate
```
- Confirm `which python` points to the ROS/WSL environment you intend to use.

## Python Dependencies
From `~/social_robot_ws/src/social_robot`:
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Notes
- `rclpy` comes from ROS 2 Jazzy, not from `pip`.
- The live ROS graph is intended for WSL2 Ubuntu, not native Windows PowerShell.
- Launching with `enable_audio:=false` skips `audio_node` entirely.
- Missing webcam or microphone devices in WSL do not crash the graph; the nodes stay alive and publish warning-level health status.
- If live devices are unavailable, playback-grounded mode remains the fallback path.
