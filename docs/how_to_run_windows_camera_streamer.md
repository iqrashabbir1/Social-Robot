# How To Run The Windows Camera Streamer

## Install Dependencies
```powershell
cd "C:\Users\masim\OneDrive\Desktop\Social Robot"
python -m pip install --upgrade pip
python -m pip install opencv-python
```

## Smoke-Test The Camera
```powershell
python .\windows_nodes\test_camera_only.py
```

## Run The Streamer
```powershell
python .\windows_nodes\camera_streamer.py --host 0.0.0.0 --port 5001 --camera-index 0 --frame-rate 10 --width 640 --height 480
```

## Notes
- This path does not require ROS 2 on Windows.
- The streamer waits for the WSL bridge to connect.
- Temporary frame failures do not terminate the process.
