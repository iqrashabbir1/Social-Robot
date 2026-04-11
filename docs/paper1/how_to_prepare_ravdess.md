# How To Prepare RAVDESS For Paper 1

## Why use RAVDESS now
RAVDESS is the easiest public dataset to download and run immediately in this repo. It is a practical bridge between the current local pilot data and the larger RAF-DB / CREMA-D plan.

## Download a small runnable subset
From the project root in PowerShell:

```powershell
cd "C:\Users\masim\OneDrive\Desktop\Social Robot"
.\.venv\Scripts\Activate.ps1
python .\scripts\download_ravdess_subset.py `
  --output-root "C:\Users\masim\OneDrive\Desktop\Social Robot\data\public\RAVDESS" `
  --actors 01 02
```

This downloads the open-access RAVDESS video-speech zips for Actors 01 and 02 and extracts them locally.

## Generate the Paper 1 labels CSV
```powershell
python .\scripts\prepare_ravdess_labels.py `
  --dataset-root "C:\Users\masim\OneDrive\Desktop\Social Robot\data\public\RAVDESS" `
  --output-csv "C:\Users\masim\OneDrive\Desktop\Social Robot\data\public\RAVDESS\labels_broad4_angry.csv" `
  --target-label-set broad4_angry
```

## Run offline evaluation
```powershell
python -m src.evaluation.run_dataset_evaluation `
  --project-root . `
  --dataset-root "data/public/RAVDESS" `
  --labels-csv "data/public/RAVDESS/labels_broad4_angry.csv" `
  --split-mode test_only `
  --output-subdir ravdess_broad4_angry `
  --target-label-set broad4_angry
```

## Run ROS dataset replay in WSL
```bash
cd ~/social_robot_ws
source /opt/ros/jazzy/setup.bash
python3 -m pip install --break-system-packages -r src/social_robot/requirements.txt
colcon build --symlink-install --packages-select social_robot
source install/setup.bash

ros2 launch social_robot dataset_replay.launch.py \
  config_path:=$HOME/social_robot_ws/src/social_robot/config/dataset_replay_ravdess.yaml \
  runtime_type:=ros2_dataset_replay \
  enable_emotion:=true
```
