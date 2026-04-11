# How To Prepare CREMA-D For Paper 1

## Goal
Prepare CREMA-D for controlled dataset replay and optional offline evaluation in the `happy/sad/neutral/angry` label space.

## Expected local layout
Place CREMA-D under:

`data/public/CREMA-D`

The preparation script uses the filename emotion codes found in common CREMA-D media files.

## Generate the Paper 1 labels CSV
From the project root in PowerShell:

```powershell
cd "C:\Users\masim\OneDrive\Desktop\Social Robot"
.\.venv\Scripts\Activate.ps1
python .\scripts\prepare_cremad_labels.py `
  --dataset-root "C:\Users\masim\OneDrive\Desktop\Social Robot\data\public\CREMA-D" `
  --output-csv "C:\Users\masim\OneDrive\Desktop\Social Robot\data\public\CREMA-D\labels_broad4_angry.csv" `
  --target-label-set broad4_angry
```

## Run offline evaluation on CREMA-D frames
```powershell
python -m src.evaluation.run_dataset_evaluation `
  --project-root . `
  --dataset-root "data/public/CREMA-D" `
  --labels-csv "data/public/CREMA-D/labels_broad4_angry.csv" `
  --split-mode test_only `
  --output-subdir cremad_broad4_angry `
  --target-label-set broad4_angry
```

## Run ROS dataset replay on CREMA-D
In WSL:

```bash
cd ~/social_robot_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --packages-select social_robot
source install/setup.bash

ros2 launch social_robot dataset_replay.launch.py \
  config_path:=$HOME/social_robot_ws/src/social_robot/config/dataset_replay_cremad.yaml \
  runtime_type:=ros2_dataset_replay \
  enable_emotion:=true
```

## Notes
- the CREMA-D preparation path is especially useful for replay-through-ROS figures because it starts from replayable audiovisual clips
- non-target classes such as disgust and fear are dropped for the current Paper 1 four-class setting
