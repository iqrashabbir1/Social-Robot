# Paper-Ready Assets

## Use these now
- `outputs/figures/paper1/paper_ready/hybrid_system_architecture_paper_ready.png`
- `outputs/figures/paper1/paper_ready/runtime_evidence_matrix_paper_ready.png`
- `outputs/figures/paper1/paper_ready/pilot_real_anchor_camera_panel.png`
- `outputs/figures/paper1/dataset_replay_sequence.png`
- `outputs/tables/paper1_table_runtime_evidence_summary.csv`
- `outputs/tables/paper1_table_mode_comparison.csv`
- `outputs/tables/paper1_table_dataset_summary.csv`
- `outputs/tables/paper1_table_runtime_vs_dataset_evidence.csv`
- `outputs/tables/paper1_table_paper_ready_assets.csv`

## Do not use yet
- `outputs/figures/paper1/hybrid_camera_fps_over_time.png`
- `outputs/figures/paper1/hybrid_camera_sample_frame.png`
- `outputs/figures/paper1/hybrid_camera_sample_panel.png`
- `outputs/figures/paper1/system_health_over_time.png`
- `outputs/figures/paper1/dataset_prediction_panel.png` when generated from the currently tracked local room-scene image set
- `outputs/figures/paper1/dataset_sample_panel.png` when generated from the currently tracked local room-scene image set
- `outputs/figures/paper1/dataset_confusion_matrix.png` unless a labeled dataset or label CSV is provided
- `outputs/figures/paper1/dataset_metrics_barplot.png` unless a labeled dataset or label CSV is provided
- `outputs/tables/paper1_table_dataset_metrics.csv` as a main-results table when the current local dataset is unlabeled

These excluded files are placeholder-safe outputs and should only be replaced after a fresh hybrid rosbag or event-log export has been copied into the repository.

## Dataset note
- `dataset_replay_sequence.png` is suitable now because it documents controlled replay through the ROS2 image pipeline.
- `dataset_prediction_panel.png` is generated correctly, but the currently tracked local image set is a small unlabeled room-scene pilot capture, so treat it as internal or supplementary only.
- `dataset_confusion_matrix.png`, `dataset_metrics_barplot.png`, and the main use of `paper1_table_dataset_metrics.csv` become strong paper evidence only after a labeled image dataset is provided.

## Regeneration
From the Windows project root:

```powershell
cd "C:\Users\masim\OneDrive\Desktop\Social Robot"
.\.venv\Scripts\Activate.ps1
python -m src.visualization.generate_paper_ready_assets --project-root .
```
