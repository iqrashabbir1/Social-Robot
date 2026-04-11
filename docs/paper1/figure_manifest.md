# Figure Manifest

## Paper 1 hybrid-runtime figures

| Figure | Output path | CSV source(s) | Script | Notes |
|---|---|---|---|---|
| Hybrid architecture | `outputs/figures/paper1/hybrid_system_architecture.png` and `.svg` | `outputs/csv/paper1/hybrid_system_architecture_nodes.csv`, `outputs/csv/paper1/hybrid_system_architecture_edges.csv` | `src/visualization/plot_hybrid_runtime.py` | Keep for the paper after minor visual cleanup; this is a methods figure, not an empirical result |
| Runtime verification | `outputs/figures/paper1/ros2_runtime_verification.png` and `.svg` | `outputs/csv/paper1/ros2_runtime_verification.csv` | `src/visualization/plot_hybrid_runtime.py` | Optional supplementary figure; better as a compact verification table in the main paper |
| Hybrid camera sample frame | `outputs/figures/paper1/hybrid_camera_sample_frame.png` | `outputs/csv/paper1/hybrid_camera_frame_manifest.csv` | `src/visualization/save_camera_sample_frames.py` | Do not include until replaced by a true hybrid frame export; current local file is placeholder-only |
| Hybrid camera sample panel | `outputs/figures/paper1/hybrid_camera_sample_panel.png` | `outputs/csv/paper1/hybrid_camera_frame_manifest.csv` | `src/visualization/save_camera_sample_frames.py` | Do not include until replaced by a true hybrid frame export; current local file is placeholder-only |
| Hybrid camera FPS over time | `outputs/figures/paper1/hybrid_camera_fps_over_time.png` and `.svg` | `outputs/csv/paper1/hybrid_camera_fps_timeseries.csv` | `src/visualization/plot_hybrid_runtime.py` | Do not include until a measured hybrid event-log export exists; current local file is placeholder-only |
| System health over time | `outputs/figures/paper1/system_health_over_time.png` and `.svg` | `outputs/csv/paper1/system_health_timeseries.csv` | `src/visualization/plot_hybrid_runtime.py` | Do not include until a measured hybrid system-health export exists; current local file is placeholder-only |
| Runtime mode comparison | `outputs/figures/paper1/runtime_mode_comparison.png` and `.svg` | `outputs/csv/paper1/runtime_mode_comparison.csv` | `src/visualization/plot_hybrid_runtime.py` | Keep only as supplementary or convert to a table; it is a qualitative framing artifact, not a primary result figure |

## Paper 1 dataset figures

| Figure | Output path | CSV source(s) | Script | Notes |
|---|---|---|---|---|
| Dataset sample panel | `outputs/figures/paper1/dataset_sample_panel.png` | `outputs/csv/paper1/dataset_eval/dataset_predictions.csv` | `src/visualization/plot_dataset_results.py` | Use only if the dataset content itself is paper-appropriate; the current tracked local room-scene pilot set is better kept out of the main manuscript |
| Dataset prediction panel | `outputs/figures/paper1/dataset_prediction_panel.png` | `outputs/csv/paper1/dataset_eval/dataset_predictions.csv` | `src/visualization/plot_dataset_results.py` | Use only with a paper-appropriate dataset; the current local room-scene set is not strong enough for a main qualitative figure |
| Dataset replay sequence | `outputs/figures/paper1/dataset_replay_sequence.png` | `outputs/csv/paper1/dataset_eval/dataset_sequence_manifest.csv` | `src/visualization/plot_dataset_results.py` | Recommended figure for showing dataset replay through the ROS2 pipeline |
| Dataset confusion matrix | `outputs/figures/paper1/dataset_confusion_matrix.png` | `outputs/csv/paper1/dataset_eval/dataset_confusion_matrix.csv` | `src/visualization/plot_dataset_results.py` | Use only when a labeled dataset or label CSV is provided |
| Dataset metrics barplot | `outputs/figures/paper1/dataset_metrics_barplot.png` | `outputs/csv/paper1/dataset_eval/dataset_metrics_summary.csv` | `src/visualization/plot_dataset_results.py` | Uses true accuracy/F1 metrics when labels exist; otherwise falls back to coverage/confidence metrics only |
