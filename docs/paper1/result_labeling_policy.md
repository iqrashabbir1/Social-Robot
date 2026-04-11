# Result Labeling Policy

Every Paper 1 result must carry:
- `data_source_type`
- `runtime_type`
- `model_status`
- `evidence_level`

## Allowed values

### `data_source_type`
- `synthetic`
- `pilot_real_anchor`
- `mixed`
- `offline_dataset_evaluation`

### `runtime_type`
- `software_only`
- `offline_dataset_evaluation`
- `ros2_dataset_replay`
- `ros2_playback_grounded`
- `ros2_live_laptop_sensors`
- `ros2_live_windows_stream_wsl_core`
- `ros2_live_simulator`
- `ros2_live_robot`

### `model_status`
- `fully_runnable`
- `partially_runnable`
- `config_only`
- `optional_not_installed`

### `evidence_level`
- `framework_validation`
- `pilot_demonstration`
- `benchmark_preliminary`
- `deployment_not_claimed`

## Application rule
- Use these labels in summary tables.
- Use them in markdown results sections.
- Prefer them in figure captions and manifests.
- Do not publish unlabeled benchmark rows.
- Do not collapse laptop-sensor demos into a generic `ros2_live` claim.
