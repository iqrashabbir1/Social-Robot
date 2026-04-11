# Table Manifest

## Runtime evidence tables

| Table | Output path | Source | Role |
|---|---|---|---|
| Runtime evidence summary | `outputs/tables/paper1_table_runtime_evidence_summary.csv` | `outputs/csv/paper1/ros2_runtime_verification.csv` plus export logic | Summarizes verified runtime components and their paper role |
| Hybrid metrics table | `outputs/tables/paper1_table_hybrid_metrics.csv` | `outputs/csv/paper1/hybrid_runtime_metrics.csv` | Holds measured or missing-source-marked hybrid runtime metrics |
| Runtime mode comparison | `outputs/tables/paper1_table_mode_comparison.csv` | `outputs/csv/paper1/runtime_mode_comparison.csv` | Positions playback, live laptop, and hybrid runtime modes |
| System summary | `outputs/tables/paper1_table_system_summary.csv` | `src/evaluation/export_results.py` | Updated high-level Paper 1 system summary including the hybrid baseline |
| Dataset summary | `outputs/tables/paper1_table_dataset_summary.csv` | `src/evaluation/run_dataset_evaluation.py` | Describes the dataset root, split mode, label availability, and sample count |
| Dataset metrics | `outputs/tables/paper1_table_dataset_metrics.csv` | `src/evaluation/run_dataset_evaluation.py` | Controlled perception metrics when labels exist; with the current tracked local unlabeled set this table should be treated as supplementary coverage/confidence metadata only |
| Runtime vs dataset evidence | `outputs/tables/paper1_table_runtime_vs_dataset_evidence.csv` | `src/evaluation/run_dataset_evaluation.py` | Explains why offline dataset evaluation and ROS dataset replay complement live runtime evidence |

## Labeling policy reminder
- `synthetic` means generated benchmark or simulated-only evidence
- `pilot real-anchor` means small local real-data capture
- `playback-grounded` means recorded-topic or emulated replay
- `mixed` means a table combines verified runtime metadata with offline or platform evidence
