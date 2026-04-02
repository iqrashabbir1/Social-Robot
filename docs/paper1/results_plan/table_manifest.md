# Table Manifest

| Table | Source | Output | Purpose |
|---|---|---|---|
| System summary | `outputs/csv/cs1/interface_spec.csv`, `outputs/csv/cs2/session_metadata.csv`, `outputs/csv/cs3/model_performance_summary.csv` | `outputs/tables/paper1_table_system_summary.csv` | Summarize the Paper 1 experimental stack and tracked components. |
| Metrics summary | `outputs/csv/cs1/latency_summary.csv`, `outputs/csv/cs2/sync_quality_metrics.csv`, `outputs/csv/cs3/model_performance_summary.csv` | `outputs/tables/paper1_table_metrics_summary.csv` | Consolidate case-study metrics in a publication-friendly long format. |
| Ablation summary | `outputs/csv/cs3/ablation_results.csv` | `outputs/tables/paper1_table_ablation_summary.csv` | Summarize modality ablation performance for B1 through B3. |
