# Figure Manifest

| Figure | Source CSV | Script | Output | Draft Caption |
|---|---|---|---|---|
| System architecture diagram | `outputs/csv/cs1/interface_spec.csv` | `src/visualization/plot_cs1.py` | `outputs/figures/cs1/system_architecture_diagram.png` | Paper 1 system architecture and ROS2 interface chain used for the digital-twin experiments. |
| CS1 latency distribution | `outputs/csv/cs1/latency_metrics.csv` | `src/visualization/plot_cs1.py` | `outputs/figures/cs1/latency_distribution.png` | End-to-end latency distribution across CS1 experiment modes. |
| CS1 synchronization error curve | `outputs/csv/cs1/sync_error_timeseries.csv` | `src/visualization/plot_cs1.py` | `outputs/figures/cs1/synchronization_error_over_time.png` | Synchronization error over time for simulator-only, control-loop, playback, and fault conditions. |
| CS1 task success comparison | `outputs/csv/cs1/latency_metrics.csv` | `src/visualization/plot_cs1.py` | `outputs/figures/cs1/task_success_comparison.png` | Task success rate across digital-twin experiment modes. |
| CS1 resource usage | `outputs/csv/cs1/latency_metrics.csv` | `src/visualization/plot_cs1.py` | `outputs/figures/cs1/resource_usage.png` | Mean CPU and memory usage by CS1 runtime mode. |
| CS2 modality availability heatmap | `outputs/csv/cs2/modality_availability.csv` | `src/visualization/plot_cs2.py` | `outputs/figures/cs2/modality_availability_heatmap.png` | Window-wise modality availability ratios for the synchronized multimodal pipeline. |
| CS2 synchronization quality comparison | `outputs/csv/cs2/sync_quality_metrics.csv` | `src/visualization/plot_cs2.py` | `outputs/figures/cs2/synchronization_quality_comparison.png` | Alignment-error comparison between nominal and missing-modality stress conditions. |
| CS2 missing-modality robustness | `outputs/csv/cs2/sync_quality_metrics.csv` | `src/visualization/plot_cs2.py` | `outputs/figures/cs2/missing_modality_robustness.png` | Availability-rate degradation under missing-modality stress. |
| CS3 model performance bar chart | `outputs/csv/cs3/model_performance_summary.csv` | `src/visualization/plot_cs3.py` | `outputs/figures/cs3/model_comparison_barplot.png` | Comparison of B0 through B3 across accuracy, macro F1, and weighted F1. |
| CS3 baseline confusion matrix | `outputs/csv/cs3/confusion_matrix_baseline.csv` | `src/visualization/plot_cs3.py` | `outputs/figures/cs3/confusion_matrix_baseline.png` | Confusion matrix for the preserved real visual baseline. |
| CS3 deep confusion matrix | `outputs/csv/cs3/confusion_matrix_deep.csv` | `src/visualization/plot_cs3.py` | `outputs/figures/cs3/confusion_matrix_deep.png` | Confusion matrix for the deep late-fusion benchmark. |
| CS3 transformer confusion matrix | `outputs/csv/cs3/confusion_matrix_transformer.csv` | `src/visualization/plot_cs3.py` | `outputs/figures/cs3/confusion_matrix_transformer.png` | Confusion matrix for the transformer-style fusion benchmark. |
| CS3 ablation comparison | `outputs/csv/cs3/ablation_results.csv` | `src/visualization/plot_cs3.py` | `outputs/figures/cs3/ablation_comparison.png` | Macro F1 comparison across modality ablations for B1 through B3. |
| CS3 training curves | `outputs/csv/cs3/training_curves.csv` | `src/visualization/plot_cs3.py` | `outputs/figures/cs3/training_curves.png` | Validation macro F1 across epochs for the deep and transformer-style models. |
