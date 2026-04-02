# CS3: Emotion Recognition Benchmark

## Goal
Benchmark baseline, classical, deep-fusion, and transformer-style models for the caregiving-robot setting.

## Benchmark Families
- `B0`: preserved existing visual baseline from the repository
- `B1`: classical SVM fusion baseline
- `B2`: deep late-fusion MLP
- `B3`: transformer-style lightweight cross-modal fusion

## Current Label Space
The repository currently supports the four classes:
- happy
- sad
- neutral
- fear

This reflects the actual preserved baseline evidence in `tests/emotion_log_labeled.csv`. The originally intended `angry` class remains a future data-collection target.

## Current Outputs
- `outputs/csv/cs3/model_performance_summary.csv`
- `outputs/csv/cs3/confusion_matrix_baseline.csv`
- `outputs/csv/cs3/confusion_matrix_deep.csv`
- `outputs/csv/cs3/confusion_matrix_transformer.csv`
- `outputs/csv/cs3/ablation_results.csv`
- `outputs/csv/cs3/training_curves.csv`
- `outputs/figures/cs3/model_comparison_barplot.png`
- `outputs/figures/cs3/confusion_matrix_baseline.png`
- `outputs/figures/cs3/confusion_matrix_deep.png`
- `outputs/figures/cs3/confusion_matrix_transformer.png`
- `outputs/figures/cs3/ablation_comparison.png`
- `outputs/figures/cs3/training_curves.png`

## Current Run Snapshot
- B0 accuracy: `0.8350`, macro F1: `0.8155`
- B1 accuracy: `1.0000`, macro F1: `1.0000`
- B2 accuracy: `0.9417`, macro F1: `0.7929`
- B3 accuracy: `0.9583`, macro F1: `0.9647`

## Interpretation
- B0 is the preserved implemented real baseline.
- B1, B2, and B3 are synthetic placeholder multimodal benchmarks used to validate the experiment and ablation pipeline.

## TODO for Real Completion
- collect aligned multimodal emotion data for the target caregiving setting
- add real multimodal labels across video, audio, and robot context
- re-run B1 through B3 on real synchronized windows
