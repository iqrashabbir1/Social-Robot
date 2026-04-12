# Artifact Index

## Primary documents for writing

- `docs/paper1/final_package/paper1_analysis.md`
- `docs/paper1/final_package/algorithms_and_comparison.md`
- `docs/paper1/final_package/detailed_approach.md`

## Primary tables

### 1. Multi-algorithm comparison
- `outputs/tables/paper1_table_multialgorithm_comparison.csv`
- `outputs/tables/paper1_table_multialgorithm_wide_comparison.csv`

Use these to show:
- all algorithms
- training-domain performance
- held-out validation performance
- external-test performance

### 2. Source-to-external generalization
- `outputs/tables/paper1_table_local_vs_public_metrics.csv`

Use this to show:
- `RAVDESS` local validation
- `CREMA-D` external public test

### 3. Runtime evidence
- `outputs/tables/paper1_table_runtime_evidence_summary.csv`
- `outputs/tables/paper1_table_runtime_vs_dataset_evidence.csv`

Use these to explain:
- what is runtime evidence
- what is controlled dataset evidence

## Primary figures

### Runtime/platform figures
- `outputs/figures/paper1/paper_ready/hybrid_system_architecture_paper_ready.png`
- `outputs/figures/paper1/paper_ready/runtime_evidence_matrix_paper_ready.png`

### Controlled dataset/replay figure
- `outputs/figures/paper1/dataset_replay_sequence.png`

## Secondary figures

These are useful for internal discussion or supplementary material, but should be used only when the supporting logs are real and complete:
- `outputs/figures/paper1/ros2_runtime_verification.png`
- `outputs/figures/paper1/runtime_mode_comparison.png`
- `outputs/figures/paper1/dataset_sample_panel.png`
- `outputs/figures/paper1/dataset_prediction_panel.png`

## Supporting analysis documents

- `docs/paper1/multialgorithm_case_study_comparison.md`
- `docs/paper1/ravdess_cremad_generalization_results.md`
- `docs/paper1/dataset_vs_live_strategy.md`
- `docs/paper1/figure_manifest.md`
- `docs/paper1/table_manifest.md`
- `docs/paper1/final_package/results/README.md`

## Recommended paper assembly

For the manuscript, the cleanest structure is:

1. Use `paper1_analysis.md` for framing and contribution logic.
2. Use `algorithms_and_comparison.md` for the main benchmark section.
3. Use `detailed_approach.md` for methodology and reproducibility.
4. Use the primary tables and figures above for the main paper body.
