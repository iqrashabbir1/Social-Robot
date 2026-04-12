# Paper 1 Benchmark Results Snapshot

This folder keeps the final lightweight benchmark snapshots that are safe to track on GitHub.

It is intended to help supervisors, collaborators, and paper reviewers follow the current Paper 1 comparison without navigating the full local `outputs/` tree.

## Included files

- `paper1_table_multialgorithm_comparison.csv`
- `paper1_table_multialgorithm_wide_comparison.csv`
- `paper1_table_local_vs_public_metrics.csv`
- `ravdess_multialgorithm_case_study_ep1000_latest_status.json`

## What these files represent

### Multi-algorithm benchmark
This is the long benchmark comparing:
- `logistic_regression`
- `rbf_svm`
- `random_forest`
- `extra_trees`
- `cnn_small`
- `cnn_batchnorm`
- `hybrid_soft_voting`

Training protocol:
- train on `RAVDESS`
- held-out validation on `20%` of `RAVDESS`
- external public test on `CREMA-D`

## Final long-run result

From the `1000`-epoch benchmark:

- best overall held-out validation model: `cnn_small`
- validation accuracy: `0.9781`
- validation macro F1: `0.9776`
- external public-test accuracy on `CREMA-D`: `0.2830`
- external public-test macro F1: `0.2507`

### Strongest classical external model
- `extra_trees`
- external macro F1: `0.2117`

### Hybrid model
- `hybrid_soft_voting`
- external accuracy: `0.2785`
- external macro F1: `0.2326`

## Main interpretation

The long-run benchmark shows that:
- the deep models improved substantially under long training
- `cnn_small` became the best model in the final benchmark
- the hybrid model remained competitive
- the classical models are still important baselines, but they did not exceed the best long-trained deep model on external test performance

## Main local source-of-truth paths

If you want the live local versions, use:

- `outputs/tables/paper1_table_multialgorithm_comparison.csv`
- `outputs/tables/paper1_table_multialgorithm_wide_comparison.csv`
- `outputs/logs/paper1/ravdess_multialgorithm_case_study_ep1000/latest_status.json`
- `outputs/logs/paper1/ravdess_multialgorithm_case_study_ep1000/training_progress_latest.csv`

