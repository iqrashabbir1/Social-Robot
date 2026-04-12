# Benchmark Scope

## Objective

The objective of this run is to generate a strong Paper 1 comparison under a long-training regime while keeping all existing baseline algorithms.

The benchmark compares:
- source-domain fit on `RAVDESS`
- held-out validation on the `20%` split of `RAVDESS`
- external public-dataset testing on `CREMA-D`

## Datasets

### Training and validation
- Dataset: `RAVDESS`
- Split: `80% train / 20% held-out validation`
- Label space:
  - `happy`
  - `sad`
  - `neutral`
  - `angry`

### External public test
- Dataset: `CREMA-D`
- Same 4-class mapping:
  - `happy`
  - `sad`
  - `neutral`
  - `angry`

## Algorithms included

### Classical baselines
- `logistic_regression`
- `rbf_svm`
- `random_forest`
- `extra_trees`

### Deep baselines
- `cnn_small`
- `cnn_batchnorm`

### Hybrid
- `hybrid_soft_voting`

## Why this benchmark matters

This benchmark is stronger than a single-model report because it shows:
- whether one family dominates only on the training domain
- whether the same ranking holds under cross-dataset shift
- whether the best deep model matches the best classical model
- whether a hybrid combination improves practical robustness

