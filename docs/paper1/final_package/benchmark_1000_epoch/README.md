# 1000-Epoch Benchmark Folder

This folder is the dedicated reference for the long Paper 1 algorithm comparison run.

It explains:
- which algorithms are included
- which algorithms actually use epochs
- how the 1000-epoch run is executed
- how to monitor progress
- which final tables to use in the manuscript

## Important clarification

Not every algorithm uses epochs in a scientifically meaningful way.

### Epoch-based models
These models are trained iteratively over epochs:
- `cnn_small`
- `cnn_batchnorm`

### Non-epoch models
These models are fit directly and do not use neural-network epochs:
- `logistic_regression`
- `rbf_svm`
- `random_forest`
- `extra_trees`

### Hybrid model
This model is built after the base models are trained:
- `hybrid_soft_voting`

So the honest interpretation of the `1000`-epoch benchmark is:
- deep models train for `1000` epochs
- classical models are still fully trained and included in the same comparison
- the hybrid model is evaluated from the trained component models

## Main documents in this folder

- `benchmark_scope.md`
- `algorithm_policy.md`
- `monitoring_and_outputs.md`

