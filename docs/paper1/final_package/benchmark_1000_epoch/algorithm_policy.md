# Algorithm Policy

## Why epochs do not apply equally to all methods

The benchmark contains mixed algorithm families.

### Deep neural models
Deep models learn iteratively and therefore use epochs.

In this benchmark:
- `cnn_small` runs for `1000` epochs
- `cnn_batchnorm` runs for `1000` epochs

### Classical machine-learning models
Classical models do not normally train with epochs. They are optimized through direct fitting, internal solver iterations, or tree-building procedures.

In this benchmark:
- `logistic_regression` is fit once using solver-based optimization
- `rbf_svm` is fit once using kernel optimization
- `random_forest` is fit once using the configured number of trees
- `extra_trees` is fit once using the configured number of trees

### Hybrid model
`hybrid_soft_voting` is not independently epoch-trained. It is constructed from the trained component models and then evaluated.

## Practical consequence for Paper 1

The correct wording in the paper is not:
- “every algorithm was trained for 1000 epochs”

The correct wording is:
- “the deep models were trained for 1000 epochs, while the classical baselines were fully fit using their native optimization procedures, and all models were evaluated under the same train/validation/test protocol.”

This is the technically correct and defensible statement.

