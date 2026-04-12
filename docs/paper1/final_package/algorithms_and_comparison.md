# Algorithms And Comparison

## Benchmark goal

The benchmark is designed to answer two separate questions:

1. Which algorithm performs best on the held-out validation split of the training dataset?
2. Which algorithm generalizes best when tested on a different public dataset?

These are not always the same model.

## Datasets used

### Training and held-out validation
- Dataset: `RAVDESS`
- Split: `80% train / 20% held-out validation`

### External test
- Dataset: `CREMA-D`

### Shared label mapping
- `happy`
- `sad`
- `neutral`
- `angry`

## Algorithms included

### Classical algorithms
- `logistic_regression`
- `rbf_svm`
- `random_forest`
- `extra_trees`

### Deep algorithms
- `cnn_small`
- `cnn_batchnorm`

### Hybrid algorithm
- `hybrid_soft_voting`

The hybrid method is a soft-voting ensemble that combines the strongest classical and deep candidates from validation.

## Comparison table

| Algorithm | Family | Train Accuracy | Validation Accuracy | Validation Macro F1 | External Accuracy | External Macro F1 | Main role |
|---|---|---:|---:|---:|---:|---:|---|
| `logistic_regression` | classical | 1.0000 | 0.8596 | 0.8556 | 0.2655 | 0.1049 | linear classical baseline |
| `rbf_svm` | classical | 0.9940 | 0.9232 | 0.9200 | 0.2655 | 0.1049 | best held-out validation model |
| `random_forest` | classical | 1.0000 | 0.8991 | 0.8954 | 0.2455 | 0.1423 | strong tree baseline |
| `extra_trees` | classical | 1.0000 | 0.9013 | 0.8947 | 0.2581 | 0.2117 | best external macro-F1 model |
| `cnn_small` | deep | 0.9102 | 0.8772 | 0.8743 | 0.2633 | 0.1814 | best deep model |
| `cnn_batchnorm` | deep | 0.9918 | 0.8728 | 0.8711 | 0.2436 | 0.0979 | deeper normalized CNN baseline |
| `hybrid_soft_voting` | hybrid | - | 0.9167 | 0.9137 | 0.2666 | 0.1547 | ensemble comparison model |

## Main findings

### Best held-out validation model
- `rbf_svm`
- validation accuracy: `0.9232`
- validation macro F1: `0.9200`

### Best external-test model
- `extra_trees`
- external accuracy: `0.2581`
- external macro F1: `0.2117`

### Best deep model
- `cnn_small`
- validation accuracy: `0.8772`
- validation macro F1: `0.8743`
- external macro F1: `0.1814`

### Hybrid model result
- validation accuracy: `0.9167`
- validation macro F1: `0.9137`
- external accuracy: `0.2666`
- external macro F1: `0.1547`

## Interpretation

This comparison is valuable because it shows that:
- `rbf_svm` is strongest for in-domain validation
- `extra_trees` is strongest for external macro-F1 robustness
- `cnn_small` is the strongest deep model
- the hybrid model is competitive but not the best external model

So the paper should not report only a single “best algorithm” without context.

The correct interpretation is:
- best source-domain model: `rbf_svm`
- best external generalization model: `extra_trees`
- best deep model: `cnn_small`
- best ensemble comparison model: `hybrid_soft_voting`

## Recommended paper reporting

The manuscript should report all algorithms in one table and then explain that:
- model ranking changes under dataset shift
- local validation and external testing measure different properties
- the final preferred model depends on whether the paper prioritizes:
  - source-domain validation
  - external robustness
  - deep-learning extensibility

## Source tables

The main CSV files for this section are:
- `outputs/tables/paper1_table_multialgorithm_comparison.csv`
- `outputs/tables/paper1_table_multialgorithm_wide_comparison.csv`
- `outputs/tables/paper1_table_local_vs_public_metrics.csv`

