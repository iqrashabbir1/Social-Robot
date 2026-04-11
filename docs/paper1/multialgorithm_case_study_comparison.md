# Multi-Algorithm Emotion Case Study Comparison

## Purpose
This case study extends the single-model `RAVDESS -> CREMA-D` workflow into a multi-algorithm benchmark so Paper 1 can compare:
- training-domain fit on `RAVDESS`
- held-out validation performance on the `20%` local split
- external public-test performance on `CREMA-D`

The main goal is not only to identify the strongest model inside the training domain, but also to check whether the same model remains strong under cross-dataset shift.

## Case-study setup

### Training and validation dataset
- `RAVDESS`
- 4-class mapping: `happy`, `sad`, `neutral`, `angry`
- split protocol: `80% train / 20% held-out validation`

### External test dataset
- `CREMA-D`
- same 4-class mapping: `happy`, `sad`, `neutral`, `angry`

### Why this is important
This is a better Paper 1 evaluation setup than random live snapshots because:
- it uses controlled public datasets
- it compares multiple algorithm families
- it separates in-domain validation from cross-domain testing
- it allows us to see whether the proposed approach is merely fitting the training set or actually generalizing

## Algorithms included
The current comparison includes seven algorithms:

### Classical models
- `logistic_regression`
- `rbf_svm`
- `random_forest`
- `extra_trees`

### Deep models
- `cnn_small`
- `cnn_batchnorm`

### Hybrid model
- `hybrid_soft_voting`

The hybrid model is a soft-voting ensemble built from the best classical-validation model and the best deep-validation model.

## Selection policy
For the benchmark, every algorithm is tested externally, not just the best one.

This is important because:
- the best local-validation model is not always the best external-test model
- external testing provides a stronger view of model validity

So the workflow is:
1. train every candidate model
2. validate every model on the held-out `RAVDESS` split
3. test every model on `CREMA-D`
4. mark the best model by validation
5. separately inspect which model generalizes best externally

## Output tables

### Long-format paper table
- `outputs/tables/paper1_table_multialgorithm_comparison.csv`

This table has one row per algorithm per stage:
- `train_split`
- `held_out_validation`
- `external_public_test`

### Wide-format comparison table
- `outputs/tables/paper1_table_multialgorithm_wide_comparison.csv`

This table is easier to read in the paper because each algorithm has one row with:
- train accuracy
- validation accuracy
- external-test accuracy
- train macro F1
- validation macro F1
- external-test macro F1
- best-model flag

## Main measured result

### Best held-out validation model
The best validation model in this run was:
- `rbf_svm`
- held-out validation accuracy: `0.9232`
- held-out validation macro F1: `0.9200`

### Best external-test model
The strongest model on `CREMA-D` in this run was:
- `extra_trees`
- external-test accuracy: `0.2581`
- external-test macro F1: `0.2117`

### Best deep model
The strongest deep model in this run was:
- `cnn_small`
- held-out validation accuracy: `0.8772`
- held-out validation macro F1: `0.8743`
- external-test accuracy: `0.2633`
- external-test macro F1: `0.1814`

### Hybrid model
The hybrid soft-voting model achieved:
- held-out validation accuracy: `0.9167`
- held-out validation macro F1: `0.9137`
- external-test accuracy: `0.2666`
- external-test macro F1: `0.1547`

## Interpretation
This result is scientifically useful for Paper 1 because it shows three different things at once:

1. Some models fit the `RAVDESS` domain extremely well.
2. The ranking changes when the same trained models are tested on `CREMA-D`.
3. Strong in-domain validation does not automatically imply strong cross-dataset generalization.

The practical conclusion is:
- `rbf_svm` is the strongest model if the main priority is local held-out validation
- `extra_trees` is currently the strongest model if the main priority is external macro-F1 robustness on `CREMA-D`
- `cnn_small` is the strongest deep model and remains competitive externally

This means Paper 1 should not present only one “winner” without context. The manuscript should clearly distinguish:
- best in-domain model
- best externally generalizing model
- best deep model
- best hybrid model

## Recommended paper wording
Use language like this:

“A multi-algorithm comparison on the same `RAVDESS` training split showed that the best held-out validation model (`rbf_svm`) was not the same as the best external-test model on `CREMA-D` (`extra_trees`). This indicates that model ranking is sensitive to cross-dataset shift and supports reporting both in-domain and external-test evidence.”

## Exact command used
```powershell
python -m src.evaluation.run_multialgorithm_emotion_case_study `
  --project-root . `
  --train-dataset-root data/public/RAVDESS `
  --train-labels-csv data/public/RAVDESS/labels_broad4_angry.csv `
  --external-dataset-root data/public/CREMA-D `
  --external-labels-csv data/public/CREMA-D/labels_broad4_angry.csv `
  --output-subdir ravdess_multialgorithm_case_study `
  --target-label-set broad4_angry `
  --deep-epochs 80 `
  --batch-size 64 `
  --device cuda `
  --log-every-epochs 10
```

## Monitoring for longer runs
For single-model long training, use:
- `src.evaluation.run_cross_dataset_generalization`
- `src.models.vision.train_image_emotion_classifier`

These save:
- training progress CSV
- latest status JSON
- per-step console logs

Important monitoring files from the long `1000`-epoch run:
- `outputs/csv/paper1/ravdess_train1000_cremad_external/training_progress_latest.csv`
- `outputs/logs/paper1/ravdess_train1000_cremad_external/latest_status.json`
- `outputs/logs/paper1/ravdess_train1000_cremad_external/training_progress_events.csv`

## Recommended use in the paper
For the manuscript, the cleanest presentation is:

1. Use the multi-algorithm table to compare all candidate models.
2. Use the long `1000`-epoch monitored run as the final detailed training example.
3. Report both:
   - local held-out validation
   - external public-dataset testing

This makes the evaluation more realistic, more practical, and more technically defensible.
