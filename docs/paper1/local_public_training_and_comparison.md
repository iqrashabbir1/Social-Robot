# Local Training And Public Dataset Comparison

## Goal
This update adds a separate real-dataset training path so Paper 1 can report:
- local training and validation metrics on an 80/20 split
- external testing on public datasets
- a comparison table that places both in one paper-ready view

## Why this is needed
The older CS3 benchmark path in this repo is still useful for synthetic placeholder benchmarking and orchestration logic, but it is not the same as training a real perception model on labeled emotion datasets.

This new path keeps the existing stable code intact and adds a clearer paper story:
1. train a real image-based classifier on a labeled dataset stored locally
2. validate it on the held-out 20 percent split
3. test the trained model on one or more public datasets
4. export one table for the manuscript

## Current recommended workflow

### Strategy A
Train on a labeled dataset stored locally, validate on the held-out 20 percent split, then test on one or more public datasets.

Use a labeled dataset stored locally. For immediate use, `RAVDESS` is already downloaded into:

`data/public/RAVDESS`

### Step 2
Train with a long run if desired, for example 1000 epochs:

```powershell
python -m src.models.vision.train_image_emotion_classifier `
  --project-root . `
  --dataset-root "data/public/RAVDESS" `
  --labels-csv "data/public/RAVDESS/labels_broad4_angry.csv" `
  --output-subdir local_ravdess_train_1000 `
  --target-label-set broad4_angry `
  --epochs 1000 `
  --batch-size 32 `
  --device cuda
```

### Step 3
Evaluate the trained model on a public dataset:

```powershell
python -m src.models.vision.evaluate_image_emotion_classifier `
  --project-root . `
  --model-path "outputs/models/paper1/local_ravdess_train_1000/best_model.pt" `
  --dataset-root "data/public/RAVDESS" `
  --labels-csv "data/public/RAVDESS/labels_broad4_angry.csv" `
  --output-subdir public_ravdess_eval `
  --target-label-set broad4_angry `
  --device cuda
```

For a second public dataset, repeat this command with a different dataset root and label CSV, for example a prepared `CREMA-D` or `RAF-DB` path.

### Step 4
Build the comparison table:

```powershell
python -m src.evaluation.build_local_public_comparison `
  --project-root . `
  --local-metrics-csv "outputs/csv/paper1/local_ravdess_train_1000/local_validation_metrics.csv" `
  --public-metrics-csv "outputs/csv/paper1/public_ravdess_eval/public_test_metrics.csv" `
  --comparison-group "local_train_then_public_test" `
  --output-path "outputs/tables/paper1_table_local_vs_public_metrics.csv"
```

### Strategy B
Train on one public dataset, then test on a different public dataset.

Example:
- train on `RAVDESS`
- test on `CREMA-D` or `RAF-DB` after those datasets are prepared locally

```powershell
python -m src.models.vision.train_image_emotion_classifier `
  --project-root . `
  --dataset-root "data/public/RAVDESS" `
  --labels-csv "data/public/RAVDESS/labels_broad4_angry.csv" `
  --output-subdir train_on_ravdess `
  --target-label-set broad4_angry `
  --epochs 1000 `
  --batch-size 32 `
  --device cuda

python -m src.models.vision.evaluate_image_emotion_classifier `
  --project-root . `
  --model-path "outputs/models/paper1/train_on_ravdess/best_model.pt" `
  --dataset-root "data/public/CREMA-D" `
  --labels-csv "data/public/CREMA-D/labels_broad4_angry.csv" `
  --output-subdir test_on_cremad `
  --target-label-set broad4_angry `
  --device cuda

python -m src.evaluation.build_local_public_comparison `
  --project-root . `
  --local-metrics-csv "outputs/csv/paper1/train_on_ravdess/local_validation_metrics.csv" `
  --public-metrics-csv "outputs/csv/paper1/test_on_cremad/public_test_metrics.csv" `
  --comparison-group "public_train_then_other_public_test" `
  --output-path "outputs/tables/paper1_table_public_to_public_metrics.csv"
```

## Main paper output
- `outputs/tables/paper1_table_local_vs_public_metrics.csv`

This table is intended to compare:
- local validation performance
- public dataset test performance

## Label requirement
- a true train/validation experiment requires labeled data
- the current small pilot room-image set is unlabeled, so it is not appropriate for supervised local training unless labels are added first

## Claim boundary
- good local validation does not guarantee cross-dataset generalization
- public-dataset testing is stronger than ad hoc live snapshots
- Paper 1 should still describe the full setup as preliminary and simulation-first plus dataset-grounded
