# RAVDESS To CREMA-D Generalization Results

## Purpose
This note documents the current Paper 1 controlled evaluation path in which:
- a vision classifier is trained on a labeled `RAVDESS` subset with an internal 80/20 split
- the held-out 20 percent is used for local validation
- the trained model is then tested on a separate public dataset, `CREMA-D`

This gives a cleaner and more defensible comparison than ad hoc live camera snapshots.

## Label space
The current controlled 4-class mapping is:
- `happy`
- `sad`
- `neutral`
- `angry`

This mapping is referred to in code as `broad4_angry`.

## Datasets used

### Training and held-out validation
- Dataset: `RAVDESS`
- Local path: `data/public/RAVDESS`
- Generated labels CSV: `data/public/RAVDESS/labels_broad4_angry.csv`
- Download path: `scripts/download_ravdess_subset.py`

The current downloaded training set contains actors `01` through `08`.

### External public test
- Dataset: `CREMA-D`
- Local path: `data/public/CREMA-D`
- Generated labels CSV: `data/public/CREMA-D/labels_broad4_angry.csv`
- Download path: `scripts/download_cremad_subset.py`

The current downloaded external-test subset contains actors:
- `1001`
- `1002`
- `1003`
- `1004`
- `1005`
- `1006`
- `1007`
- `1008`

## Training run
- Model: `simple_emotion_cnn`
- Device: `cuda`
- Epochs: `1000`
- Batch size: `64`
- Image size: `128`
- Output subdir: `ravdess_train1000_cremad_external`

The long run completed successfully and saved its best checkpoint at:

`outputs/models/paper1/ravdess_train1000_cremad_external/best_model.pt`

## Monitoring added for long runs
The trainer now provides:
- live console logs for `epoch/total_epochs`
- live console logs for `step/steps_per_epoch`
- per-step loss reporting
- ETA reporting
- saved progress CSV
- saved latest-status JSON

Monitoring files from the completed run:
- `outputs/csv/paper1/ravdess_train1000_cremad_external/training_progress_latest.csv`
- `outputs/logs/paper1/ravdess_train1000_cremad_external/latest_status.json`
- `outputs/logs/paper1/ravdess_train1000_cremad_external/training_progress_events.csv`

The best local-validation epoch observed in this completed run was:
- epoch `896`
- validation accuracy `0.9539`
- validation macro F1 `0.9531`

## Main comparison table
Paper-ready comparison table:

`outputs/tables/paper1_table_local_vs_public_metrics.csv`

This table now includes:
- `train_dataset_name`
- `test_dataset_name`
- `evaluation_role`
- `accuracy`
- `macro_f1`
- `weighted_f1`
- `unweighted_recall`

## Current measured results

### Held-out local validation on RAVDESS
- samples: `456`
- accuracy: `0.9539`
- macro F1: `0.9531`
- weighted F1: `0.9539`
- unweighted recall: `0.9513`

### External public test on CREMA-D
- samples: `5386`
- accuracy: `0.2778`
- macro F1: `0.1544`
- weighted F1: `0.1594`
- unweighted recall: `0.2643`

## Interpretation for Paper 1
This is a useful and honest result for the paper.

What it shows:
- the model learns the `RAVDESS` training domain strongly
- the same model degrades substantially on `CREMA-D`
- this indicates that the current proposed vision baseline has limited cross-dataset generalization

Why this is still valuable:
- it is much stronger than using random room snapshots
- it gives a controlled public-dataset comparison
- it helps justify future multimodal adaptation, domain generalization, and broader benchmarking

## Exact commands used

### Download RAVDESS subset
```powershell
python .\scripts\download_ravdess_subset.py --output-root data/public/RAVDESS --actors 01 02 03 04 05 06 07 08
```

### Prepare RAVDESS labels
```powershell
python .\scripts\prepare_ravdess_labels.py --dataset-root data/public/RAVDESS --output-csv data/public/RAVDESS/labels_broad4_angry.csv --target-label-set broad4_angry
```

### Download CREMA-D subset
```powershell
python .\scripts\download_cremad_subset.py --output-root data/public/CREMA-D --actor-ids 1001 1002 1003 1004 1005 1006 1007 1008
```

### Prepare CREMA-D labels
```powershell
python .\scripts\prepare_cremad_labels.py --dataset-root data/public/CREMA-D --output-csv data/public/CREMA-D/labels_broad4_angry.csv --target-label-set broad4_angry
```

### Train on RAVDESS and test on CREMA-D
```powershell
python -m src.evaluation.run_cross_dataset_generalization `
  --project-root . `
  --train-dataset-root data/public/RAVDESS `
  --train-labels-csv data/public/RAVDESS/labels_broad4_angry.csv `
  --external-dataset "CREMA-D::data/public/CREMA-D::data/public/CREMA-D/labels_broad4_angry.csv" `
  --output-subdir ravdess_train1000_cremad_external `
  --target-label-set broad4_angry `
  --epochs 1000 `
  --batch-size 64 `
  --device cuda `
  --log-every-epochs 10 `
  --log-every-steps 10
```

## Next extension already prepared
Support for a second public video dataset has also been added:
- dataset support: `EmoryNLP`
- download script: `scripts/download_emorynlp_dataset.py`
- label builder: `scripts/prepare_emorynlp_labels.py`
- ROS replay config: `config/dataset_replay_emorynlp.yaml`

This second public dataset is intended for a later follow-up comparison after the current `RAVDESS -> CREMA-D` benchmark.
