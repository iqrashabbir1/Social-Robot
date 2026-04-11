# How To Run Dataset Evaluation

## Offline dataset evaluation from Windows

```powershell
cd "C:\Users\masim\OneDrive\Desktop\Social Robot"
.\.venv\Scripts\Activate.ps1
python -m src.evaluation.run_dataset_evaluation --project-root .
```

## Run on a specific dataset folder

```powershell
cd "C:\Users\masim\OneDrive\Desktop\Social Robot"
.\.venv\Scripts\Activate.ps1
python -m src.evaluation.run_dataset_evaluation `
  --project-root . `
  --dataset-root "C:\path\to\image_dataset" `
  --labels-csv "C:\path\to\labels.csv" `
  --split-mode train_test `
  --test-size 0.2 `
  --output-subdir custom_dataset_eval
```

## Outputs
- `outputs/csv/paper1/dataset_eval/dataset_predictions.csv`
- `outputs/csv/paper1/dataset_eval/dataset_metrics_summary.csv`
- `outputs/figures/paper1/dataset_prediction_panel.png`
- `outputs/figures/paper1/dataset_replay_sequence.png`
- `outputs/tables/paper1_table_dataset_summary.csv`
- `outputs/tables/paper1_table_dataset_metrics.csv`
