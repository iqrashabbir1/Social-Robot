# Monitoring And Outputs

## Active run

The current long benchmark run uses:
- output subdirectory: `ravdess_multialgorithm_case_study_ep1000`

## Monitoring files

### Live status JSON
`outputs/logs/paper1/ravdess_multialgorithm_case_study_ep1000/latest_status.json`

### Progress CSV
`outputs/logs/paper1/ravdess_multialgorithm_case_study_ep1000/training_progress_latest.csv`

### Standard output log
`outputs/logs/paper1/ravdess_multialgorithm_case_study_ep1000/stdout.log`

### Standard error log
`outputs/logs/paper1/ravdess_multialgorithm_case_study_ep1000/stderr.log`

## Commands to monitor

### Watch stdout
```powershell
Get-Content "C:\Users\masim\OneDrive\Desktop\Social Robot\outputs\logs\paper1\ravdess_multialgorithm_case_study_ep1000\stdout.log" -Wait
```

### Read latest status
```powershell
Get-Content "C:\Users\masim\OneDrive\Desktop\Social Robot\outputs\logs\paper1\ravdess_multialgorithm_case_study_ep1000\latest_status.json"
```

### Read the latest progress rows
```powershell
Import-Csv "C:\Users\masim\OneDrive\Desktop\Social Robot\outputs\logs\paper1\ravdess_multialgorithm_case_study_ep1000\training_progress_latest.csv" | Select-Object -Last 10
```

## Final output tables

Once the run completes, the main comparison tables are:

- `outputs/tables/paper1_table_multialgorithm_comparison.csv`
- `outputs/tables/paper1_table_multialgorithm_wide_comparison.csv`

These are the tables that should be used in the paper for:
- training-domain comparison
- held-out validation comparison
- external public-test comparison

## Main benchmark command

```powershell
python -u -m src.evaluation.run_multialgorithm_emotion_case_study `
  --project-root . `
  --train-dataset-root data/public/RAVDESS `
  --train-labels-csv data/public/RAVDESS/labels_broad4_angry.csv `
  --external-dataset-root data/public/CREMA-D `
  --external-labels-csv data/public/CREMA-D/labels_broad4_angry.csv `
  --output-subdir ravdess_multialgorithm_case_study_ep1000 `
  --target-label-set broad4_angry `
  --deep-epochs 1000 `
  --batch-size 64 `
  --device cuda `
  --log-every-epochs 10
```

