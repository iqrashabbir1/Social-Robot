# How To Prepare RAF-DB For Paper 1

## Goal
Prepare RAF-DB as the main controlled face-image dataset using the `happy/sad/neutral/angry` label space.

## Expected local layout
Place RAF-DB under:

`data/public/RAF-DB`

Typical RAF-DB layouts include:
- `Image/aligned/`
- `Image/original/`
- `EmoLabel/list_patition_label.txt`

## Generate the Paper 1 labels CSV
From the project root in PowerShell:

```powershell
cd "C:\Users\masim\OneDrive\Desktop\Social Robot"
.\.venv\Scripts\Activate.ps1
python .\scripts\prepare_rafdb_labels.py `
  --dataset-root "C:\Users\masim\OneDrive\Desktop\Social Robot\data\public\RAF-DB" `
  --output-csv "C:\Users\masim\OneDrive\Desktop\Social Robot\data\public\RAF-DB\labels_broad4_angry.csv" `
  --target-label-set broad4_angry
```

## Run offline evaluation on RAF-DB
```powershell
python -m src.evaluation.run_dataset_evaluation `
  --project-root . `
  --dataset-root "data/public/RAF-DB" `
  --labels-csv "data/public/RAF-DB/labels_broad4_angry.csv" `
  --split-mode train_test `
  --output-subdir rafdb_broad4_angry `
  --target-label-set broad4_angry
```

## Notes
- the current RAF mapping assumes the common basic-expression index ordering used by RAF-DB
- non-target classes such as surprise, fear, disgust, and contempt are dropped for this four-class Paper 1 setting
