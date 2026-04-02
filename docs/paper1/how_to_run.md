# How to Run Paper 1

## Environment Setup
```powershell
cd D:\emotion_assistant
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run CS1
```powershell
.\.venv\Scripts\python.exe -m src.digital_twin.run_cs1 --project-root . --config configs/cs1/default.yaml
```

## Run CS2
```powershell
.\.venv\Scripts\python.exe -m src.data.run_cs2 --project-root . --config configs/cs2/default.yaml
```

## Run CS3
```powershell
.\.venv\Scripts\python.exe -m src.evaluation.ablation_runner --project-root . --config configs/cs3/default.yaml
```

## Export Paper 1 Tables
```powershell
.\.venv\Scripts\python.exe -m src.evaluation.export_results --project-root .
```

## Generate All Paper 1 Figures
```powershell
.\.venv\Scripts\python.exe -m src.visualization.generate_all_figures --project-root .
```

## Suggested Full Sequence
```powershell
.\.venv\Scripts\python.exe -m src.digital_twin.run_cs1 --project-root . --config configs/cs1/default.yaml
.\.venv\Scripts\python.exe -m src.data.run_cs2 --project-root . --config configs/cs2/default.yaml
.\.venv\Scripts\python.exe -m src.evaluation.ablation_runner --project-root . --config configs/cs3/default.yaml
.\.venv\Scripts\python.exe -m src.evaluation.export_results --project-root .
.\.venv\Scripts\python.exe -m src.visualization.generate_all_figures --project-root .
```
