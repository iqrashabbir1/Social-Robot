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

## Run CS3 Multi-Algorithm Comparison
This config compares several classical, deep, transformer, and hybrid candidates and writes an automatic ranking plus best-model summary.

```powershell
.\.venv\Scripts\python.exe -m src.evaluation.ablation_runner --project-root . --config configs/cs3/comparison.yaml
```

Key outputs:
- `outputs/csv/cs3/comparison_run/model_ranking.csv`
- `outputs/csv/cs3/comparison_run/best_model_summary.csv`
- `outputs/csv/cs3/comparison_run/training_curves.csv`

## Run CS3 on GPU
The repository now keeps two paths:
- standard CPU benchmark path
- optional GPU benchmark path for deep and transformer families

Classical models remain CPU-based even in the GPU run, while deep and transformer variants switch to the PyTorch CUDA implementation.

### 1. Install the GPU runtime
Install a CUDA-enabled PyTorch build that matches your machine. One common Windows command is:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

If your CUDA/PyTorch combination differs, adjust the PyTorch wheel source accordingly.

### 2. Run the GPU benchmark version
```powershell
.\.venv\Scripts\python.exe -m src.evaluation.ablation_runner --project-root . --config configs/cs3/comparison_gpu.yaml
```

### 3. Override epochs on GPU from the command line
```powershell
.\.venv\Scripts\python.exe -m src.evaluation.ablation_runner --project-root . --config configs/cs3/comparison_gpu.yaml --runtime-backend gpu --torch-device cuda --deep-epochs 150 --transformer-epochs 220 --ablation-epochs 40 --log-every 10 --output-subdir comparison_gpu_ep150_220
```

GPU outputs are written to:
- `outputs/csv/cs3/comparison_gpu_run/`
- or the folder given by `--output-subdir`

## Override Epochs from the Command
You can override the training length directly from PowerShell without editing YAML.

```powershell
.\.venv\Scripts\python.exe -m src.evaluation.ablation_runner --project-root . --config configs/cs3/comparison.yaml --deep-epochs 150 --transformer-epochs 220 --ablation-epochs 40 --output-subdir comparison_run_ep150_220
```

Available command-line overrides:
- `--deep-epochs`
- `--transformer-epochs`
- `--ablation-epochs`
- `--checkpoint-every`
- `--log-every`
- `--runtime-backend`
- `--torch-device`
- `--output-subdir`
- `--n-samples`

Tracking files generated during a run:
- `outputs/logs/paper1_cs3_<run_tag>_progress.jsonl`
- `outputs/logs/paper1_cs3_<run_tag>_latest_status.json`
- `outputs/csv/cs3/<run_tag>/epoch_progress.csv`

Tracked fields now include:
- current model id and algorithm
- current epoch and total epochs
- model elapsed time
- estimated remaining time for the current model
- total run elapsed time
- CPU usage
- memory usage
- GPU utilization and memory when `nvidia-smi` is available

You can watch live progress in PowerShell with:

```powershell
Get-Content "outputs\logs\paper1_cs3_comparison_run_ep150_220_progress.jsonl" -Wait
```

You can inspect the latest snapshot at any time with:

```powershell
Get-Content "outputs\logs\paper1_cs3_comparison_run_ep150_220_latest_status.json"
```

You can inspect epoch-by-epoch timing and resource metrics with:

```powershell
Import-Csv "outputs\csv\cs3\comparison_run_ep150_220\epoch_progress.csv" | Select-Object -Last 10
```

## Run CS3 Long Training
This path is intended for longer optimization runs of the synthetic Paper 1 multimodal benchmark without the full ablation sweep. The epoch counts are tunable directly in `configs/cs3/long_train.yaml`; they are no longer fixed to 1000.

```powershell
.\.venv\Scripts\python.exe -m src.evaluation.ablation_runner --project-root . --config configs/cs3/long_train.yaml
```

Long-training outputs are written to:
- `outputs/csv/cs3/long_train_comparison/`
- `outputs/logs/cs3_checkpoints/long_train_comparison/`

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
