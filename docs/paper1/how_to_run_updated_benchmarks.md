# How To Run Updated Benchmarks

## Check XGBoost
```powershell
cd "C:\Users\masim\OneDrive\Desktop\Social Robot"
.\.venv\Scripts\Activate.ps1
python scripts\check_xgboost.py
```

## Run updated CS3 benchmark
```powershell
python -m src.evaluation.benchmark_runner --project-root . --config configs/cs3/benchmark_all.yaml
```

## Regenerate tables and figures
```powershell
python -m src.evaluation.export_results --project-root .
python -m src.visualization.generate_all_figures --project-root .
```

## Optional: rerun synthetic CS2
```powershell
python -m src.data.run_cs2 --project-root . --config configs/cs2/video_audio_robotstate.yaml
```
