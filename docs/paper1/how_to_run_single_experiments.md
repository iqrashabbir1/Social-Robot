# How To Run Single Experiments

All commands below assume:

```powershell
cd "C:\Users\masim\OneDrive\Desktop\Social Robot"
.\.venv\Scripts\Activate.ps1
```

## CS1

### Simulator only
```powershell
python -m src.digital_twin.run_cs1 --project-root . --config configs/cs1/simulator_only.yaml
```

### Control loop
```powershell
python -m src.digital_twin.run_cs1 --project-root . --config configs/cs1/control_loop.yaml
```

### Playback
```powershell
python -m src.digital_twin.run_cs1 --project-root . --config configs/cs1/playback.yaml
```

### Fault injection
```powershell
python -m src.digital_twin.run_cs1 --project-root . --config configs/cs1/fault_injection.yaml
```

## CS2

### Video + audio
```powershell
python -m src.data.run_cs2 --project-root . --config configs/cs2/video_audio.yaml
```

### Video + audio + robot state
```powershell
python -m src.data.run_cs2 --project-root . --config configs/cs2/video_audio_robotstate.yaml
```

### Missing audio stress
```powershell
python -m src.data.run_cs2 --project-root . --config configs/cs2/missing_audio.yaml
```

### Missing video stress
```powershell
python -m src.data.run_cs2 --project-root . --config configs/cs2/missing_video.yaml
```

## CS3

### Baseline visual benchmark
```powershell
python -m src.models.classical.train_classical --project-root . --config configs/cs3/baseline_visual.yaml
```

### Classical models
```powershell
python -m src.models.classical.train_classical --project-root . --config configs/cs3/svm_video.yaml
python -m src.models.classical.train_classical --project-root . --config configs/cs3/svm_video_audio.yaml
python -m src.models.classical.train_classical --project-root . --config configs/cs3/rf_video_audio.yaml
python -m src.models.classical.train_classical --project-root . --config configs/cs3/xgboost_video_audio_context.yaml
```

### Deep models
```powershell
python -m src.models.deep.train_deep_fusion --project-root . --config configs/cs3/deep_fusion_video_audio.yaml
python -m src.models.deep.train_deep_fusion --project-root . --config configs/cs3/deep_fusion_video_audio_context.yaml
```

### Transformer models
```powershell
python -m src.models.transformer.train_transformer_fusion --project-root . --config configs/cs3/transformer_video_audio.yaml
python -m src.models.transformer.train_transformer_fusion --project-root . --config configs/cs3/transformer_video_audio_context.yaml
```

## Output layout
Each run writes to:
- `outputs/csv/<case-study>/<experiment-name>/`
- `outputs/figures/<case-study>/<experiment-name>/`
- `outputs/logs/<case-study-lower>/<experiment-name>/`

Each single experiment exports:
- `config_snapshot.yaml`
- `metrics.csv`
- `summary.json`
- figure-ready CSVs
- optional model artifact for trainable CS3 models
