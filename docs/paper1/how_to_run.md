# How to Run Paper 1

Paper 1 now prefers the refactored workflow:
- single experiment configs for CS1, CS2, and CS3
- `src/evaluation/benchmark_runner.py` only for CS3 aggregation

Use these companion docs:
- `docs/paper1/how_to_run_single_experiments.md`
- `docs/paper1/how_to_run_benchmarks.md`
- `docs/paper1/cs3_config_catalog.md`

## Quick start

```powershell
cd "C:\Users\masim\OneDrive\Desktop\Social Robot"
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Core Paper 1 commands

```powershell
python -m src.digital_twin.run_cs1 --project-root . --config configs/cs1/simulator_only.yaml
python -m src.data.run_cs2 --project-root . --config configs/cs2/video_audio_robotstate.yaml
python -m src.models.classical.train_classical --project-root . --config configs/cs3/svm_video_audio.yaml
python -m src.evaluation.benchmark_runner --project-root . --config configs/cs3/benchmark_all.yaml
python -m src.evaluation.export_results --project-root .
python -m src.visualization.generate_all_figures --project-root .
```

## Legacy note
`src/evaluation/ablation_runner.py` is still present for backward compatibility, but it is no longer the preferred Paper 1 path.
